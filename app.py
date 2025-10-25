import os
import json
import ast
import re
import logging
import inspect
import tempfile
import shutil
import hashlib
import textwrap

import streamlit as st
import pandas as pd
import openai
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

from helper_function import (
    get_description_list,
    get_description_llm,
    get_modeling_llm,
    get_llm
)
# NOTE: We keep all the neo4j_runway imports
from neo4j_runway import Discovery, GraphDataModeler, PyIngest
try:
    # FIX: Re-importing the Config Generator, as it is essential
    from neo4j_runway.code_generation import PyIngestConfigGenerator
except Exception:
    PyIngestConfigGenerator = None 

from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from prompts import CHAT_PROMPT 

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = ".cache_insight_graph"
os.makedirs(CACHE_DIR, exist_ok=True)

# Session state defaults
if 'disc' not in st.session_state:
    st.session_state['disc'] = None
if 'gdm' not in st.session_state:
    st.session_state['gdm'] = None
if 'neo4j_creds' not in st.session_state:
    st.session_state['neo4j_creds'] = {}
if 'ingestion_done' not in st.session_state:
    st.session_state['ingestion_done'] = False
if 'kg_schema' not in st.session_state:
    st.session_state['kg_schema'] = None
if 'current_hash' not in st.session_state:
    st.session_state['current_hash'] = None


# ----------------- Utilities -----------------
# NOTE: All utility functions (hashing, caching, graphviz) are fine.
def df_hash(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return hashlib.md5(csv_bytes).hexdigest()

def cache_path_for(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def save_cache(key: str, data: dict):
    path = cache_path_for(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_cache(key: str):
    path = cache_path_for(key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def safe_extract_json_from_string(s: str):
    if not isinstance(s, str):
        return None
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None

def get_dot_from_model(model):
    if model is None:
        return None
    method_names = ["visualize", "to_dot", "to_graphviz", "to_dot_string", "render", "__str__"]
    for name in method_names:
        if hasattr(model, name):
            attr = getattr(model, name)
            try:
                maybe = attr() if callable(attr) else attr
            except Exception as e:
                logger.debug("Calling %s on model raised %s", name, e)
                maybe = None
            if maybe:
                try:
                    if hasattr(maybe, "source"):
                        return str(maybe.source)
                except Exception:
                    pass
                if isinstance(maybe, str):
                    return maybe
                if isinstance(maybe, bytes):
                    try:
                        return maybe.decode("utf-8")
                    except Exception:
                        return str(maybe)
                try:
                    return str(maybe)
                except Exception:
                    continue
    return None

def display_model_graphviz(model):
    dot = get_dot_from_model(model)
    if dot:
        try:
            st.graphviz_chart(dot, use_container_width=True)
            return True
        except Exception as e:
            logger.debug("graphviz_chart failed: %s", e)
    st.warning("Model visualization not available. Showing model object for debugging.")
    try:
        st.write(model)
    except Exception:
        st.write(repr(model))
    return False

def visualize_graph(driver):
    st.subheader("Graph Visualization (Sample)")
    try:
        with driver.session() as session:
            check = session.run("MATCH (n) RETURN count(n) AS count")
            if check.single()["count"] == 0:
                st.warning("Database is empty. No graph to visualize.")
                return

            results = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
            net = Network(height="750px", width="100%", notebook=False, directed=True)
            nodes_seen = set()
            for record in results:
                n, r, m = record["n"], record["r"], record["m"]
                if n.id not in nodes_seen:
                    nodes_seen.add(n.id)
                    net.add_node(n.id, label=next(iter(n.labels), "Node"), title=str(dict(n.items())))
                if m.id not in nodes_seen:
                    nodes_seen.add(m.id)
                    net.add_node(m.id, label=next(iter(m.labels), "Node"), title=str(dict(m.items())))
                net.add_edge(n.id, m.id, label=r.type)

            if not nodes_seen:
                # FIX: This is the error you are seeing. It's correct because the
                # graph build failed and only :Record nodes were created.
                st.info("Ingestion complete, but no relationships were found to visualize.")
                return

            net.show("graph.html")
            with open("graph.html", "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=800, width=1000)
    except Exception as e:
        st.error(f"Error connecting to Neo4j or visualizing graph: {e}")


def direct_ingest_to_neo4j(uri, username, password, df: pd.DataFrame, label: str = "Record", batch_size: int = 500):
    """
    Fallback ingestion: directly write each row as a node with label `label`.
    WARNING: This does NOT create the graph structure, only disconnected nodes.
    """
    logger.warning("Using fallback ingestion. This will only create disconnected :Record nodes.")
    cols = list(df.columns)
    safe_cols = [re.sub(r'\W+', '_', c).strip('_') or f"col_{i}" for i, c in enumerate(cols)]
    mapping = dict(zip(cols, safe_cols))

    rows = []
    for idx, row in df.iterrows():
        rowdict = {}
        for orig_col, safe_col in mapping.items():
            val = row[orig_col]
            if pd.isna(val):
                rowdict[safe_col] = None
            else:
                if isinstance(val, (pd.Timestamp,)):
                    rowdict[safe_col] = str(val)
                elif isinstance(val, (pd.Series, pd.DataFrame)):
                    rowdict[safe_col] = str(val)
                else:
                    rowdict[safe_col] = val
        rowdict["_csv_index"] = int(idx)
        rows.append(rowdict)

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            # Clear database first
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing database before fallback ingestion.")
            
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(
                    "UNWIND $rows AS row "
                    f"MERGE (n:{label} {{_csv_index: row._csv_index}}) "
                    "SET n += row",
                    {"rows": batch}
                )
    finally:
        driver.close()


# ==================================================================
# !! FIX: Re-adding the robust YAML generator from your original code !!
# This is necessary to handle the library version mismatches.
# ==================================================================
def generate_py_ingest_yaml_robust(data_model, username, password, uri, database, dataframe, preferred_csv_name="data.csv"):
    """
    Attempt to construct a PyIngestConfigGenerator and get a YAML string.
    Returns (yaml_string, tmp_dir_or_None). tmp_dir must be cleaned by caller.
    """
    tmp_dir = None
    
    if PyIngestConfigGenerator is None:
        raise RuntimeError("PyIngestConfigGenerator not importable from neo4j_runway.code_generation.")

    if data_model is None:
         raise ValueError("DataModel object is None. Cannot generate YAML.")
         
    # This was the check that failed before. The `gdm.current_model` is likely
    # a different class type, but it might have the .to_dict() method.
    # We will rely on the constructor to fail if the model is truly incompatible.
    
    # Try to get constructor signature
    try:
        sig = inspect.signature(PyIngestConfigGenerator)
        params = set(sig.parameters.keys())
    except Exception:
        # Fallback if signature check fails
        params = {"data_model", "username", "password", "uri", "database", "csv_name"}

    base_kwargs = {
        "data_model": data_model,
        "username": username,
        "password": password,
        "uri": uri,
        "database": database,
    }

    attempts = []
    
    # Attempt 1: Simple, assumes csv_name is a parameter
    k = dict(base_kwargs)
    k["csv_name"] = preferred_csv_name
    attempts.append(k)

    # Attempt 2: More complex, for versions that need a CSV *directory*
    if "csv_dir" in params or "file_directory" in params or "csv_directory" in params:
        tmp_dir = tempfile.mkdtemp(prefix="pyingest_tmp_")
        csv_path = os.path.join(tmp_dir, preferred_csv_name)
        try:
            dataframe.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"Failed to write temporary CSV for PyIngest: {e}")
            shutil.rmtree(tmp_dir)
            tmp_dir = None # Don't return a bad dir
            pass # We can still try the other constructors
        
        if tmp_dir: # Only add this attempt if CSV was written
            k = dict(base_kwargs)
            if "csv_dir" in params: k["csv_dir"] = tmp_dir
            if "file_directory" in params: k["file_directory"] = tmp_dir
            if "csv_directory" in params: k["csv_directory"] = tmp_dir
            if "csv_name" in params: k["csv_name"] = preferred_csv_name
            if "source_name" in params: k["source_name"] = preferred_csv_name
            attempts.append({kk: vv for kk, vv in k.items() if vv is not None})

    # Attempt 3: Base kwargs only
    attempts.append(base_kwargs)

    last_error = None
    tried_list = []

    for kwargs in attempts:
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        tried_list.append(kwargs.keys())
        try:
            gen = PyIngestConfigGenerator(**kwargs)
            
            # Found a valid constructor, now get the YAML string
            yaml_config = None
            methods_to_try = [
                "generate_pyingest_yaml_string",
                "generate_config_string",
                "generate_pyingest_yaml",
                "generate_yaml_string",
            ]
            for name in methods_to_try:
                if hasattr(gen, name):
                    try:
                        meth = getattr(gen, name)
                        result_val = meth()
                        if isinstance(result_val, str) and result_val.strip():
                            yaml_config = result_val
                            break
                    except Exception:
                        continue # Try next method
            
            if yaml_config:
                return yaml_config, tmp_dir # SUCCESS
            else:
                last_error = RuntimeError(f"Constructed {type(gen)} but failed to extract YAML string.")
                continue

        except TypeError as te:
            last_error = te # Likely a constructor mismatch, try next
            continue
        except Exception as e:
            last_error = e # A more serious error
            logger.exception(f"PyIngestConfigGenerator failed with {kwargs.keys()}")
            break # Stop trying

    # If we loop through all and fail
    if tmp_dir:
        try: shutil.rmtree(tmp_dir)
        except Exception: pass
        
    raise TypeError(f"Unable to construct PyIngestConfigGenerator or get YAML. Tried kwargs: {tried_list}. Last error: {last_error}")


# ------------------- Streamlit app ------------------------------------------

def load_openai_key():
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        openai.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        st.success("API key loaded for this session.")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")
    return api_key

def main():
    st.title("Insight Graph 🧠: CSV → Knowledge Graph")

    api_key = load_openai_key()
    if not api_key:
        st.stop()

    st.sidebar.title("Upload Your CSV")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is None:
        st.info("Upload a CSV to start.")
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("1) Uploaded Data (preview)")
    st.dataframe(df.head(50))

    data_hash = df_hash(df)
    
    # FIX: Clear session state if a new file is uploaded
    if st.session_state['current_hash'] != data_hash:
        st.info("New file detected. Clearing previous models.")
        st.session_state['disc'] = None
        st.session_state['gdm'] = None
        st.session_state['current_hash'] = data_hash


    # Column descriptions (file-caching is fine for this)
    desc_cache_key = f"{data_hash}_descriptions"
    descriptions = load_cache(desc_cache_key)
    if descriptions:
        st.info("Reused cached column descriptions.")
    else:
        with st.spinner("Generating column descriptions (LLM call - cached after first run)..."):
            descriptions = get_description_list(df, api_key, model="gpt3.5")
            save_cache(desc_cache_key, descriptions)
            st.success("Descriptions cached.")
    st.write("### Column Descriptions")
    st.json(descriptions)

    # ==================================================================
    # !! Step 2: Discovery (Using st.session_state) !!
    # ==================================================================
    st.subheader("2) Discovery Output")
    
    if st.session_state.get('disc') is None:
        st.info("Running Discovery (LLM call)...")
        with st.spinner("Running neo4j-runway Discovery (may call an LLM once)..."):
            try:
                disc_llm = get_description_llm()
                disc_obj = Discovery(llm=disc_llm, user_input=descriptions, data=df)
                disc_obj.run()
                st.session_state['disc'] = disc_obj
                st.success("Discovery complete.")
            except Exception as e:
                st.error(f"Discovery.run() failed: {e}")
                logger.exception("Discovery.run() raised")
                st.session_state['disc'] = None 
    else:
        st.info("Reusing cached Discovery object (no LLM call).")

    disc_obj_from_session = st.session_state.get('disc')
    if disc_obj_from_session:
        disc_content_to_display = getattr(disc_obj_from_session, "discovery", "Discovery object found, but text is empty.")
        if isinstance(disc_content_to_display, (dict, list)):
            st.json(disc_content_to_display)
        elif isinstance(disc_content_to_display, str):
            st.text(disc_content_to_display)
        else:
            st.write(disc_content_to_display)
    else:
        st.warning("Discovery step failed or was skipped.")


    # ==================================================================
    # !! Step 3: GraphDataModeler (Using st.session_state) !!
    # ==================================================================
    st.header("3) Graph Data Model")
    
    if st.session_state.get('gdm') is None:
        if disc_obj_from_session:
            st.info("Running Graph Data Modeler (LLM call)...")
            with st.spinner("Generating initial graph model (may call LLM)..."):
                try:
                    modeling_llm = get_modeling_llm()
                    gdm = GraphDataModeler(
                        llm=modeling_llm,
                        discovery=disc_obj_from_session 
                    )
                    gdm.create_initial_model()
                    st.session_state['gdm'] = gdm
                    st.success("Graph model created.")
                except Exception as e:
                    st.error(f"GraphDataModeler failed: {e}")
                    logger.exception("GraphDataModeler.create_initial_model failed")
                    st.session_state['gdm'] = None
        else:
            st.warning("Cannot create Graph Model because Discovery step failed.")
    else:
        st.info("Reusing cached Graph Data Model (no LLM call).")

    gdm_obj_from_session = st.session_state.get('gdm')
    if gdm_obj_from_session:
        display_model_graphviz(getattr(gdm_obj_from_session, "current_model", None))
    else:
        st.warning("Graph Model object not available.")

    # Model refinement iteration
    feedback = st.radio("Refine model?", ("No", "Yes"))
    if feedback == "Yes":
        suggestions = st.text_area("Enter model refinement instructions:", height=150)
        if st.button("Apply iteration"):
            if st.session_state.get('gdm') is None:
                st.error("Graph Model object not found. Cannot iterate.")
            else:
                with st.spinner("Iterating model (LLM call)..."):
                    try:
                        st.session_state['gdm'].iterate_model(user_corrections=suggestions)
                        st.success("Model updated.")
                        display_model_graphviz(getattr(st.session_state['gdm'], "current_model", None))
                    except Exception as e:
                        st.error(f"Iteration failed: {e}")

    # ==================================================================
    # !! Step 4: Neo4j Ingestion (FIXED) !!
    # We now use the robust YAML generator, as the articles imply.
    # ==================================================================
    st.header("4) Neo4j Ingestion")
    st.subheader("Enter Neo4j credentials")
    with st.form("neo4j_credentials"):
        username = st.text_input("Neo4j username", value="neo4j")
        password = st.text_input("Neo4j password", type="password", value="StrongPassword123") # Use your own password
        uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        database = st.text_input("Neo4j database", value="neo4j")
        submitted = st.form_submit_button("Clear Database & Ingest Data")

    if submitted:
        st.session_state['neo4j_creds'] = {"uri": uri, "username": username, "password": password, "database": database}

        gdm_model = getattr(st.session_state.get('gdm'), "current_model", None)
        ingestion_succeeded = False
        yaml_config = None
        maybe_tmp_dir = None

        if gdm_model is None:
            st.error("Graph Data Model is missing. Cannot proceed with ingestion.")
        else:
            # --- Primary Ingestion Attempt: Generate YAML ---
            try:
                st.info("Attempt 1: Generating PyIngest YAML config...")
                with st.spinner("Generating config..."):
                    yaml_config, maybe_tmp_dir = generate_py_ingest_yaml_robust(
                        data_model=gdm_model,
                        username=username,
                        password=password,
                        uri=uri,
                        database=database,
                        dataframe=df
                    )
                st.subheader("Generated Ingestion YAML")
                st.code(yaml_config, language="yaml")

            except Exception as e1:
                st.warning(f"YAML config generation failed: {e1}")
                logger.warning("generate_py_ingest_yaml_robust failed", exc_info=True)
                yaml_config = None # Ensure it's None

            # --- Second Ingestion Attempt: Run PyIngest with YAML ---
            if yaml_config:
                try:
                    st.info("Attempt 2: Ingesting with generated YAML...")
                    with st.spinner("Running ingestion via PyIngest..."):
                        # This constructor (config string + dataframe) matches the Medium article
                        ingestor = PyIngest(config=yaml_config, dataframe=df)
                        ingestor.ingest_data() # Call ingest_data with no args
                    st.success("Ingestion via PyIngest (YAML) completed.")
                    ingestion_succeeded = True
                except TypeError:
                    try:
                        # Fallback: maybe ingest_data needs the df
                        st.info("Retrying with different PyIngest signature...")
                        ingestor = PyIngest(config=yaml_config)
                        ingestor.ingest_data(dataframe=df)
                        st.success("Ingestion via PyIngest (YAML) completed.")
                        ingestion_succeeded = True
                    except Exception as e2:
                        st.warning(f"YAML Ingestion failed: {e2}")
                        logger.warning("PyIngest(config=...) failed", exc_info=True)
                except Exception as e2:
                    st.warning(f"YAML Ingestion failed: {e2}")
                    logger.warning("PyIngest(config=...) failed", exc_info=True)

            # --- Fallback Ingestion Attempt (Your original fallback) ---
            if not ingestion_succeeded:
                st.error("Graph model ingestion (YAML path) failed critically.")
                st.info("Attempt 3: Falling back to 'Direct Ingestion' (will create flat nodes, not a graph).")
                try:
                    with st.spinner("Directly writing CSV rows into Neo4j (fallback)..."):
                        direct_ingest_to_neo4j(uri=uri, username=username, password=password, df=df, label="Record")
                    
                    st.success("Direct ingestion completed (nodes created as :Record).")
                    st.error("WARNING: The graph structure failed to build. "
                            "Data was loaded as disconnected ':Record' nodes. "
                            "The chat functionality will likely not work as expected.")
                    ingestion_succeeded = True # It "succeeded" in writing *something*
                
                except Exception as e3:
                    st.error(f"Direct ingestion fallback ALSO failed: {e3}")
                    logger.exception("direct_ingest_to_neo4j failed")
                    ingestion_succeeded = False

        st.session_state['ingestion_done'] = ingestion_succeeded
        
        # Cleanup temp dir
        if maybe_tmp_dir:
            try: shutil.rmtree(maybe_tmp_dir)
            except Exception: pass

        # Visualize graph and refresh schema (if ingestion succeeded)
        if st.session_state['ingestion_done']:
            try:
                driver = GraphDatabase.driver(uri, auth=(username, password))
                visualize_graph(driver)
                
                kg = Neo4jGraph(url=uri, username=username, password=password, database=database)
                try:
                    kg.refresh_schema()
                    st.session_state['kg_schema'] = kg.schema
                    logger.info(f"Refreshed schema: {kg.schema}")
                except Exception as e:
                    st.warning(f"Schema refresh failed: {e}")
                
                driver.close()
            except Exception as e:
                st.error(f"Error connecting to Neo4j after ingestion: {e}")

    # ==================================================================
    # !! Step 5: Chat (FIXED) !!
    # ==================================================================
    if st.session_state.get('ingestion_done', False):
        st.header("5) Chat with your Knowledge Graph")
        creds = st.session_state['neo4j_creds']
        
        try:
            kg = Neo4jGraph(url=creds['uri'], username=creds['username'], password=creds['password'], database=creds['database'])
            kg.schema = st.session_state.get('kg_schema', 'No schema cached') 
        except Exception as e:
            st.error(f"Failed to initialize Neo4jGraph: {e}")
            st.stop()
            
        st.subheader("Graph schema")
        schema_text = st.session_state.get('kg_schema', 'No schema cached or database is empty.')
        st.text(textwrap.fill(str(schema_text), 80))

        if "No schema" in schema_text or not schema_text:
             st.warning("Schema is empty. Chat will not work. This likely means the graph ingestion failed to create a proper graph.")
        else:
            chat_llm = get_llm("gpt3.5", api_key)
            
            cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=CHAT_PROMPT)
            
            try:
                # ======================================================
                # FIX: Add allow_dangerous_requests=True
                # ======================================================
                cypher_chain = GraphCypherQAChain.from_llm(
                    chat_llm, 
                    graph=kg, 
                    verbose=True, 
                    cypher_prompt=cypher_prompt,
                    allow_dangerous_requests=True 
                )
            except Exception as e:
                # This is where your 'allow_dangerous_requests' error appeared
                st.error(f"Failed to build GraphCypherQAChain: {e}")
                logger.exception("Chain creation failed")
                cypher_chain = None

            query = st.text_area("Ask a question about your data:", height=100)
            if query and cypher_chain:
                with st.spinner("Generating Cypher and answering..."):
                    try:
                        answer = cypher_chain.invoke(query)
                        st.subheader("Answer")
                        st.write(answer.get('result', 'No answer found.'))
                        
                        st.subheader("Generated Cyfpher")
                        st.code(answer.get('generated_cypher', 'Could not display query.'), language='cypher')
                        
                    except Exception as e:
                        st.error(f"Query error: {e}")
                        logger.exception("Cypher chain query failed")

if __name__ == "__main__":
    main()