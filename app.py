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
# FIX: Import Neo4j error classes to catch specific exceptions
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError 
from pyvis.network import Network
import streamlit.components.v1 as components

from helper_function import (
    get_description_list,
    get_description_llm,
    get_modeling_llm,
    get_llm
)
# Correct imports for v0.4.3
from neo4j_runway import Discovery, GraphDataModeler, LLM, IngestionGenerator, PyIngest

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
# NOTE: Hashing, caching, graphviz utilities remain the same.
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
    # ... (code unchanged) ...
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
    # ... (code unchanged) ...
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
    # ... (code unchanged) ...
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

def visualize_graph(driver, database): # FIX: Added database parameter
    st.subheader("Graph Visualization (Sample)")
    try:
        # FIX: Use the specified database for the session
        with driver.session(database=database) as session:
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
                st.info("Ingestion complete, but no relationships were found to visualize (this often happens with the fallback ingestion).") # Modified message
                return

            net.show("graph.html")
            with open("graph.html", "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=800, width=1000)
    except Exception as e:
        st.error(f"Error connecting to Neo4j or visualizing graph: {e}")

# ==================================================================
# !! FIX: Add 'database' parameter to the fallback function !!
# Also add index creation.
# ==================================================================
def direct_ingest_to_neo4j(uri, username, password, database, df: pd.DataFrame, label: str = "Record", batch_size: int = 500):
    """
    Fallback ingestion: directly write each row as a node with label `label`.
    WARNING: This does NOT create the graph structure, only disconnected nodes.
    Now includes database parameter and index creation.
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
            # FIX: Handle potential numpy types more robustly
            elif hasattr(val, 'item'): # Checks for numpy types like int64, float64
                 rowdict[safe_col] = val.item()
            elif isinstance(val, (pd.Timestamp,)):
                rowdict[safe_col] = str(val)
            elif isinstance(val, (pd.Series, pd.DataFrame)):
                 rowdict[safe_col] = str(val) # Should ideally not happen per row
            else:
                rowdict[safe_col] = val
        rowdict["_csv_index"] = int(idx) # Ensure _csv_index is standard int
        rows.append(rowdict)

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            # Clear database first
            logger.info(f"Clearing database '{database}' before fallback ingestion...")
            session.run("MATCH (n) DETACH DELETE n")

            # Create index for faster lookups later (optional but good practice)
            logger.info(f"Creating index on :{label}(_csv_index) if it doesn't exist...")
            try:
                session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n._csv_index)")
            except ClientError as ce:
                # Ignore index creation errors if it already exists in another form, etc.
                logger.warning(f"Could not create index on :{label}(_csv_index): {ce.message}")


            logger.info(f"Ingesting {len(rows)} rows as :{label} nodes...")
            # Insert in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (n:{label} {{_csv_index: row._csv_index}})
                    SET n += row
                    """,
                    rows=batch
                )
            logger.info("Fallback ingestion batch complete.")
    finally:
        driver.close()


# --- Streamlit app ---

def load_openai_key():
    # ... (code unchanged) ...
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

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    st.subheader("1) Uploaded Data (preview)")
    st.dataframe(df.head(50))

    data_hash = df_hash(df)

    if st.session_state.get('current_hash') != data_hash:
        st.info("New file detected. Clearing previous models.")
        st.session_state['disc'] = None
        st.session_state['gdm'] = None
        st.session_state['ingestion_done'] = False # Reset ingestion status
        st.session_state['kg_schema'] = None
        st.session_state['current_hash'] = data_hash


    # Column descriptions
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

    # --- Step 2: Discovery ---
    st.subheader("2) Discovery Output")
    disc_obj_from_session = st.session_state.get('disc')
    if disc_obj_from_session is None:
        st.info("Running Discovery (LLM call)...")
        with st.spinner("Running neo4j-runway Discovery..."):
            try:
                disc_llm = get_description_llm()
                # Ensure user_input is passed correctly if needed by this version
                disc_obj = Discovery(llm=disc_llm, user_input=descriptions, data=df)
                disc_obj.run()
                st.session_state['disc'] = disc_obj
                st.success("Discovery complete.")
                disc_obj_from_session = disc_obj # Use the newly created object
            except Exception as e:
                st.error(f"Discovery.run() failed: {e}")
                logger.exception("Discovery.run() raised")
                st.session_state['disc'] = None
    else:
        st.info("Reusing cached Discovery object (no LLM call).")

    # Display Discovery output
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

    # --- Step 3: GraphDataModeler ---
    st.header("3) Graph Data Model")
    gdm_obj_from_session = st.session_state.get('gdm')
    if gdm_obj_from_session is None:
        if disc_obj_from_session:
            st.info("Running Graph Data Modeler (LLM call)...")
            with st.spinner("Generating initial graph model..."):
                try:
                    modeling_llm = get_modeling_llm()
                    gdm = GraphDataModeler(
                        llm=modeling_llm,
                        discovery=disc_obj_from_session
                    )
                    gdm.create_initial_model()
                    st.session_state['gdm'] = gdm
                    st.success("Graph model created.")
                    gdm_obj_from_session = gdm # Use the newly created object
                except Exception as e:
                    st.error(f"GraphDataModeler failed: {e}")
                    logger.exception("GraphDataModeler.create_initial_model failed")
                    st.session_state['gdm'] = None
        else:
            st.warning("Cannot create Graph Model because Discovery step failed.")
    else:
        st.info("Reusing cached Graph Data Model (no LLM call).")

    # Display Graph Model
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

    # --- Step 4: Neo4j Ingestion ---
    st.header("4) Neo4j Ingestion")
    st.subheader("Enter Neo4j credentials")
    with st.form("neo4j_credentials"):
        username = st.text_input("Neo4j username", value="neo4j")
        password = st.text_input("Neo4j password", type="password", value="StrongPassword123")
        uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        database = st.text_input("Neo4j database", value="neo4j")
        submitted = st.form_submit_button("Clear Database & Ingest Data")

    if submitted:
        st.session_state['neo4j_creds'] = {"uri": uri, "username": username, "password": password, "database": database}

        gdm_model = getattr(st.session_state.get('gdm'), "current_model", None)
        ingestion_succeeded = False
        primary_ingestion_error = None

        if gdm_model is None:
            st.error("Graph Data Model is missing. Cannot proceed with ingestion.")
        else:
            # --- Primary Ingestion Attempt: Use IngestionGenerator + PyIngest ---
            try:
                st.info("Attempt 1: Trying primary ingestion path (IngestionGenerator + PyIngest)...")
                yaml_config = None
                with st.spinner("Generating ingestion config..."):
                    gen = IngestionGenerator(data_model=gdm_model,
                                             username=username,
                                             password=password,
                                             uri=uri,
                                             database=database)
                    yaml_config = gen.generate_pyingest_yaml_string()

                st.subheader("Generated Ingestion YAML")
                st.code(yaml_config, language="yaml")

                with st.spinner("Running ingestion via PyIngest..."):
                    ingestor = PyIngest(yaml_string=yaml_config, dataframe=df)
                    ingestor.ingest_data()

                st.success("Primary ingestion completed successfully!")
                ingestion_succeeded = True

            except ClientError as ce:
                # Catch specific Neo4j errors, like the constraint error
                primary_ingestion_error = f"Neo4j Error: {ce.message}"
                logger.error(f"Primary ingestion failed with Neo4j ClientError: {ce.message}", exc_info=True)
                if "ConstraintCreationFailed" in ce.code and "Enterprise Edition" in ce.message:
                     st.warning("Primary ingestion failed, possibly due to Enterprise features (like composite Node Keys) used with Community Edition.")
                else:
                     st.warning(f"Primary ingestion failed: {primary_ingestion_error}")
            except ImportError as ie:
                primary_ingestion_error = f"Import Error: {ie}. This might indicate a missing class like PyIngestConfigGenerator in this library version."
                st.warning(primary_ingestion_error)
                logger.error(primary_ingestion_error, exc_info=True)
            except Exception as e:
                # Catch other potential errors during generation or ingestion
                primary_ingestion_error = f"An unexpected error occurred: {e}"
                st.warning(f"Primary ingestion failed: {primary_ingestion_error}")
                logger.error("Primary ingestion failed", exc_info=True)


            # --- Fallback Ingestion ---
            if not ingestion_succeeded:
                st.error(f"Primary graph model ingestion failed: {primary_ingestion_error}")
                st.info("Attempting fallback: 'Direct Ingestion' (will create flat nodes, not the graph structure).")
                try:
                    with st.spinner("Directly writing CSV rows into Neo4j (fallback)..."):
                        # FIX: Pass the database name correctly
                        direct_ingest_to_neo4j(uri=uri, username=username, password=password, database=database, df=df, label="Record")

                    st.success("Direct ingestion (fallback) completed.")
                    st.error("⚠️ **Warning:** The graph structure failed to build. Data was loaded as disconnected ':Record' nodes. The chat functionality will likely not work as expected.")
                    ingestion_succeeded = True # Fallback "succeeded"

                except Exception as e_fallback:
                    # This is where your 'unexpected keyword argument' error happened
                    st.error(f"Fallback ingestion ALSO failed: {e_fallback}")
                    logger.exception("direct_ingest_to_neo4j failed")
                    ingestion_succeeded = False # Both failed

        st.session_state['ingestion_done'] = ingestion_succeeded

        # Visualize graph and refresh schema
        if st.session_state['ingestion_done']:
            try:
                driver = GraphDatabase.driver(uri, auth=(username, password))
                # FIX: Pass database name to visualize_graph
                visualize_graph(driver, database)

                # Refresh schema (important for chat)
                kg = Neo4jGraph(url=uri, username=username, password=password, database=database)
                try:
                    logger.info(f"Attempting to refresh schema for database: {database}")
                    kg.refresh_schema()
                    st.session_state['kg_schema'] = kg.schema
                    logger.info(f"Refreshed schema: {kg.schema}")
                    if not kg.schema:
                         logger.warning("Schema is empty after refresh. This is expected if fallback ingestion was used.")
                except Exception as e_schema:
                    st.warning(f"Schema refresh failed: {e_schema}")
                    logger.error("Schema refresh failed", exc_info=True)

                driver.close()
            except Exception as e_post:
                st.error(f"Error during post-ingestion steps: {e_post}")
                logger.error("Post-ingestion failed", exc_info=True)

    # --- Step 5: Chat ---
    if st.session_state.get('ingestion_done', False):
        st.header("5) Chat with your Knowledge Graph")
        creds = st.session_state['neo4j_creds']

        try:
            kg = Neo4jGraph(url=creds['uri'], username=creds['username'], password=creds['password'], database=creds['database'])
            # Use schema directly from session state if available
            kg.schema = st.session_state.get('kg_schema', 'Schema not available.')
        except Exception as e:
            st.error(f"Failed to initialize Neo4jGraph for chat: {e}")
            st.stop() # Stop if graph connection fails

        st.subheader("Current Graph Schema")
        schema_text = kg.schema if kg.schema else "Schema not available or database is empty."
        st.text(textwrap.fill(str(schema_text), 80))

        # Check if schema looks valid for chat
        # A simple check: if it's empty or contains only ':Record', chat won't work well.
        is_fallback_schema = not schema_text or schema_text == "Node properties are the following: Record {_csv_index: INTEGER}\nThe relationships are the following:"

        if is_fallback_schema:
             st.warning("⚠️ Schema appears empty or only contains fallback ':Record' nodes. Chat functionality may be limited or inaccurate.")

        # Proceed with chat setup even if schema is limited
        chat_llm = get_llm("gpt3.5", api_key)
        cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=CHAT_PROMPT)

        try:
            cypher_chain = GraphCypherQAChain.from_llm(
                chat_llm,
                graph=kg,
                verbose=True,
                cypher_prompt=cypher_prompt,
                allow_dangerous_requests=True # Acknowledge security risk
            )
        except Exception as e:
            st.error(f"Failed to build GraphCypherQAChain: {e}")
            logger.exception("Chain creation failed")
            cypher_chain = None

        query = st.text_area("Ask a question about your data:", height=100)
        if query and cypher_chain:
            with st.spinner("Generating Cypher and answering..."):
                try:
                    # Check if kg.schema exists before invoking
                    if not kg.schema:
                         st.error("Cannot run query: Graph schema is missing.")
                    else:
                        answer = cypher_chain.invoke({"query": query}) # Pass query correctly
                        st.subheader("Answer")
                        st.write(answer.get('result', 'No answer found or query failed.'))

                        st.subheader("Generated Cypher")
                        st.code(answer.get('intermediate_steps', [{}])[0].get('query', 'Could not display query.'), language='cypher')

                except Exception as e:
                    st.error(f"Query execution error: {e}")
                    logger.exception("Cypher chain query failed")
        elif query and not cypher_chain:
             st.error("Chat chain failed to initialize. Cannot process query.")


if __name__ == "__main__":
    main()