import streamlit as st
import pandas as pd
import openai
import os
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import textwrap
from helper_function import (
    get_description_list, 
    get_description_llm, 
    get_modeling_llm, 
    get_llm
)
from neo4j_runway import Discovery, GraphDataModeler, PyIngest
from neo4j_runway.code_generation import PyIngestConfigGenerator
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from prompts import CHAT_PROMPT

# Set environment variable for API key
os.environ['OPENAI_API_KEY'] = '<key>'

# Initialize session state for variable persistence
if 'disc' not in st.session_state:
    st.session_state['disc'] = None
if 'gdm' not in st.session_state:
    st.session_state['gdm'] = None

def load_openai_key():
    """Load OpenAI API key securely via Streamlit UI."""
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        openai.api_key = api_key
        st.success("API key loaded successfully!")
    else:
        st.warning("Please enter your OpenAI API key.")
    return api_key

def main():
    st.title("Insight Graph")

    # Step 1: Load OpenAI API Key
    api_key = load_openai_key()
    if not api_key:
        st.stop()

    # Step 2: CSV Ingestion
    st.sidebar.title("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df)

        # Step 3: LLM Discovery
        st.title("LLM Discovery Process")
        if st.session_state['disc']:
            reuse_discovery = st.radio(
                "Reuse previous discovery or generate new?",
                ("Reuse", "Get New"),
                key="discovery_radio"
            )
        else:
            reuse_discovery = "Get New"

        if reuse_discovery == "Get New":
            descriptions = get_description_list(df, api_key)
            disc_llm = get_description_llm()
            st.session_state['disc'] = Discovery(
                llm=disc_llm, user_input=descriptions, data=df
            )
            st.session_state['disc'].run()

        disc = st.session_state['disc']
        st.write(disc.discovery)

        # Step 4: Data Modeling
        st.title("Data Model Visualization")

        if st.session_state['gdm']:
            reuse_modeling = st.radio(
                "Reuse previous model or generate new?",
                ("Reuse", "Get New"),
                key="modeling_radio"
            )
        else:
            reuse_modeling = "Get New"

        if reuse_modeling == "Get New":
            modeling_llm = get_modeling_llm()
            st.session_state['gdm'] = GraphDataModeler(
                llm=modeling_llm, discovery=disc
            )
            st.session_state['gdm'].create_initial_model()

        gdm = st.session_state['gdm']
        st.graphviz_chart(
            gdm.current_model.visualize(),
            use_container_width=True
        )

         # Feedback for Iteration
        feedback = st.radio(
            "Do you have additional inputs for the current visualization?",
            ("No", "Yes")
        )

        if feedback == "Yes":
            suggestions = st.text_area("Enter your suggestions:", height=300)
            if suggestions:
                gdm.iterate_model(user_corrections=suggestions)
                st.graphviz_chart(
                    gdm.current_model.visualize(),
                    use_container_width=True
                )

        # Step 5: Neo4j Configuration and Ingestion
        st.title("Neo4j Configuration")
        st.subheader("Enter Neo4j Credentials")
        with st.form("neo4j_credentials_form"):
            username = st.text_input("Neo4j Username", value="neo4j")
            password = st.text_input("Neo4j Password", type="password", value="StrongPassword123")
            uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
            database = st.text_input("Database Name", value="neo4j")
            csv_name = st.text_input("CSV Filename", value="input_data.csv")

            submitted = st.form_submit_button("Generate Ingestion Code")
        
        if submitted:
            gen = PyIngestConfigGenerator(
                data_model=gdm.current_model,
                username=username,
                password=password,
                uri=uri,
                database=database,
                csv_name=csv_name
            )

            yaml_config = gen.generate_config_string()
            st.subheader("Generated Ingestion YAML")
            st.code(yaml_config, language="yaml")

            PyIngest(config=yaml_config, dataframe=df)
            gen.generate_config_yaml(file_name="ingestion.yaml")

            driver = GraphDatabase.driver(uri, auth=(username, password))
            def fetch_graph_data():
                query = """
                MATCH (n)-[r]->(m)
                RETURN n, r, m
                """
                with driver.session() as session:
                    results = session.run(query)
                    nodes, edges = [], []
                    for record in results:
                        n, m, r = record["n"], record["m"], record["r"]
                        nodes.append(n)
                        nodes.append(m)
                        edges.append((n.id, m.id, r.type))
                    nodes = {n.id: n for n in nodes}.values()
                    return nodes, edges

            nodes, edges = fetch_graph_data()

            net = Network(height="750px", width="100%", notebook=False)
            for node in nodes:
                label = str(node.labels) if isinstance(node.labels, frozenset) else node.labels
                net.add_node(node.id, label=str(node.id), title=label)
            for edge in edges:
                net.add_edge(edge[0], edge[1], title=edge[2])

            net.show("graph.html")
            HtmlFile = open("graph.html", "r", encoding="utf-8")
            components.html(HtmlFile.read(), height=800, width=1000)

        # Step 5: Inference
        kg = Neo4jGraph(
            url=uri, username=username, password=password, database=database
        )
        kg.refresh_schema()

        st.text(textwrap.fill(kg.schema, 60))
        schema = kg.schema

        chat_llm = get_llm("gpt3.5 Turbo", api_key)
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CHAT_PROMPT
        )

        cypher_chain = GraphCypherQAChain.from_llm(
            chat_llm, graph=kg, verbose=True, cypher_prompt=cypher_prompt
        )

        st.header("Enter Your Query")
        query = st.text_area("Question", height=300)
        if query:
            response = cypher_chain.run(query)
            st.text(response)
        
if __name__ == "__main__":
    main()
