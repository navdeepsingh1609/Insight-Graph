import ast
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from prompts import DESCRIPTION_PROMPT

# FIX: The neo4j-runway==0.4.3 library has a different structure.
# We import the single 'LLM' class as shown in the 0.4.x docs.
from neo4j_runway import LLM 

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Map friendly names -> actual OpenAI model IDs
MODEL_MAPPING = {"gpt3.5": "gpt-3.5-turbo", "gpt4": "gpt-4"}

def get_llm(model: str, api_key: str):
    """
    Initialize and return a ChatOpenAI model instance.
    """
    if model not in MODEL_MAPPING:
        logger.error("Invalid model name: %s (allowed: %s)", model, list(MODEL_MAPPING.keys()))
        raise ValueError(f"Unsupported model name: {model}")
    try:
        return ChatOpenAI(openai_api_key=api_key, model=MODEL_MAPPING[model])
    except Exception as e:
        logger.exception("Error initializing ChatOpenAI. Check your langchain_community package.")
        raise

def get_description_list(df, api_key: str, model: str = "gpt3.5"):
    """
    Use an LLM to generate a dictionary mapping column -> short description.
    Uses LangChain's JsonOutputParser for robust parsing.
    """
    try:
        llm = get_llm(model, api_key)
        columns = ", ".join(df.columns)
        sample_data = df.head(3).to_string(index=False)

        # 1. Setup the parser
        parser = JsonOutputParser(pydantic_object=None) 

        # 2. Create a prompt template that includes format instructions
        prompt = PromptTemplate(
            template=DESCRIPTION_PROMPT,
            input_variables=["COLUMN_NAMES", "df"],
            partial_variables={"format_instructions": "Return a JSON object where each key is a column name and each value is its description."}
        )

        # 3. Create a chain
        chain = prompt | llm | parser

        # 4. Invoke the chain
        response_dict = chain.invoke({
            "COLUMN_NAMES": columns,
            "df": sample_data
        })

        if not isinstance(response_dict, dict):
            logger.error("LLM did not return a dictionary for descriptions.")
            return {col: "" for col in df.columns}

        # Ensure all columns are present, even if LLM missed some
        final_descriptions = {col: response_dict.get(col, "No description generated.") for col in df.columns}
        return final_descriptions

    except Exception as e:
        logger.exception("get_description_list failed.")
        # Fallback to empty descriptions
        return {col: "" for col in df.columns}

# ==================================================================
# !! FIX: These functions are modified for neo4j-runway==0.4.3 !!
# The LLM constructor takes NO arguments.
# ==================================================================

def get_description_llm():
    """
    neo4j-runway 0.4.3 expects a single LLM class.
    We initialize it with no arguments, as per the docs for that version.
    It will pick up the OPENAI_API_KEY from the environment.
    """
    try:
        return LLM()
    except Exception as e:
        logger.exception("Failed to initialize neo4j_runway.LLM()")
        raise

def get_modeling_llm():
    """
    neo4j-runway 0.4.3 expects a single LLM class.
    We initialize it with no arguments, as per the docs for that version.
    It will pick up the OPENAI_API_KEY from the environment.
    """
    try:
        return LLM()
    except Exception as e:
        logger.exception("Failed to initialize neo4j_runway.LLM()")
        raise