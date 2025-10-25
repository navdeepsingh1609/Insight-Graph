import ast
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from prompts import DESCRIPTION_PROMPT

# neo4j-runway LLM helpers
from neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM

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

    FIX: This function is rewritten to use LangChain's JsonOutputParser
    for robust parsing, instead of brittle string splitting (e.g., find('{')).
    """
    try:
        llm = get_llm(model, api_key)
        columns = ", ".join(df.columns)
        sample_data = df.head(3).to_string(index=False)

        # 1. Setup the parser
        # This will create a dictionary where keys are column names and values are descriptions
        parser = JsonOutputParser(pydantic_object=None) # Using simple dict parser

        # 2. Create a prompt template that includes format instructions
        prompt = PromptTemplate(
            template=DESCRIPTION_PROMPT,
            input_variables=["COLUMN_NAMES", "df"],
            # Instruct the LLM to format its output as JSON
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

def get_description_llm():
    """
    neo4j-runway expects a Discovery LLM class. This returns that wrapper.
    """
    return OpenAIDiscoveryLLM()

def get_modeling_llm():
    """
    Return a modeling LLM wrapper used by neo4j-runway for graph modeling.
    """
    return OpenAIDataModelingLLM()