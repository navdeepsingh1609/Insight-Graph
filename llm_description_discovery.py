import ast
import logging
from langchain_community.chat_models import ChatOpenAI
from prompts import DESCRIPTION_PROMPT
from neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_MAPPING = {"gpt3.5": "gpt-3.5-turbo", "gpt4": "gpt-4"}

def get_llm(model: str, api_key: str):
    """Initialize and return a ChatOpenAI model."""
    if model not in MODEL_MAPPING:
        logging.error(f"Invalid model name: {model}")
        raise ValueError("Unsupported model.")
    try:
        return ChatOpenAI(openai_api_key=api_key, model=MODEL_MAPPING[model])
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        raise

def get_description_list(df, api_key: str):
    """Generate column descriptions using an LLM."""
    try:
        llm = get_llm("gpt3.5", api_key)
        columns = ", ".join(df.columns)
        sample_data = df.head(3).to_string(index=False)

        prompt = DESCRIPTION_PROMPT.format(COLUMN_NAMES=columns, df=sample_data)
        response = llm.invoke(prompt).content

        # Validate response
        if "=" not in response:
            raise ValueError("Unexpected response format.")
        cleaned_response = response.split("=", 1)[1].strip()
        return ast.literal_eval(cleaned_response)
    except (ValueError, SyntaxError) as e:
        logging.error(f"Error parsing LLM response: {e}")
        raise

def get_description_llm():
    """Return a Discovery LLM instance."""
    return OpenAIDiscoveryLLM()

def get_modeling_llm():
    """Return a Data Modeling LLM instance."""
    return OpenAIDataModelingLLM()
