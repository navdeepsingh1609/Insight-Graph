# FIX: Simplified the description prompt and instructed the parser.
# We will use a JsonOutputParser for this, so the prompt needs to mention the format.
DESCRIPTION_PROMPT = """
A table with the following column names : {COLUMN_NAMES} and the the sample dataframe is as follows : {df}.
###Instruction###
You are a spreadsheet analyzer. Analyze the dataframe and write a short, one-sentence summary
of what each column represents.

{format_instructions}
"""

# FIX: Drastically simplified the CHAT_PROMPT.
# The original prompt was "over-fit" to a specific healthcare schema (diseases, symptoms).
# This will fail for any other CSV.
# The GraphCypherQAChain automatically injects the *correct* schema into the {schema} variable.
# We should trust that process and provide a simple, generic prompt.
CHAT_PROMPT = """
You are an expert Cypher translator. Given a graph database schema and a user's question,
generate a Cypher statement to answer the question.

DO NOT use any node labels, relationship types, or property keys that are not
explicitly present in the provided schema.
DO NOT add any comments or preamble to your Cypher query.
DO NOT wrap the Cypher query in backticks or markdown.
ONLY output the Cypher query.

Schema:
{schema}

Question:
{question}
"""