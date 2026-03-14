from llama_index.core import Settings

def rewrite_query(user_query: str) -> str:
    """
    Uses the LLM to rewrite the user query into a more
    specific search query for retrieval.
    """

    prompt = f"""
You are an AI assistant helping improve search queries.

Rewrite the user's question so it becomes a clearer
and more detailed query for retrieving career information.

User Question:
{user_query}

Improved Query:
"""

    response = Settings.llm.complete(prompt)

    return response.text.strip()