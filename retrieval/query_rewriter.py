from llama_index.core import Settings


def rewrite_query(user_query: str) -> str:
    """
    Uses the configured LLM (Groq Llama-3) to rewrite the query
    for better retrieval.
    """

    prompt = f"""
Rewrite the following user question to be clearer and optimized
for retrieving information from a knowledge base.

Question: {user_query}

Rewritten question:
"""

    response = Settings.llm.complete(prompt)

    return str(response).strip()