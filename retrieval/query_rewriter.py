from llama_index.core import Settings


def rewrite_query(user_query: str) -> str:
    """
    Uses the configured LLM (Groq Llama 3) to rewrite
    the user query for better retrieval.
    """

    prompt = f"""
Rewrite the following user question so it is clearer and better
suited for retrieving information from a knowledge base.

Original Question:
{user_query}

Rewritten Question:
"""

    response = Settings.llm.complete(prompt)

    # convert LLM response to clean string
    rewritten_query = str(response).strip()

    return rewritten_query