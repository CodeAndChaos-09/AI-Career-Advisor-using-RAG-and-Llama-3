import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import rerank_nodes

# Load environment variables
load_dotenv()

# -------------------------
# Embedding model
# -------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# LLM
# -------------------------
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------
# Load index
# -------------------------
storage_context = StorageContext.from_defaults(
    persist_dir="storage"
)

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=3)

print("🎓 AI Career Advisor Ready!")

while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    # Step 1: Rewrite query
    rewritten_query = rewrite_query(question)

    print(f"\n🔎 Rewritten Query: {rewritten_query}\n")

    # Step 2: Retrieve nodes
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(rewritten_query)

    # Step 3: Rerank nodes
    reranked_nodes = rerank_nodes(rewritten_query, nodes)

    # Step 4: Generate answer
    response = query_engine.response_synthesizer.synthesize(
        rewritten_query,
        reranked_nodes
    )

    print("\nAdvisor:", response)