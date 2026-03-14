import streamlit as st
import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import rerank_nodes

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# -------------------------
# Setup models
# -------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------
# Load vector index
# -------------------------
storage_context = StorageContext.from_defaults(
    persist_dir="storage"
)

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

retriever = index.as_retriever(similarity_top_k=5)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Career Advisor")

st.title("🎓 AI Career Advisor")
st.write("Ask questions about tech careers, skills, and learning paths.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask a career question")

if prompt:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Query rewriting
    rewritten_query = rewrite_query(prompt)

    # Retrieve nodes
    nodes = retriever.retrieve(rewritten_query)

    # Rerank nodes
    reranked_nodes = rerank_nodes(rewritten_query, nodes)

    # Generate answer
    response = query_engine.response_synthesizer.synthesize(
        rewritten_query,
        reranked_nodes
    )

    answer = str(response)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )