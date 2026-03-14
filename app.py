import streamlit as st
import os

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex
)

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retrieval.query_rewriter import rewrite_query


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎓",
)

st.title("🎓 AI Career Advisor")
st.write("Ask questions about tech careers, skills, and learning paths.")


# -----------------------------
# Load Models (cached)
# -----------------------------
@st.cache_resource
def load_models():

    api_key = st.secrets["GROQ_API_KEY"]

    # Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Llama-3 via Groq
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

load_models()


# -----------------------------
# Load or Build Vector Index
# -----------------------------
@st.cache_resource
def load_index():

    PERSIST_DIR = "storage"

    if not os.path.exists(PERSIST_DIR):

        st.info("Building vector index for the first time...")

        documents = SimpleDirectoryReader("data/career").load_data()

        index = VectorStoreIndex.from_documents(documents)

        index.storage_context.persist(persist_dir=PERSIST_DIR)

    else:

        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR
        )

        index = load_index_from_storage(storage_context)

    return index


index = load_index()

query_engine = index.as_query_engine(similarity_top_k=3)


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("Ask a career question")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):

        rewritten_query = rewrite_query(prompt)

        response = query_engine.query(rewritten_query)

        answer = str(response)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )