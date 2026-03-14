import streamlit as st
import os
api_key = st.secrets["GROQ_API_KEY"]
from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex
)

from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

from retrieval.query_rewriter import rewrite_query

load_dotenv()

st.set_page_config(page_title="AI Career Advisor", page_icon="🎓")

st.title("🎓 AI Career Advisor")
st.write("Ask questions about tech careers, skills and learning paths.")

# -----------------------
# Load Models
# -----------------------

@st.cache_resource
def load_models():

    api_key = st.secrets["GROQ_API_KEY"]

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
        api_base="https://api.groq.com/openai/v1"
    )

    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

load_models()

# -----------------------
# Load or Build Index
# -----------------------

@st.cache_resource
def load_index():

    PERSIST_DIR = "storage"

    if not os.path.exists(PERSIST_DIR):

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

# -----------------------
# Chat UI
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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