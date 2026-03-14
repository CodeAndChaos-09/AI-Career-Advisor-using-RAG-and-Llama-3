# AI Career Advisor using RAG and Llama 3

A Streamlit application that provides career advice using Retrieval-Augmented Generation (RAG) with Llama 3.

## Features

- 🎓 Career advice for tech roles
- 🔍 RAG-based information retrieval
- 💬 Interactive chat interface
- 🤖 Powered by Llama 3 and Groq

## Deployment

This app is deployed on Streamlit Cloud and uses:
- Groq API for LLM inference
- HuggingFace embeddings for semantic search
- Local vector storage for career data

## Setup

1. Clone this repository
2. Add your Groq API key to Streamlit secrets
3. Run `streamlit run app.py`

## Data

The app uses career information stored in `data/career/` directory including:
- AI Engineer
- Backend Engineer  
- Data Scientist
- DevOps Engineer
- ML Engineer
