from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# -------------------------
# Embedding model
# -------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Load documents
# -------------------------
documents = SimpleDirectoryReader("data/career").load_data()

# -------------------------
# Chunk documents
# -------------------------
parser = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=50
)

nodes = parser.get_nodes_from_documents(documents)

# -------------------------
# Build index
# -------------------------
index = VectorStoreIndex(nodes)

# -------------------------
# Save index
# -------------------------
index.storage_context.persist(persist_dir="storage")

print("✅ Index built successfully with chunking!")