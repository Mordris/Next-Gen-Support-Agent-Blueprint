# app/core/retriever.py
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag-agent-v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_retriever():
    """Initializes and returns a configured contextual compression retriever."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # --- THIS IS THE FIX ---
    # Connect to ChromaDB without any authentication credentials
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    reranker = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )

    return compression_retriever
