# app/core/retriever.py
import logging
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# --- Configuration ---
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
# --- THIS IS THE FIX ---
# Point to the new collection name
COLLECTION_NAME = "ebay-policies-v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_retriever():
    """Initializes and returns a singleton instance of the retriever."""
    logger.info("Initializing retriever...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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
    logger.info("Retriever initialized successfully.")
    return compression_retriever


def prewarm_retriever():
    """Triggers retriever initialization and caching."""
    logger.info("Pre-warming retriever...")
    get_retriever()
    logger.info("Retriever pre-warming complete.")
