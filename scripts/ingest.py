# scripts/ingest.py
import logging
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
DATA_PATH = "data/attention_is_all_you_need.pdf"
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag-agent-v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():
    """Main function to run the ingestion pipeline."""
    logging.info(f"Loading document from {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # --- THIS IS THE FIX ---
    # Connect to ChromaDB without any authentication credentials
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    logging.info(f"Storing chunks in collection: {COLLECTION_NAME}")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
    )
    logging.info(f"Successfully stored {len(chunks)} chunks in ChromaDB.")


if __name__ == "__main__":
    main()
