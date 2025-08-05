# scripts/ingest.py
import logging
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- THIS IS THE LINE TO CHANGE ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging to see the progress
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
DATA_PATH = "data/attention_is_all_you_need.pdf"
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000  # Correct internal port
COLLECTION_NAME = "rag-agent-v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():
    """Main function to run the ingestion pipeline."""

    logging.info(f"Loading document from {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        logging.error("No documents were loaded. Check the file path.")
        return
    logging.info(f"Loaded {len(documents)} document pages.")

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")

    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # The HuggingFaceEmbeddings class now comes from the new package
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    logging.info("Embedding model loaded.")

    logging.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    logging.info(f"Storing chunks in collection: {COLLECTION_NAME}")
    # It's safe to run this again. Chroma will 'upsert' the data,
    # meaning it will update existing entries or add new ones.
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
    )
    logging.info(f"Successfully stored {len(chunks)} chunks in ChromaDB.")
    logging.info("Ingestion complete.")


if __name__ == "__main__":
    main()
