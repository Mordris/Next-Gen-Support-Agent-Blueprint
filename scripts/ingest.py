# scripts/ingest.py
import logging
import chromadb
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# Use glob to find all PDF files in the data directory
DATA_PATH = "data/*.pdf"
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
# Let's create a new collection for the new data
COLLECTION_NAME = "ebay-policies-v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():
    """Finds all PDFs in the data folder, processes them, and ingests them into ChromaDB."""
    pdf_files = glob.glob(DATA_PATH)
    if not pdf_files:
        logging.error("No PDF files found in the 'data' directory. Exiting.")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        logging.info(f"Loading document: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logging.info(f"Splitting {len(documents)} pages into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        logging.info(f"Created {len(chunks)} chunks for this document.")

    if not all_chunks:
        logging.error("No content could be extracted from the PDF files. Exiting.")
        return

    logging.info(f"Total chunks to ingest: {len(all_chunks)}")
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    logging.info(f"Storing chunks in ChromaDB collection: {COLLECTION_NAME}")
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
    )
    logging.info("Ingestion complete.")


if __name__ == "__main__":
    main()
