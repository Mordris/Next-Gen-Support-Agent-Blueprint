# app/core/retriever.py

from langchain_chroma import Chroma
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# --- Configuration ---
# Must match the details from our ingestion script
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag-agent-v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_retriever():
    """
    Initializes and returns a configured contextual compression retriever.
    This retriever performs a two-step process:
    1. Fetches documents from ChromaDB based on vector similarity.
    2. Uses a re-ranker model to improve the relevance of the results.
    """
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Initialize the ChromaDB client
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # 1. Initialize the base retriever
    # This retriever fetches a larger number of documents (k=10) initially.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 2. Initialize the re-ranker
    # This model will take the top 10 documents and re-rank them for relevance.
    # It's computationally cheaper than an LLM but very effective.
    reranker = FlashrankRerank()

    # 3. Create the Contextual Compression Retriever
    # This special retriever wraps the base retriever and the re-ranker.
    # It will automatically handle the "retrieve-then-rerank" process.
    # It returns only the top 3 most relevant documents after re-ranking.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )

    return compression_retriever


# --- For testing purposes ---
if __name__ == "__main__":
    # This block allows you to test the retriever directly
    # by running `python -m app.core.retriever` from the root directory.
    print("Testing the retriever...")
    test_retriever = get_retriever()
    query = "what is a transformer model?"
    print(f"Running query: '{query}'")
    results = test_retriever.invoke(query)

    print(f"\nFound {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:400]}...")  # Print first 400 chars
        print(f"Metadata: {doc.metadata}")
