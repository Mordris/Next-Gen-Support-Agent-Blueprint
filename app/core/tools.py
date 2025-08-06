from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from langchain_core.documents import Document

from app.core.retriever import get_retriever


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieves relevant context from the knowledge base (the 'Attention Is All You Need' paper)
    based on the user's query. Use this tool to answer questions about the Transformer model,
    attention mechanisms, or machine translation.
    """
    retriever = get_retriever()
    relevant_docs: List[Document] = retriever.invoke(query)

    if not relevant_docs:
        return "No relevant documents found."

    # Return the full context for the agent to use
    context = "\n\n---\n\n".join(
        [
            f"Document {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ]
    )

    # Add a summary at the beginning for better context
    summary = f"Retrieved {len(relevant_docs)} relevant document(s) from the knowledge base.\n\n"

    return summary + context


class OrderStatusInput(BaseModel):
    order_id: str = Field(description="The unique identifier of the order.")


@tool(args_schema=OrderStatusInput)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Checks the status of a specific order by its ID.
    Use this tool when a user asks for the status of their order.
    """
    # Simulate a database lookup
    orders_db = {
        "123": {
            "status": "Shipped",
            "tracking_number": "ABC123XYZ",
            "estimated_delivery": "2025-08-08",
        },
        "456": {
            "status": "Processing",
            "expected_delivery": "2025-08-10",
            "processing_stage": "Packaging",
        },
        "789": {
            "status": "Delivered",
            "delivered_date": "2025-08-05",
            "tracking_number": "XYZ789ABC",
        },
    }

    if order_id in orders_db:
        return orders_db[order_id]
    else:
        return {
            "status": "Not Found",
            "message": f"Order ID '{order_id}' not found in our system.",
            "suggestion": "Please check the order ID and try again, or contact customer support.",
        }
