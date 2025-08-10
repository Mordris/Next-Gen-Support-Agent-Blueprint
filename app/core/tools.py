# app/core/tools.py
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime
import pytz  # For timezone-aware datetimes

from app.core.retriever import get_retriever


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieves relevant context from the knowledge base about eBay's policies.
    Use this tool to answer questions about the User Agreement, Privacy Policy,
    or the Money Back Guarantee policy.
    """
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in relevant_docs)


class OrderStatusInput(BaseModel):
    order_id: str = Field(description="The unique identifier of the order.")


@tool(args_schema=OrderStatusInput)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Checks the status of a specific order by its ID.
    Use this tool when a user asks for the status of their order.
    """
    # Expanded mock database
    orders_db = {
        "123": {
            "status": "Shipped",
            "tracking_number": "ABC123XYZ",
            "estimated_delivery": "2025-08-08",
        },
        "456": {"status": "Processing", "expected_delivery": "2025-08-10"},
    }
    return orders_db.get(
        order_id,
        {"status": "Not Found", "message": f"Order ID '{order_id}' not found."},
    )


@tool
def get_current_time() -> str:
    """
    Returns the current date and time in a standard format.
    Use this tool when the user asks for the current time, date, or day or when you need to understand the current context.
    """
    # Using a specific timezone for consistency
    utc_tz = pytz.timezone("UTC")
    return datetime.now(utc_tz).strftime("%A, %Y-%m-%d %H:%M:%S %Z")
