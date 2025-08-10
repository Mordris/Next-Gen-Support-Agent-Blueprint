# app/core/tools.py
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Union
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
    try:
        retriever = get_retriever()
        relevant_docs = retriever.invoke(query)
        if not relevant_docs:
            return "No relevant context found in the knowledge base."
        return "\n\n".join(doc.page_content for doc in relevant_docs)
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


class OrderStatusInput(BaseModel):
    order_id: str = Field(description="The unique identifier of the order.")


@tool(args_schema=OrderStatusInput)
def get_order_status(order_id: str) -> Union[Dict[str, Any], str]:
    """
    Checks the status of a specific order by its ID.
    Use this tool when a user asks for the status of their order.
    The order_id parameter is required - if the user hasn't provided it,
    you should ask them for their order ID first.
    """
    if not order_id or not order_id.strip():
        return "ERROR: Order ID is required. Please ask the user to provide their order ID."

    # Clean the order ID
    order_id = order_id.strip()

    # Expanded mock database
    orders_db = {
        "123": {
            "status": "Shipped",
            "tracking_number": "ABC123XYZ",
            "estimated_delivery": "2025-08-08",
        },
        "456": {"status": "Processing", "expected_delivery": "2025-08-10"},
        "789": {
            "status": "Delivered",
            "delivery_date": "2025-08-05",
            "delivered_to": "Front door",
        },
    }

    result = orders_db.get(order_id)
    if result:
        return result
    else:
        return {
            "status": "Not Found",
            "message": f"Order ID '{order_id}' was not found in our system. Please verify the order ID is correct.",
        }


@tool
def get_current_time() -> str:
    """
    Returns the current date and time in a standard format.
    Use this tool when the user asks for the current time, date, or day
    or when you need to understand the current context.
    """
    try:
        # Using a specific timezone for consistency
        utc_tz = pytz.timezone("UTC")
        return datetime.now(utc_tz).strftime("%A, %Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error getting current time: {str(e)}"


@tool
def ask_for_order_id() -> str:
    """
    Use this tool when a user asks about their order status but hasn't provided an order ID.
    This tool returns a message asking the user to provide their order ID.
    """
    return "To check your order status, I'll need your order ID. You can find this in your purchase confirmation email or your eBay account under 'My eBay' > 'Purchase history'. Could you please provide your order ID?"
