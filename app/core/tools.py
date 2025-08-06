# app/core/tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any

# Import the retriever function we already built
from app.core.retriever import get_retriever


# 1. Define the RAG Tool
# We wrap our advanced retriever in a LangChain @tool decorator.
# This makes it a component that an agent can select and use.
@tool
def retrieve_context(query: str) -> str:
    """
    Retrieves relevant context from the knowledge base (the 'Attention Is All You Need' paper)
    based on the user's query. Use this tool to answer questions about the Transformer model,
    attention mechanisms, or machine translation.
    """
    retriever = get_retriever()
    # The invoke method runs the retriever and returns the documents
    relevant_docs = retriever.invoke(query)
    # We format the documents into a single string to be passed to the LLM
    return "\n\n".join(doc.page_content for doc in relevant_docs)


# 2. Define the Order Status Tool
# We define a Pydantic model for the arguments of this tool.
# This provides structure and validation for the tool's inputs.
class OrderStatusInput(BaseModel):
    order_id: str = Field(description="The unique identifier of the order.")


@tool(args_schema=OrderStatusInput)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Checks the status of a specific order by its ID.
    Use this tool when a user asks for the status of their order.
    """
    # In a real application, this would query a database or an external API.
    # For this lab, we will return mock data.
    if order_id == "123":
        return {"status": "Shipped", "tracking_number": "ABC123XYZ"}
    elif order_id == "456":
        return {"status": "Processing", "expected_delivery": "2025-08-10"}
    else:
        return {"status": "Not Found", "message": f"Order ID '{order_id}' not found."}
