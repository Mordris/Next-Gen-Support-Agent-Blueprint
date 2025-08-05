# app/main.py
import asyncio  # Add this import

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from fastapi.responses import StreamingResponse  # Add this import

from app.core.security import api_key_auth

# Create the FastAPI app instance
app = FastAPI(title="Next-Gen Support Agent API")


# --- Define Request and Response Models ---
class HealthResponse(BaseModel):
    status: bool


# Add a model for our chat request body
class ChatRequest(BaseModel):
    message: str


# --- Helper Generator for Streaming ---
async def echo_stream(message: str):
    """
    A simple async generator that echoes the message back, word by word.
    """
    for word in message.split():
        yield word + " "
        await asyncio.sleep(0.1)  # Simulate a delay


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Endpoint to check if the API is running."""
    return {"status": True}


@app.get("/secure", tags=["Testing"])
async def secure_endpoint(api_key: str = Depends(api_key_auth)):
    """A secure endpoint that requires a valid API key."""
    return {"message": "This is a secure endpoint."}


# Add the new streaming endpoint
@app.post("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(
    request: ChatRequest,
    api_key: str = Depends(api_key_auth),  # This endpoint is protected
):
    """
    Receives a message and streams an echo response back.
    """
    return StreamingResponse(echo_stream(request.message), media_type="text/plain")
