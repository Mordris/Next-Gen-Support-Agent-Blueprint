# app/main.py
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional

from pydantic import BaseModel
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from redis import Redis
from redis.exceptions import RedisError

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI

from app.core.security import api_key_auth
from app.core.retriever import prewarm_retriever
from app.core.agent import SelfCorrectingAgent


# --- Configuration & State ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
redis_client: Optional[Redis] = None
in_memory_histories: Dict[str, InMemoryChatMessageHistory] = {}
agent: Optional[SelfCorrectingAgent] = None


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, agent
    logger.info("Starting up application...")

    try:
        redis_client = Redis.from_url("redis://redis:6379", decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established.")
    except RedisError as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory history.")
        redis_client = None

    # Initialize the new self-correcting agent
    logger.info("Initializing self-correcting agent...")
    powerful_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    agent = SelfCorrectingAgent(powerful_llm)
    logger.info("Agent initialized successfully.")

    prewarm_retriever()
    yield

    logger.info("Shutting down application...")
    if redis_client:
        redis_client.close()


# --- FastAPI App Initialization ---
app = FastAPI(title="Next-Gen Support Agent API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class HealthResponse(BaseModel):
    status: bool
    redis_connected: bool


class ChatRequest(BaseModel):
    message: str
    session_id: str


# --- Memory Configuration ---
def get_session_history(session_id: str):
    if redis_client:
        return RedisChatMessageHistory(session_id, url="redis://redis:6379")
    if session_id not in in_memory_histories:
        in_memory_histories[session_id] = InMemoryChatMessageHistory()
    return in_memory_histories[session_id]


# --- Event Processing ---
def process_langgraph_event(event: Dict[str, Any]) -> list[Dict]:
    """
    Processes LangGraph events to extract UI-relevant information.
    """
    ui_events = []

    # LangGraph events have a different structure than LangChain
    for node_name, node_data in event.items():
        if node_name == "reasoning":
            ui_events.append(
                {
                    "event": "reasoning_start",
                    "step": "reasoning",
                    "iteration": node_data.get("iteration_count", 0),
                }
            )

            # Check if the reasoning node produced tool calls
            messages = node_data.get("messages", [])
            for message in messages:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        ui_events.append(
                            {"event": "tool_start", "name": tool_call["name"]}
                        )

        elif node_name == "tools":
            # Tool execution completed
            messages = node_data.get("messages", [])
            for message in messages:
                if hasattr(message, "name"):  # ToolMessage
                    ui_events.append(
                        {
                            "event": "tool_end",
                            "name": message.name,
                            "output": str(message.content) if message.content else "",
                        }
                    )

        elif node_name == "response_generation":
            ui_events.append({"event": "response_start", "step": "generating_response"})

            # Extract the response content
            messages = node_data.get("messages", [])
            for message in messages:
                if hasattr(message, "content") and message.content:
                    ui_events.append(
                        {"event": "final_response", "content": message.content}
                    )

        elif node_name == "self_assessment":
            confidence = node_data.get("confidence_score", 0.0)
            needs_correction = node_data.get("needs_correction", False)
            ui_events.append(
                {
                    "event": "self_assessment",
                    "confidence": confidence,
                    "needs_correction": needs_correction,
                    "reason": node_data.get("correction_reason"),
                }
            )

        elif node_name == "correction":
            ui_events.append(
                {
                    "event": "correction_start",
                    "reason": node_data.get("correction_reason"),
                }
            )

        elif node_name == "error_handling":
            ui_events.append(
                {
                    "event": "error",
                    "data": "The agent encountered an error and is attempting recovery.",
                }
            )

    return ui_events


# --- Streaming Logic ---
async def event_streamer(
    request: Request, message: str, session_id: str
) -> AsyncGenerator[str, None]:
    """
    Streams events from the LangGraph agent.
    """
    if not agent:
        yield f"data: {json.dumps({'event': 'error', 'data': 'Agent not initialized'})}\n\n"
        return

    logger.info(
        f"Processing message: '{message[:50]}...' for session: {session_id[:8]}..."
    )

    try:
        # Stream from the LangGraph agent
        async for event in agent.astream_response(message, session_id):
            if await request.is_disconnected():
                break

            # Check for errors
            if "error" in event:
                yield f"data: {json.dumps({'event': 'error', 'data': event['error']})}\n\n"
                continue

            # Process the event and convert to UI format
            ui_events = process_langgraph_event(event)
            for ui_event in ui_events:
                yield f"data: {json.dumps(ui_event)}\n\n"

    except Exception as e:
        logger.error(f"Error in event streaming: {e}")
        yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status=True,
        redis_connected=redis_client is not None,
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: Request, chat_request: ChatRequest, _=Depends(api_key_auth)
):
    return StreamingResponse(
        event_streamer(request, chat_request.message, chat_request.session_id),
        media_type="text/event-stream",
    )


# --- Debug Endpoint for Development ---
@app.get("/agent/status")
async def agent_status(_=Depends(api_key_auth)):
    """Debug endpoint to check agent status."""
    if not agent:
        return {"status": "not_initialized"}

    return {
        "status": "initialized",
        "tools_count": len(agent.tools),
        "graph_nodes": list(agent.graph.nodes.keys())
        if hasattr(agent.graph, "nodes")
        else [],
    }
