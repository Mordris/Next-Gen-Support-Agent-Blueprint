import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Redis and LangChain imports
from redis import Redis
from redis.exceptions import RedisError
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import RunnableConfig

from app.core.security import api_key_auth
from app.core.tools import retrieve_context, get_order_status

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- State Management ---
redis_client: Optional[Redis] = None
in_memory_histories: Dict[str, InMemoryChatMessageHistory] = {}


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    global redis_client
    logger.info("Starting up application...")
    try:
        redis_client = Redis.from_url("redis://redis:6379", decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established.")
    except RedisError as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory history.")
        redis_client = None
    yield
    logger.info("Shutting down application...")
    if redis_client:
        redis_client.close()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Next-Gen Support Agent API", version="1.0.1", lifespan=lifespan
)  # Final Version
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


# --- Agent Configuration ---
tools = [retrieve_context, get_order_status]
prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=False, handle_parsing_errors=True
)


def get_session_history(session_id: str):
    if redis_client:
        return RedisChatMessageHistory(session_id, url="redis://redis:6379")
    if session_id not in in_memory_histories:
        in_memory_histories[session_id] = InMemoryChatMessageHistory()
    return in_memory_histories[session_id]


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# --- Real-Time Streaming Logic ---
async def event_streamer(
    request: Request, message: str, session_id: str
) -> AsyncGenerator[str, None]:
    """Streams agent events in real-time, correctly handling tool names."""
    config: RunnableConfig = {"configurable": {"session_id": session_id}}

    # Track tool execution state
    current_tool_name = None
    tool_execution_stack = []

    try:
        async for event in agent_with_chat_history.astream_events(
            {"input": message}, config=config, version="v2"
        ):
            if await request.is_disconnected():
                break

            event_name = event["event"]
            ui_event = None

            if event_name == "on_tool_start":
                # Get tool name from multiple possible locations
                tool_name = None

                # Try to get from event data
                if "data" in event:
                    tool_name = event["data"].get("name")
                    if not tool_name and "input" in event["data"]:
                        # Sometimes the tool name is in the input
                        tool_name = event["data"]["input"].get(
                            "tool", event["data"]["input"].get("name")
                        )

                # Try to get from event name or tags
                if not tool_name and "name" in event:
                    tool_name = event["name"]

                # Try to get from tags (sometimes tools are identified here)
                if not tool_name and "tags" in event:
                    tags = event["tags"]
                    for tag in tags:
                        if tag in ["retrieve_context", "get_order_status"]:
                            tool_name = tag
                            break

                # Fallback: try to extract from the runnable name
                if not tool_name and "run_id" in event:
                    # Sometimes we need to look at metadata or other fields
                    metadata = event.get("metadata", {})
                    if "name" in metadata:
                        tool_name = metadata["name"]

                if tool_name:
                    current_tool_name = tool_name
                    tool_execution_stack.append(tool_name)
                    ui_event = {"event": "tool_start", "name": tool_name}
                    logger.info(f"Tool started: {tool_name}")
                else:
                    logger.warning(f"Could not determine tool name from event: {event}")

            elif event_name == "on_tool_end":
                # Use the most recent tool from our stack
                tool_name = current_tool_name
                if tool_execution_stack:
                    tool_name = tool_execution_stack.pop()

                # If we still don't have a name, try to extract it from the event
                if not tool_name:
                    if "data" in event:
                        tool_name = event["data"].get("name")
                        if not tool_name and "input" in event["data"]:
                            tool_name = event["data"]["input"].get(
                                "tool", event["data"]["input"].get("name")
                            )

                # Get the output
                output = ""
                if "data" in event and "output" in event["data"]:
                    output = str(event["data"]["output"])

                ui_event = {
                    "event": "tool_end",
                    "name": tool_name or "unknown_tool",
                    "output": output,
                }

                logger.info(f"Tool ended: {tool_name or 'unknown'}")
                current_tool_name = None

            elif event_name == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    ui_event = {"event": "final_token", "data": token}

            if ui_event:
                json_payload = json.dumps(ui_event)
                logger.info(f"SENDING: {json_payload}")
                yield f"data: {json_payload}\n\n"

        yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status=True, redis_connected=redis_client is not None)


@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: Request, chat_request: ChatRequest, _=Depends(api_key_auth)
):
    return StreamingResponse(
        event_streamer(request, chat_request.message, chat_request.session_id),
        media_type="text/event-stream",
    )
