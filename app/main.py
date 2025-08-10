import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Literal, Optional, List

from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from redis import Redis
from redis.exceptions import RedisError

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessageChunk

from app.core.security import api_key_auth
from app.core.tools import retrieve_context, get_order_status, get_current_time
from app.core.retriever import prewarm_retriever


# --- Configuration & State ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
redis_client: Optional[Redis] = None
in_memory_histories: Dict[str, InMemoryChatMessageHistory] = {}


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    logger.info("Starting up application...")

    try:
        redis_client = Redis.from_url("redis://redis:6379", decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established.")
    except RedisError as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory history.")
        redis_client = None

    prewarm_retriever()
    yield

    logger.info("Shutting down application...")
    if redis_client:
        redis_client.close()


# --- FastAPI App Initialization ---
app = FastAPI(title="Next-Gen Support Agent API", version="1.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Changed from "" to allow all origins
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


class RouteQuery(BaseModel):
    datasource: Literal["agent", "conversation"] = Field(
        ..., description="Route a user query to the most relevant datasource."
    )


# --- LLM and Agent Configuration ---
powerful_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
fast_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)

agent_system_prompt = """
You are an expert customer support agent for eBay.
Your primary goal is to provide accurate and helpful information based exclusively on the context retrieved from the available tools.

Core Directives:
- Prioritize Tools: For any user question about eBay policies (refunds, returns, privacy, user agreement), order statuses, or the current time, you MUST use the provided tools.
- Ground Your Answers: Base your answers directly on the output of the tools. If the retrieved context does not contain the answer, clearly state you do not have that information.
- Be Conversational but accurate.

Tool Usage:
- retrieve_context → policy-related questions
- get_order_status → order-related questions
- get_current_time → time/date questions
"""

tools = [retrieve_context, get_order_status, get_current_time]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_tools_agent(powerful_llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=False, handle_parsing_errors=True
)

# Simple Conversational Chain
conversational_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and friendly assistant."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ]
)
conversational_chain = conversational_prompt | fast_llm | StrOutputParser()

# Router Chain
router_prompt_template = (
    "You are an expert at routing a user query to the appropriate data source.\n"
    "Classify as 'agent' for questions about eBay's policies, order statuses, or the current time/date.\n"
    "Otherwise, classify as 'conversation'."
)
router_prompt = ChatPromptTemplate.from_messages(
    [("system", router_prompt_template), ("human", "{input}")]
)
router_chain = router_prompt | fast_llm.with_structured_output(
    RouteQuery, method="function_calling"
)


# --- Memory Configuration ---
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
conversational_chain_with_history = RunnableWithMessageHistory(
    conversational_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# --- Robust Event Processing ---
def process_event(event: Any) -> List[Dict]:
    """Processes a LangChain event stream to find relevant UI events."""
    ui_events = []
    event_name = event.get("event")

    if event_name == "on_tool_start":
        if name := event.get("name"):
            ui_events.append({"event": "tool_start", "name": name})

    elif event_name == "on_tool_end":
        if name := event.get("name"):
            output = event.get("data", {}).get("output")
            ui_events.append(
                {
                    "event": "tool_end",
                    "name": name,
                    "output": str(output) if output is not None else "",
                }
            )

    elif event_name == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            ui_events.append({"event": "final_token", "data": chunk.content})

    return ui_events


# --- Streaming Logic ---
async def event_streamer(
    request: Request, message: str, session_id: str
) -> AsyncGenerator[str, None]:
    config: RunnableConfig = {"configurable": {"session_id": session_id}}

    logger.info(f"Routing query: '{message[:50]}...'")
    route_result = await router_chain.ainvoke({"input": message})

    if not isinstance(route_result, RouteQuery):
        logger.error(f"Router returned unexpected type: {type(route_result)}")
        route_decision = "conversation"
    else:
        route_decision = route_result.datasource

    logger.info(f"Router decided: '{route_decision}'")

    if route_decision == "agent":
        stream_generator = agent_with_chat_history.astream_events(
            {"input": message}, config=config, version="v2"
        )
    else:

        async def conversation_event_generator():
            async for chunk in conversational_chain_with_history.astream(
                {"input": message}, config=config
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content=chunk)},
                }

        stream_generator = conversation_event_generator()

    async for event in stream_generator:
        if await request.is_disconnected():
            break
        ui_events = process_event(event)
        for ui_event in ui_events:
            yield f"data: {json.dumps(ui_event)}\n\n"

    yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"


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
