import asyncio
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

from app.core.security import api_key_auth
from app.core.tools import retrieve_context, get_order_status

app = FastAPI(title="Next-Gen Support Agent API")


class HealthResponse(BaseModel):
    status: bool


class ChatRequest(BaseModel):
    message: str


tools = [retrieve_context, get_order_status]
prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


async def agent_final_answer_streamer(message: str) -> AsyncGenerator[str, None]:
    """
    Streams the final answer from the agent with a thinking animation
    followed by character-by-character streaming for better UX.
    """
    # Send thinking indicator first
    yield "🤔 Thinking..."

    # Get the complete response from the agent (this takes time)
    result = await agent_executor.ainvoke({"input": message})
    final_answer = result.get("output", "")

    # Clear the thinking message and start streaming the real response
    yield "\r"  # Carriage return to overwrite

    # Stream the response character by character
    for char in final_answer:
        yield char
        # Add a small delay to simulate human-like typing
        # Adjust this value to make it faster/slower
        await asyncio.sleep(0.02)  # 50 chars per second


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return {"status": True}


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(
    request: ChatRequest, api_key: str = Depends(api_key_auth)
):
    return StreamingResponse(
        agent_final_answer_streamer(request.message), media_type="text/plain"
    )
