# app/main.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local imports
from app.core.security import api_key_auth
from app.core.retriever import get_retriever

# Create the FastAPI app instance
app = FastAPI(title="Next-Gen Support Agent API")


# --- Pydantic Models ---
class HealthResponse(BaseModel):
    status: bool


class ChatRequest(BaseModel):
    message: str


# --- RAG Chain Setup ---
# 1. Create the prompt template
template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 2. Initialize the LLM
# LangSmith will automatically track this
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 3. Get our advanced retriever
retriever = get_retriever()

# 4. Define the RAG chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Endpoint to check if the API is running."""
    return {"status": True}


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(
    request: ChatRequest, api_key: str = Depends(api_key_auth)
):
    """
    Receives a message and streams the RAG chain's response back.
    """
    return StreamingResponse(
        rag_chain.stream(request.message),  # The chain is now the source of the stream
        media_type="text/plain",
    )
