"""
Fate RAG Chatbot — FastAPI Application
Lambda-compatible via Mangum adapter.

Endpoints:
  POST /chat    — RAG-powered chat with conversation history
  GET  /health  — Health check
"""

import json
import logging
import os
from typing import Any, AsyncGenerator

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, Field

from prompt import SYSTEM_PROMPT, build_messages
from retriever import FateRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", AWS_REGION)
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20251029-v2:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fate RAG Chatbot API",
    description=(
        "A Retrieval-Augmented Generation chatbot for the Fate Series universe, "
        "powered by AWS Bedrock and OpenSearch."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared clients (initialised lazily to avoid cold-start penalty) ───────────
_retriever: FateRetriever | None = None
_bedrock_client: Any | None = None


def get_retriever() -> FateRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FateRetriever()
    return _retriever


def get_bedrock_client() -> Any:
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock_client


# ── Request / Response schemas ───────────────────────────────────────────────
class ConversationTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    category_filter: str | None = Field(
        default=None,
        description="Optional category to restrict retrieval (servant/master/lore/route/noble_phantasm)",
    )
    stream: bool = Field(default=False, description="Enable streaming response")


class RetrievedDoc(BaseModel):
    text: str
    title: str
    source_url: str
    category: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_docs: list[RetrievedDoc]
    model_id: str


# ── Generation helper ────────────────────────────────────────────────────────
def _invoke_bedrock(
    messages: list[dict[str, Any]],
    stream: bool = False,
) -> Any:
    """Call Bedrock Claude with the assembled messages."""
    client = get_bedrock_client()
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }

    try:
        if stream:
            return client.invoke_model_with_response_stream(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
        else:
            return client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
    except ClientError as e:
        logger.error(f"Bedrock invocation error: {e}")
        raise HTTPException(status_code=502, detail=f"Model invocation failed: {e}")


async def _stream_response(
    messages: list[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks from a Bedrock streaming response."""
    response = _invoke_bedrock(messages, stream=True)
    event_stream = response.get("body")

    for event in event_stream:
        chunk = event.get("chunk")
        if not chunk:
            continue
        data = json.loads(chunk["bytes"])
        if data.get("type") == "content_block_delta":
            delta = data.get("delta", {})
            text = delta.get("text", "")
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"

    yield "data: [DONE]\n\n"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "fate-rag-chatbot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
    """
    Main chat endpoint.
    1. Retrieve relevant Fate lore from OpenSearch.
    2. Build RAG-augmented prompt.
    3. Call Bedrock Claude for generation.
    4. Return answer + retrieved sources.
    """
    logger.info(f"Chat request: '{request.message[:80]}...'")

    # Step 1: Retrieve
    retriever = get_retriever()
    try:
        retrieved_docs = retriever.retrieve(
            query=request.message,
            category_filter=request.category_filter,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Retrieval service unavailable.")

    # Step 2: Build messages
    history = [{"role": t.role, "content": t.content} for t in request.conversation_history]
    messages = build_messages(
        question=request.message,
        retrieved_docs=retrieved_docs,
        conversation_history=history,
    )

    # Step 3: Generate
    if request.stream:
        return StreamingResponse(
            _stream_response(messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = _invoke_bedrock(messages, stream=False)
    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]

    return ChatResponse(
        answer=answer,
        retrieved_docs=[RetrievedDoc(**d) for d in retrieved_docs],
        model_id=BEDROCK_MODEL_ID,
    )


# ── Lambda handler ───────────────────────────────────────────────────────────
handler = Mangum(app, lifespan="off")
