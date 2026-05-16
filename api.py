"""
api.py
------
FastAPI application for the HR Policy RAG Chatbot.

Endpoints:
  GET  /             → API info
  GET  /health       → Health check (used by AWS ALB / ECS)
  POST /chat         → Ask an HR policy question
  POST /ingest       → Trigger document ingestion
  GET  /stats        → Vector store stats

Run locally:
    uvicorn api:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import ask
from config import app_config, embedding_config, llm_config, validate_all
from ingest import ingest_all, ingest_file, reset_vector_store
from models import ChatHistory, MessageType, QueryRequest, Role
from rag import get_collection_stats
from schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestFileResultSchema,
    IngestRequest,
    IngestResponse,
    MessageTypeSchema,
    SourceSchema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# In-memory session store
# Maps session_id → ChatHistory
# For production, swap this with Redis or a database
# ══════════════════════════════════════════════════════════════════════════════

_sessions: dict[str, ChatHistory] = {}

MAX_SESSION_MESSAGES = 20   # trim history beyond this to control context window


def get_or_create_session(session_id: str) -> ChatHistory:
    """Retrieve existing session or create a new one."""
    if session_id not in _sessions:
        _sessions[session_id] = ChatHistory(session_id=session_id)
        logger.info(f"New session created: {session_id}")
    return _sessions[session_id]


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown logic
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before serving traffic."""
    logger.info("Starting HR Policy RAG API...")
    try:
        validate_all()
        logger.info("Configuration validated.")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise RuntimeError(f"Startup failed: {e}")

    logger.info(f"Model     : {llm_config.MODEL}")
    logger.info(f"Embeddings: {embedding_config.MODEL}")
    logger.info("API ready.")

    yield   # app runs here

    logger.info("Shutting down HR Policy RAG API.")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = app_config.TITLE,
    description = (
        "REST API for the HR Policy RAG Chatbot. "
        "Ask questions about HR policies grounded in your company's documents."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# CORS Middleware
# Allows the Streamlit app (app.py) and any frontend to call this API
# ══════════════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],    # tighten this in production to your domain
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Global Exception Handler
# Returns consistent ErrorResponse for any unhandled exception
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = ErrorResponse(
            error   = "internal_server_error",
            message = "An unexpected error occurred. Please try again.",
            details = str(exc) if not app_config.is_production() else None
        ).model_dump(mode="json")
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helper: convert internal QueryResponse → ChatResponse (API schema)
# ══════════════════════════════════════════════════════════════════════════════

def _to_chat_response(response, session_id: str) -> ChatResponse:
    """Map internal QueryResponse to the external ChatResponse schema."""
    sources = [
        SourceSchema(
            source_file      = chunk.source_file,
            page_number      = chunk.page_number,
            section          = chunk.section,
            similarity_score = chunk.similarity_score,
            excerpt          = chunk.content[:300]
        )
        for chunk in response.sources
    ]

    return ChatResponse(
        answer       = response.answer,
        session_id   = session_id,
        message_type = MessageTypeSchema(response.message_type.value),
        sources      = sources,
        source_files = response.source_files,
        latency_ms   = response.latency_ms,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
async def root():
    """API info and available endpoints."""
    return {
        "name":        app_config.TITLE,
        "version":     "1.0.0",
        "description": "HR Policy RAG Chatbot API",
        "endpoints": {
            "chat":   "POST /chat",
            "ingest": "POST /ingest",
            "health": "GET  /health",
            "stats":  "GET  /stats",
            "docs":   "GET  /docs",
        }
    }


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["General"],
    summary        = "Health check — used by AWS ALB and ECS"
)
async def health_check():
    """
    Returns service health status.
    AWS ALB expects HTTP 200 from this endpoint to route traffic.
    Returns 503 if the vector store is empty (not yet ingested).
    """
    try:
        stats = get_collection_stats()
    except Exception as e:
        logger.warning(f"Health check — vector store error: {e}")
        stats = {"error": str(e)}

    total_chunks = stats.get("total_chunks", 0)
    status_str   = "healthy" if total_chunks and total_chunks > 0 else "degraded"

    return HealthResponse(
        status          = status_str,
        vector_store    = stats,
        llm_model       = llm_config.MODEL,
        embedding_model = embedding_config.MODEL,
    )


# ── Chat ─────────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model  = ChatResponse,
    responses       = {
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags            = ["Chat"],
    summary         = "Ask an HR policy question"
)
async def chat(request: ChatRequest):
    """
    Submit an HR policy question and receive a grounded answer.

    - Retrieves relevant chunks from the HR policy vector store
    - Generates an answer using Llama 3 via Groq
    - Returns the answer with source citations

    **Multi-turn conversations**: pass the `session_id` from the previous
    response back in the next request to maintain conversation context.
    """
    # Auto-generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Validate question
    if not request.question.strip():
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = ErrorResponse(
                error   = "empty_question",
                message = "Question cannot be empty."
            ).model_dump(mode="json")
        )

    # Get or create session history
    history = get_or_create_session(session_id)

    # Trim history if too long (context window management)
    if len(history.messages) > MAX_SESSION_MESSAGES:
        history.messages = history.messages[-MAX_SESSION_MESSAGES:]

    # Add user message to history
    history.add(Role.USER, request.question)

    # Build internal request
    query_request = QueryRequest(
        question   = request.question,
        session_id = session_id,
        top_k      = request.top_k,
        filters    = request.filters,
    )

    # Run through RAG agent
    try:
        response = ask(query_request, history)
    except RuntimeError as e:
        logger.error(f"Agent error for session {session_id}: {e}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = ErrorResponse(
                error   = "agent_error",
                message = str(e)
            ).model_dump(mode="json")
        )

    # Add assistant reply to history
    history.add(Role.ASSISTANT, response.answer)

    logger.info(
        f"[/chat] session={session_id} | "
        f"type={response.message_type.value} | "
        f"sources={len(response.sources)} | "
        f"latency={response.latency_ms}ms"
    )

    return _to_chat_response(response, session_id)


# ── Ingest ───────────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model = IngestResponse,
    responses      = {
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags           = ["Admin"],
    summary        = "Ingest HR policy documents into the vector store"
)
async def ingest(request: IngestRequest):
    """
    Trigger document ingestion from the HR docs folder.

    - Loads PDF, DOCX, TXT, and MD files
    - Chunks and embeds them into the vector store
    - Use `reset=true` to wipe and rebuild from scratch
    - Use `filename` to ingest a single specific file

    **Note**: This is an admin endpoint. In production, protect it
    with an API key or restrict access to internal networks only.
    """
    try:
        # Optional reset
        if request.reset:
            logger.info("[/ingest] Resetting vector store...")
            reset_vector_store()

        # Single file mode
        if request.filename:
            from config import ingest_config
            file_path = ingest_config.HR_DOCS_PATH / request.filename
            if not file_path.exists():
                raise HTTPException(
                    status_code = status.HTTP_400_BAD_REQUEST,
                    detail      = ErrorResponse(
                        error   = "file_not_found",
                        message = f"File '{request.filename}' not found in HR docs folder."
                    ).model_dump(mode="json")
                )
            result  = ingest_file(file_path)
            summary_results = [result]
            total_files      = 1
            successful_files = 1 if result.success else 0
            failed_files     = 0 if result.success else 1
            total_chunks     = result.total_chunks

        # All files mode
        else:
            summary = ingest_all()
            summary_results  = summary.results
            total_files      = summary.total_files
            successful_files = summary.successful_files
            failed_files     = summary.failed_files
            total_chunks     = summary.total_chunks

        logger.info(
            f"[/ingest] files={total_files} | "
            f"ok={successful_files} | "
            f"fail={failed_files} | "
            f"chunks={total_chunks}"
        )

        return IngestResponse(
            total_files      = total_files,
            successful_files = successful_files,
            failed_files     = failed_files,
            total_chunks     = total_chunks,
            results          = [
                IngestFileResultSchema(
                    filename      = r.filename,
                    success       = r.success,
                    total_chunks  = r.total_chunks,
                    error_message = r.error_message
                )
                for r in summary_results
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/ingest] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = ErrorResponse(
                error   = "ingest_error",
                message = f"Ingestion failed: {str(e)}"
            ).model_dump(mode="json")
        )


# ── Stats ────────────────────────────────────────────────────────────────────

@app.get(
    "/stats",
    tags    = ["Admin"],
    summary = "Vector store statistics"
)
async def stats():
    """Return vector store stats — total chunks, store type, path."""
    try:
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = ErrorResponse(
                error   = "stats_error",
                message = str(e)
            ).model_dump(mode="json")
        )
