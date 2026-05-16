"""
schemas.py
----------
FastAPI request and response schemas — the API contract.

Separate from models.py because:
  - models.py = internal data shapes (used by rag.py, agents.py, tools.py)
  - schemas.py = external API shapes (what clients send and receive)

This separation means internal model changes don't accidentally
break the public API contract.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class MessageTypeSchema(str, Enum):
    ANSWER   = "answer"
    FALLBACK = "fallback"
    CLARIFY  = "clarify"
    ERROR    = "error"


# ══════════════════════════════════════════════════════════════════════════════
# Chat Schemas
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    POST /chat — incoming question from a client.
    """
    question:   str            = Field(..., min_length=1, max_length=1000, description="User's HR policy question")
    session_id: Optional[str]  = Field(None, description="Session ID for multi-turn chat. Auto-generated if not provided.")
    top_k:      Optional[int]  = Field(None, ge=1, le=20, description="Number of chunks to retrieve (1–20)")
    filters:    Optional[dict] = Field(None, description="Metadata filters e.g. {'source_file': 'leave_policy.pdf'}")

    class Config:
        json_schema_extra = {
            "example": {
                "question":   "How many casual leaves am I entitled to per year?",
                "session_id": "abc-123",
                "top_k":      5
            }
        }


class SourceSchema(BaseModel):
    """A single source document chunk referenced in an answer."""
    source_file:      str            = Field(..., description="Source document filename")
    page_number:      Optional[int]  = Field(None, description="Page number in source document")
    section:          Optional[str]  = Field(None, description="Section heading if available")
    similarity_score: float          = Field(..., description="Relevance score (0.0 – 1.0)")
    excerpt:          str            = Field(..., description="Relevant chunk text excerpt (first 300 chars)")


class ChatResponse(BaseModel):
    """
    POST /chat — response returned to the client.
    """
    answer:       str                  = Field(..., description="LLM-generated answer")
    session_id:   str                  = Field(..., description="Session ID (use in next request for multi-turn)")
    message_type: MessageTypeSchema    = Field(..., description="Type of response: answer | fallback | error")
    sources:      list[SourceSchema]   = Field(default_factory=list, description="Source chunks used to generate the answer")
    source_files: list[str]            = Field(default_factory=list, description="Unique list of source document names")
    latency_ms:   Optional[float]      = Field(None, description="Total response time in milliseconds")
    timestamp:    datetime             = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "answer":       "You are entitled to 12 casual leaves per year according to the Leave Policy.",
                "session_id":   "abc-123",
                "message_type": "answer",
                "sources": [
                    {
                        "source_file":      "leave_policy.pdf",
                        "page_number":      3,
                        "section":          "Casual Leave",
                        "similarity_score": 0.92,
                        "excerpt":          "Employees are entitled to 12 casual leaves per calendar year..."
                    }
                ],
                "source_files": ["leave_policy.pdf"],
                "latency_ms":   820.5,
                "timestamp":    "2024-01-15T10:30:00Z"
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ingest Schemas
# ══════════════════════════════════════════════════════════════════════════════

class IngestRequest(BaseModel):
    """
    POST /ingest — trigger document ingestion.
    """
    reset: bool = Field(
        False,
        description="If true, wipe the vector store before ingesting. Use when documents have changed significantly."
    )
    filename: Optional[str] = Field(
        None,
        description="Ingest a specific file by name. If not provided, all documents in HR_DOCS_PATH are ingested."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reset":    False,
                "filename": "leave_policy.pdf"
            }
        }


class IngestFileResultSchema(BaseModel):
    """Result for a single ingested file."""
    filename:      str           = Field(..., description="Name of the file")
    success:       bool          = Field(..., description="Whether ingestion succeeded")
    total_chunks:  int           = Field(..., description="Number of chunks stored")
    error_message: Optional[str] = Field(None, description="Error details if failed")


class IngestResponse(BaseModel):
    """
    POST /ingest — ingestion summary returned to the client.
    """
    total_files:      int                        = Field(..., description="Total files processed")
    successful_files: int                        = Field(..., description="Successfully ingested files")
    failed_files:     int                        = Field(..., description="Failed files")
    total_chunks:     int                        = Field(..., description="Total chunks stored in vector store")
    results:          list[IngestFileResultSchema] = Field(default_factory=list, description="Per-file results")
    timestamp:        datetime                   = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "total_files":      3,
                "successful_files": 3,
                "failed_files":     0,
                "total_chunks":     142,
                "results": [
                    {"filename": "leave_policy.pdf",   "success": True, "total_chunks": 48,  "error_message": None},
                    {"filename": "code_of_conduct.pdf","success": True, "total_chunks": 61,  "error_message": None},
                    {"filename": "travel_policy.docx", "success": True, "total_chunks": 33,  "error_message": None},
                ],
                "timestamp": "2024-01-15T10:00:00Z"
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# Health Check Schema
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """
    GET /health — service health check.
    Used by AWS ALB, ECS, and monitoring tools to verify the service is up.
    """
    status:         str            = Field(..., description="'healthy' or 'degraded'")
    version:        str            = Field("1.0.0", description="API version")
    vector_store:   dict           = Field(default_factory=dict, description="Vector store stats")
    llm_model:      str            = Field(..., description="Active LLM model name")
    embedding_model: str           = Field(..., description="Active embedding model name")
    timestamp:      datetime       = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status":          "healthy",
                "version":         "1.0.0",
                "vector_store":    {"store_type": "chroma", "total_chunks": 142},
                "llm_model":       "llama-3.1-8b-instant",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "timestamp":       "2024-01-15T10:30:00Z"
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# Error Schema
# ══════════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """
    Standard error response for all API errors.
    Returned with appropriate HTTP status codes (400, 422, 500, etc.)
    """
    error:      str            = Field(..., description="Short error type e.g. 'validation_error'")
    message:    str            = Field(..., description="Human-readable error message")
    details:    Optional[str]  = Field(None, description="Additional debug details")
    timestamp:  datetime       = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error":     "validation_error",
                "message":   "Question cannot be empty.",
                "details":   None,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
