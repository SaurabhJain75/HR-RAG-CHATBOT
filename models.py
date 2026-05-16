"""
models.py
---------
Pydantic data models / schemas for the HR Policy RAG Chatbot.
These are shared across rag.py, agents.py, tools.py, and app.py.
Think of this as the "contract" between all layers of the app.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class Role(str, Enum):
    """Who sent a chat message."""
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class MessageType(str, Enum):
    """What kind of response the assistant gave."""
    ANSWER     = "answer"       # Normal RAG answer grounded in docs
    FALLBACK   = "fallback"     # No relevant docs found
    CLARIFY    = "clarify"      # Agent asked user to clarify
    ERROR      = "error"        # Something went wrong


# ══════════════════════════════════════════════════════════════════════════════
# Document / Chunk Models
# ══════════════════════════════════════════════════════════════════════════════

class DocumentChunk(BaseModel):
    """
    A single chunk of text extracted from an HR policy document.
    Stored in the vector store alongside its embedding.
    """
    chunk_id:    str            = Field(..., description="Unique ID: '<filename>_chunk_<n>'")
    source_file: str            = Field(..., description="Original filename e.g. 'leave_policy.pdf'")
    page_number: Optional[int]  = Field(None, description="Page number in source document")
    section:     Optional[str]  = Field(None, description="Section heading if detectable")
    content:     str            = Field(..., description="Raw text of this chunk")
    metadata:    dict           = Field(default_factory=dict, description="Extra metadata for filtering")

    class Config:
        extra = "allow"


class RetrievedChunk(DocumentChunk):
    """
    A DocumentChunk returned from vector similarity search,
    enriched with its similarity score.
    """
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 – 1.0)")


# ══════════════════════════════════════════════════════════════════════════════
# Chat Message Models
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role:       Role      = Field(..., description="user | assistant | system")
    content:    str       = Field(..., description="Text content of the message")
    timestamp:  datetime  = Field(default_factory=datetime.utcnow)

    def to_llm_dict(self) -> dict:
        """Format for sending to the LLM API (role + content only)."""
        return {"role": self.role.value, "content": self.content}


class ChatHistory(BaseModel):
    """Full conversation history for a session."""
    session_id: str              = Field(..., description="Unique session identifier")
    messages:   list[ChatMessage] = Field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))

    def to_llm_messages(self) -> list[dict]:
        """Return all messages formatted for the LLM API."""
        return [m.to_llm_dict() for m in self.messages]

    def last_n(self, n: int) -> list[ChatMessage]:
        """Return the last N messages (for context window management)."""
        return self.messages[-n:]

    def clear(self) -> None:
        self.messages = []


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    Incoming user query — the input to the RAG pipeline.
    """
    question:   str            = Field(..., min_length=1, description="User's HR policy question")
    session_id: str            = Field(..., description="Session ID to track conversation")
    top_k:      Optional[int]  = Field(None, description="Override default TOP_K for retrieval")
    filters:    Optional[dict] = Field(None, description="Metadata filters e.g. {'department': 'engineering'}")


class QueryResponse(BaseModel):
    """
    Final response returned to the user after RAG + LLM synthesis.
    """
    answer:       str                   = Field(..., description="LLM-generated answer")
    sources:      list[RetrievedChunk]  = Field(default_factory=list, description="Chunks used to generate the answer")
    message_type: MessageType           = Field(MessageType.ANSWER, description="Type of response")
    session_id:   str                   = Field(..., description="Echo back the session ID")
    latency_ms:   Optional[float]       = Field(None, description="Total response time in milliseconds")

    @property
    def source_files(self) -> list[str]:
        """Unique list of source files referenced in the answer."""
        return list({c.source_file for c in self.sources})


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Models
# ══════════════════════════════════════════════════════════════════════════════

class IngestResult(BaseModel):
    """
    Result of ingesting one HR policy document into the vector store.
    """
    filename:       str  = Field(..., description="Name of the file ingested")
    total_chunks:   int  = Field(..., description="Number of chunks created")
    success:        bool = Field(..., description="Whether ingestion succeeded")
    error_message:  Optional[str] = Field(None, description="Error details if failed")


class IngestSummary(BaseModel):
    """
    Aggregated result after ingesting all documents in the HR docs folder.
    """
    results:          list[IngestResult] = Field(default_factory=list)
    total_files:      int = 0
    successful_files: int = 0
    failed_files:     int = 0
    total_chunks:     int = 0

    def add_result(self, result: IngestResult) -> None:
        self.results.append(result)
        self.total_files += 1
        if result.success:
            self.successful_files += 1
            self.total_chunks += result.total_chunks
        else:
            self.failed_files += 1
