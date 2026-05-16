"""
tools.py
--------
Tool functions available to the HR Policy Agent (agents.py).
Each tool wraps a specific capability the agent can invoke:
  - Searching HR policy documents (core RAG tool)
  - Listing available policy documents
  - Getting collection stats

Tools are plain Python functions here — agents.py decides when to call them
based on the user's query. This keeps tools testable independently of the agent.
"""

import logging
from typing import Optional

from config import ingest_config, retrieval_config
from models import RetrievedChunk
from prompts import format_context
from rag import retrieve, get_collection_stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: Search HR Policy Documents (Core RAG Tool)
# ══════════════════════════════════════════════════════════════════════════════

def search_hr_policy(
    query: str,
    top_k: Optional[int] = None,
    source_file: Optional[str] = None,
) -> dict:
    """
    Search the HR policy vector store for chunks relevant to the query.
    This is the primary tool — called for almost every user question.

    Args:
        query:       The user's question or a refined search query
        top_k:       Number of results to fetch (defaults to config TOP_K)
        source_file: Optional — restrict search to a specific policy document
                     e.g. "leave_policy.pdf"

    Returns:
        {
            "found":        bool,
            "chunks":       list[RetrievedChunk],
            "context":      str,   # formatted context string for the LLM prompt
            "source_files": list[str],
            "top_score":    float
        }
    """
    logger.info(f"[Tool: search_hr_policy] Query: '{query[:80]}'")

    # Build optional metadata filter
    filters = {"source_file": source_file} if source_file else None

    chunks: list[RetrievedChunk] = retrieve(
        query=query,
        top_k=top_k or retrieval_config.TOP_K,
        filters=filters
    )

    if not chunks:
        logger.info("[Tool: search_hr_policy] No relevant chunks found.")
        return {
            "found":        False,
            "chunks":       [],
            "context":      "",
            "source_files": [],
            "top_score":    0.0
        }

    context = format_context(chunks)
    source_files = list({c.source_file for c in chunks})
    top_score    = chunks[0].similarity_score

    logger.info(
        f"[Tool: search_hr_policy] Found {len(chunks)} chunks. "
        f"Top score: {top_score:.4f}. Sources: {source_files}"
    )

    return {
        "found":        True,
        "chunks":       chunks,
        "context":      context,
        "source_files": source_files,
        "top_score":    top_score
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: List Available Policy Documents
# ══════════════════════════════════════════════════════════════════════════════

def list_policy_documents() -> dict:
    """
    Return a list of HR policy documents available in the docs folder.
    Useful when the user asks "what policies do you have?" or
    "which documents are available?"

    Returns:
        {
            "documents": list[str],   # filenames
            "count":     int
        }
    """
    logger.info("[Tool: list_policy_documents] Listing available HR documents.")

    docs_path = ingest_config.HR_DOCS_PATH
    supported = ingest_config.SUPPORTED_EXTENSIONS

    if not docs_path.exists():
        logger.warning(f"HR docs folder not found: {docs_path}")
        return {"documents": [], "count": 0}

    files = [
        f.name
        for f in docs_path.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ]
    files.sort()

    logger.info(f"[Tool: list_policy_documents] Found {len(files)} document(s).")
    return {
        "documents": files,
        "count":     len(files)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: Get Vector Store Stats
# ══════════════════════════════════════════════════════════════════════════════

def get_vector_store_stats() -> dict:
    """
    Return stats about the vector store — total chunks, store type, path.
    Useful for admin/debug purposes or a health check page.

    Returns:
        {
            "store_type":   str,
            "total_chunks": int,
            "persist_path": str
        }
    """
    logger.info("[Tool: get_vector_store_stats] Fetching vector store stats.")
    stats = get_collection_stats()
    logger.info(f"[Tool: get_vector_store_stats] Stats: {stats}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registry
# Used by agents.py to know which tools exist and what they do
# ══════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "search_hr_policy": {
        "fn":          search_hr_policy,
        "description": (
            "Search HR policy documents for information relevant to the user's question. "
            "Use this for any question about leave, benefits, reimbursements, "
            "code of conduct, resignation, performance, or any other HR topic."
        ),
        "required_args": ["query"]
    },
    "list_policy_documents": {
        "fn":          list_policy_documents,
        "description": (
            "List all available HR policy documents. "
            "Use when the user asks what policies are available or wants to know "
            "which documents the assistant can reference."
        ),
        "required_args": []
    },
    "get_vector_store_stats": {
        "fn":          get_vector_store_stats,
        "description": (
            "Get stats about the vector store (total chunks, store type). "
            "Use for admin or debug queries."
        ),
        "required_args": []
    }
}


def get_tool(name: str):
    """
    Fetch a tool function by name from the registry.

    Args:
        name: Tool name e.g. 'search_hr_policy'

    Returns:
        The tool function

    Raises:
        KeyError: If tool name is not in registry
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    return TOOL_REGISTRY[name]["fn"]
