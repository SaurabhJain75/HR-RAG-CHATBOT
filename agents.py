"""
agents.py
---------
The orchestration layer of the HR Policy RAG Chatbot.
Responsible for:
  1. Initializing the Groq LLM client
  2. Deciding which tool to call based on the user's query
  3. Building the final prompt with retrieved context
  4. Calling the LLM and returning a structured QueryResponse

This is the "brain" — it connects tools.py (retrieval) with the LLM (generation).

Used by:
  - app.py → calls ask() for every user message
"""

import logging
import time
from typing import Optional

from openai import OpenAI  # Groq uses OpenAI-compatible SDK

from config import llm_config
from models import (
    ChatHistory,
    MessageType,
    QueryRequest,
    QueryResponse,
    Role,
)
from prompts import (
    FALLBACK_MESSAGE,
    SYSTEM_PROMPT,
    WELCOME_MESSAGE,
    build_fallback_prompt,
    build_rag_prompt,
)
from tools import list_policy_documents, search_hr_policy

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Groq Client (singleton)
# ══════════════════════════════════════════════════════════════════════════════

_groq_client: Optional[OpenAI] = None


def get_groq_client() -> OpenAI:
    """
    Initialize the Groq client once and reuse it.
    Groq is OpenAI-compatible — we just point the base_url to Groq's endpoint.
    """
    global _groq_client
    if _groq_client is None:
        if not llm_config.GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Get your free key at https://console.groq.com/keys and add it to .env"
            )
        _groq_client = OpenAI(
            api_key=llm_config.GROQ_API_KEY,
            base_url=llm_config.BASE_URL,   # https://api.groq.com/openai/v1
        )
        logger.info(f"Groq client initialized. Model: {llm_config.MODEL}")
    return _groq_client


# ══════════════════════════════════════════════════════════════════════════════
# LLM Call
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(messages: list[dict]) -> str:
    """
    Send a list of messages to Groq and return the assistant's reply.

    Args:
        messages: List of {"role": ..., "content": ...} dicts

    Returns:
        Assistant reply as a plain string

    Raises:
        RuntimeError: On API errors (auth, rate limit, connection)
    """
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model       = llm_config.MODEL,
            messages    = messages,
            temperature = llm_config.TEMPERATURE,
            max_tokens  = llm_config.MAX_TOKENS,
        )
        reply = response.choices[0].message.content.strip()
        logger.info(f"LLM reply received ({len(reply)} chars).")
        return reply

    except Exception as e:
        error_str = str(e).lower()

        if "authentication" in error_str or "401" in error_str:
            raise RuntimeError(
                "Invalid Groq API key. Update GROQ_API_KEY in your .env file."
            ) from e
        if "rate limit" in error_str or "429" in error_str:
            raise RuntimeError(
                "Groq rate limit exceeded. Wait a few seconds and try again."
            ) from e
        if "connection" in error_str or "timeout" in error_str:
            raise RuntimeError(
                "Could not connect to Groq API. Check your internet connection."
            ) from e

        raise RuntimeError(f"Unexpected Groq API error: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
# Intent Detection
# Decide which tool (if any) to call based on the user's question
# ══════════════════════════════════════════════════════════════════════════════

def detect_intent(question: str) -> str:
    """
    Simple keyword-based intent detection.
    Returns the tool name to call, or 'search' as the default.

    Intents:
      - "list_docs"  → user wants to know what policies are available
      - "search"     → default, search the vector store (covers 95% of queries)

    Args:
        question: The user's raw question

    Returns:
        Intent string: "list_docs" | "search"
    """
    q = question.lower().strip()

    list_triggers = [
        "what policies", "which policies", "what documents",
        "which documents", "list policies", "show policies",
        "available policies", "what can you help", "what topics",
        "what do you know about"
    ]

    if any(trigger in q for trigger in list_triggers):
        logger.info("[Intent] → list_docs")
        return "list_docs"

    logger.info("[Intent] → search")
    return "search"


# ══════════════════════════════════════════════════════════════════════════════
# Core Agent: ask()
# The single entry point called by app.py for every user message
# ══════════════════════════════════════════════════════════════════════════════

def ask(request: QueryRequest, history: ChatHistory) -> QueryResponse:
    """
    Process a user question through the full RAG pipeline:
      1. Detect intent
      2. Call the appropriate tool
      3. Build the LLM prompt with retrieved context
      4. Call Groq and return a structured response

    Args:
        request: QueryRequest with question, session_id, optional filters
        history: ChatHistory for this session (for multi-turn context)

    Returns:
        QueryResponse with answer, sources, message_type, latency
    """
    start_time = time.time()
    question   = request.question.strip()

    logger.info(f"[Agent] Question: '{question[:100]}'")

    # ── Guard: empty question ────────────────────────────────────────────────
    if not question:
        return QueryResponse(
            answer       = "Please type a question and I'll do my best to help!",
            sources      = [],
            message_type = MessageType.CLARIFY,
            session_id   = request.session_id,
            latency_ms   = 0.0
        )

    # ── Step 1: Detect intent ────────────────────────────────────────────────
    intent = detect_intent(question)

    # ── Step 2: Handle list_docs intent (no LLM needed) ─────────────────────
    if intent == "list_docs":
        result = list_policy_documents()
        if result["count"] == 0:
            answer = (
                "No HR policy documents have been loaded yet. "
                "Please contact your HR administrator."
            )
        else:
            docs   = "\n".join(f"- {d}" for d in result["documents"])
            answer = (
                f"I have access to the following {result['count']} HR policy document(s):\n\n"
                f"{docs}\n\n"
                "Feel free to ask me anything about these policies!"
            )
        return QueryResponse(
            answer       = answer,
            sources      = [],
            message_type = MessageType.ANSWER,
            session_id   = request.session_id,
            latency_ms   = round((time.time() - start_time) * 1000, 2)
        )

    # ── Step 3: Search HR policy vector store ───────────────────────────────
    search_result = search_hr_policy(
        query       = question,
        top_k       = request.top_k,
        source_file = (request.filters or {}).get("source_file")
    )

    # ── Step 4a: No results → fallback ──────────────────────────────────────
    if not search_result["found"]:
        logger.info("[Agent] No relevant chunks found — using fallback.")

        # Option A: Static fallback message (fast, no extra LLM call)
        answer = FALLBACK_MESSAGE

        # Option B: LLM-generated fallback (more empathetic, costs one extra call)
        # Uncomment if you prefer a softer, LLM-written response:
        # fallback_prompt = build_fallback_prompt(question)
        # answer = call_llm([
        #     {"role": "system", "content": SYSTEM_PROMPT},
        #     {"role": "user",   "content": fallback_prompt}
        # ])

        return QueryResponse(
            answer       = answer,
            sources      = [],
            message_type = MessageType.FALLBACK,
            session_id   = request.session_id,
            latency_ms   = round((time.time() - start_time) * 1000, 2)
        )

    # ── Step 4b: Build RAG prompt with retrieved context ────────────────────
    rag_prompt = build_rag_prompt(
        context  = search_result["context"],
        question = question
    )

    # ── Step 5: Build message list for LLM ──────────────────────────────────
    # Structure:
    #   [system prompt]
    #   [last N turns of conversation for multi-turn context]
    #   [current RAG prompt as the latest user message]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include last 6 messages (3 turns) for conversational context
    # Exclude the very last user message — we'll send the enriched RAG prompt instead
    recent = history.last_n(6)
    for msg in recent[:-1]:   # skip last user message (already in rag_prompt)
        messages.append(msg.to_llm_dict())

    messages.append({"role": "user", "content": rag_prompt})

    # ── Step 6: Call Groq LLM ───────────────────────────────────────────────
    try:
        answer = call_llm(messages)
        message_type = MessageType.ANSWER
    except RuntimeError as e:
        logger.error(f"[Agent] LLM call failed: {e}")
        answer       = f"I'm sorry, I encountered an error while processing your request. Please try again. ({e})"
        message_type = MessageType.ERROR

    latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"[Agent] Response ready in {latency_ms}ms.")

    return QueryResponse(
        answer       = answer,
        sources      = search_result["chunks"],
        message_type = message_type,
        session_id   = request.session_id,
        latency_ms   = latency_ms
    )


# ══════════════════════════════════════════════════════════════════════════════
# Welcome Message Helper
# Called by app.py on session start
# ══════════════════════════════════════════════════════════════════════════════

def get_welcome_message() -> str:
    """Return the welcome message shown when a new chat session starts."""
    return WELCOME_MESSAGE
