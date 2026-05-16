"""
exceptions.py
-------------
Custom exception hierarchy for the HR Policy RAG Chatbot.
Raising specific exceptions instead of generic ones means:
  - Cleaner error messages for the user
  - Easier debugging (you know exactly where it broke)
  - app.py can catch each type and respond differently
"""


# ══════════════════════════════════════════════════════════════════════════════
# Base Exception
# ══════════════════════════════════════════════════════════════════════════════

class HRChatbotError(Exception):
    """
    Base class for all HR Chatbot exceptions.
    Catch this if you want to handle any chatbot error in one place.
    """
    def __init__(self, message: str, details: str = ""):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ══════════════════════════════════════════════════════════════════════════════
# Configuration Errors
# ══════════════════════════════════════════════════════════════════════════════

class ConfigurationError(HRChatbotError):
    """
    Raised when a required config value is missing or invalid.
    Typically caught at startup in app.py — fail fast before any user traffic.

    Example:
        raise ConfigurationError("GROQ_API_KEY is not set", "Add it to your .env file")
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Errors
# ══════════════════════════════════════════════════════════════════════════════

class IngestionError(HRChatbotError):
    """
    Base class for errors during document ingestion (ingest.py).
    """
    pass


class UnsupportedFileTypeError(IngestionError):
    """
    Raised when a file in the HR docs folder has an unsupported extension.

    Example:
        raise UnsupportedFileTypeError("File type not supported", "File: report.xlsx")
    """
    pass


class DocumentLoadError(IngestionError):
    """
    Raised when a supported file exists but cannot be read/parsed.
    e.g. corrupted PDF, password-protected DOCX.

    Example:
        raise DocumentLoadError("Failed to load document", "File: leave_policy.pdf — may be corrupted")
    """
    pass


class ChunkingError(IngestionError):
    """
    Raised when text splitting/chunking fails unexpectedly.

    Example:
        raise ChunkingError("Chunking failed", "Empty document after extraction")
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Vector Store Errors
# ══════════════════════════════════════════════════════════════════════════════

class VectorStoreError(HRChatbotError):
    """
    Base class for vector store errors (rag.py).
    """
    pass


class VectorStoreNotInitializedError(VectorStoreError):
    """
    Raised when a query is attempted before documents have been ingested.
    Helps give a clear message: "Run ingest.py first."

    Example:
        raise VectorStoreNotInitializedError(
            "Vector store is empty",
            "Run `python ingest.py` to load HR policy documents first."
        )
    """
    pass


class EmbeddingError(VectorStoreError):
    """
    Raised when the embedding model fails to encode text.

    Example:
        raise EmbeddingError("Failed to generate embeddings", f"Input length: {len(text)} chars")
    """
    pass


class RetrievalError(VectorStoreError):
    """
    Raised when similarity search fails (not the same as returning 0 results).
    0 results = normal; a crash during search = this exception.

    Example:
        raise RetrievalError("Vector search failed", str(e))
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# LLM / Agent Errors
# ══════════════════════════════════════════════════════════════════════════════

class LLMError(HRChatbotError):
    """
    Base class for LLM-related errors (agents.py).
    """
    pass


class LLMConnectionError(LLMError):
    """
    Raised when the Groq API is unreachable or returns a connection error.

    Example:
        raise LLMConnectionError("Could not connect to Groq API", "Check your internet connection")
    """
    pass


class LLMAuthenticationError(LLMError):
    """
    Raised when the Groq API key is invalid or expired.

    Example:
        raise LLMAuthenticationError("Invalid Groq API key", "Update GROQ_API_KEY in .env")
    """
    pass


class LLMRateLimitError(LLMError):
    """
    Raised when Groq's rate limit is hit.
    The free tier has limits — this helps surface that clearly.

    Example:
        raise LLMRateLimitError("Groq rate limit exceeded", "Retry after a few seconds")
    """
    pass


class LLMResponseError(LLMError):
    """
    Raised when the LLM returns an unexpected or unparseable response.

    Example:
        raise LLMResponseError("Unexpected LLM response format", f"Raw response: {raw}")
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Query / RAG Pipeline Errors
# ══════════════════════════════════════════════════════════════════════════════

class QueryError(HRChatbotError):
    """
    Base class for errors in the query pipeline.
    """
    pass


class EmptyQueryError(QueryError):
    """
    Raised when the user submits an empty or whitespace-only question.

    Example:
        raise EmptyQueryError("Question cannot be empty")
    """
    pass


class NoResultsError(QueryError):
    """
    Raised when retrieval returns zero chunks above the similarity threshold.
    This is a soft error — app.py should respond with a helpful fallback message
    rather than showing a generic error to the user.

    Example:
        raise NoResultsError(
            "No relevant HR policy found",
            "Query: 'What is the stock vesting schedule?'"
        )
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Helper: map Groq API errors → our custom exceptions
# ══════════════════════════════════════════════════════════════════════════════

def handle_groq_error(error: Exception) -> HRChatbotError:
    """
    Converts a raw Groq/OpenAI SDK exception into a typed HRChatbotError.
    Used in agents.py inside the LLM call try/except block.

    Usage:
        except Exception as e:
            raise handle_groq_error(e) from e
    """
    error_str = str(error).lower()

    if "authentication" in error_str or "api key" in error_str or "401" in error_str:
        return LLMAuthenticationError(
            "Invalid or missing Groq API key.",
            "Update GROQ_API_KEY in your .env file."
        )
    if "rate limit" in error_str or "429" in error_str:
        return LLMRateLimitError(
            "Groq rate limit hit.",
            "Wait a few seconds and try again, or switch to a larger model tier."
        )
    if "connection" in error_str or "timeout" in error_str or "network" in error_str:
        return LLMConnectionError(
            "Could not reach the Groq API.",
            "Check your internet connection and retry."
        )

    return LLMResponseError(
        "Unexpected error from Groq API.",
        str(error)
    )
