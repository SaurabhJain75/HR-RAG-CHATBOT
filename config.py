"""
config.py
---------
Central configuration for the HR Policy RAG Chatbot.
Reads all values from .env and exposes them as typed constants.
Every other file imports from here — never read .env directly elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── HuggingFace Spaces persistent storage ──────────────────────────────────────
# HF Spaces provides /data as persistent volume — use for vectorstore
# Falls back to BASE_DIR for local/Docker runs
HF_DATA_DIR = Path("/data") if Path("/data").exists() else BASE_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _require(key: str) -> str:
    """Read a required env variable; raise early if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[config] Required environment variable '{key}' is not set. "
            f"Please add it to your .env file."
        )
    return value


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

class LLMConfig:
    GROQ_API_KEY: str  = os.getenv("GROQ_API_KEY", "")

    # Default: Llama 3.1 8B — best balance of speed + instruction-following for HR Q&A
    # Other good Groq models:
    #   llama-3.3-70b-versatile   → higher quality, slightly slower
    #   mixtral-8x7b-32768        → 32k token context window
    #   gemma2-9b-it              → Google Gemma 2
    MODEL: str         = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    MAX_TOKENS: int    = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # Groq uses OpenAI-compatible API — works with openai SDK out of the box
    BASE_URL: str      = "https://api.groq.com/openai/v1"

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise EnvironmentError(
                "[config] GROQ_API_KEY is not set. "
                "Get your free key at https://console.groq.com/keys"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Vector Store
# ══════════════════════════════════════════════════════════════════════════════

class VectorStoreConfig:
    TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")        # chroma | faiss
    PATH: Path = HF_DATA_DIR / os.getenv("VECTOR_STORE_PATH", "vectorstore")

    @classmethod
    def validate(cls):
        allowed = {"chroma", "faiss"}
        if cls.TYPE not in allowed:
            raise EnvironmentError(
                f"[config] VECTOR_STORE_TYPE must be one of {allowed}, got '{cls.TYPE}'."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Embeddings
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingConfig:
    MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Document Ingestion
# ══════════════════════════════════════════════════════════════════════════════

class IngestConfig:
    HR_DOCS_PATH: Path = BASE_DIR / os.getenv("HR_DOCS_PATH", "data/hr_policies")
    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Supported file extensions for ingestion
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]

    @classmethod
    def validate(cls):
        if not cls.HR_DOCS_PATH.exists():
            raise FileNotFoundError(
                f"[config] HR docs folder not found: {cls.HR_DOCS_PATH}. "
                f"Create the folder and add your policy documents."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalConfig:
    TOP_K: int               = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))


# ══════════════════════════════════════════════════════════════════════════════
# App / General
# ══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    TITLE: str   = os.getenv("APP_TITLE", "HR Policy Assistant")
    ENV: str     = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    BASE_DIR: Path = BASE_DIR

    @classmethod
    def is_production(cls) -> bool:
        return cls.ENV.lower() == "production"


# ══════════════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════════════

class CacheConfig:
    REDIS_URL: str = os.getenv("REDIS_URL", "")           # empty = no Redis, use in-memory
    TTL: int       = int(os.getenv("CACHE_TTL", "3600"))  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# Database (optional — chat history)
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseConfig:
    URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/hr_chatbot.db")


# ══════════════════════════════════════════════════════════════════════════════
# Validate all configs at import time (fails fast on bad setup)
# ══════════════════════════════════════════════════════════════════════════════

def validate_all():
    """
    Call this once at app startup (e.g. in app.py) to catch
    missing env variables before any user traffic hits.
    """
    LLMConfig.validate()
    VectorStoreConfig.validate()
    # IngestConfig.validate()   # uncomment after you've added HR docs


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: single import point
# ══════════════════════════════════════════════════════════════════════════════

llm_config        = LLMConfig()
vector_config     = VectorStoreConfig()
embedding_config  = EmbeddingConfig()
ingest_config     = IngestConfig()
retrieval_config  = RetrievalConfig()
app_config        = AppConfig()
cache_config      = CacheConfig()
db_config         = DatabaseConfig()
