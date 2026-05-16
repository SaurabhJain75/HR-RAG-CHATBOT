"""
rag.py
------
RAG (Retrieval-Augmented Generation) pipeline for the HR Policy Chatbot.
Responsible for:
  1. Initializing the embedding model
  2. Connecting to / creating the vector store (Chroma or FAISS)
  3. Adding document chunks to the vector store
  4. Retrieving the most relevant chunks for a user query

Used by:
  - ingest.py  → calls add_chunks() to populate the vector store
  - tools.py   → calls retrieve() to search at query time
"""

import logging
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import embedding_config, vector_config, retrieval_config
from models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Model (singleton — loaded once, reused everywhere)
# Uses sentence-transformers with CPU-only torch (no NVIDIA/CUDA)
# Model: all-MiniLM-L6-v2 — ~90MB, fast, good quality for HR Q&A
# ══════════════════════════════════════════════════════════════════════════════

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model once and cache it.
    Downloads ~90MB on first run, then loads from cache.
    Subsequent calls return the cached instance instantly.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {embedding_config.MODEL}")
        _embedding_model = SentenceTransformer(
            embedding_config.MODEL,
            device="cpu"   # force CPU — no NVIDIA/CUDA needed
        )
        logger.info("Embedding model loaded.")
    return _embedding_model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed

    Returns:
        List of embedding vectors (each a list of floats)
    """
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Vector Store — Chroma
# ══════════════════════════════════════════════════════════════════════════════

def _get_chroma_collection():
    """
    Initialize and return a persistent ChromaDB collection.
    Creates the collection if it doesn't exist yet.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError(
            "chromadb is not installed. Run: pip install chromadb"
        )

    persist_path = str(vector_config.PATH)
    Path(persist_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name="hr_policies",
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )
    logger.info(f"ChromaDB collection ready at: {persist_path}")
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# Vector Store — FAISS
# ══════════════════════════════════════════════════════════════════════════════

def _get_faiss_index():
    """
    Load a FAISS index from disk, or return None if not yet created.
    FAISS indices are created fresh during ingestion and saved to disk.
    """
    try:
        import faiss
        import pickle
    except ImportError:
        raise ImportError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        )

    index_path = vector_config.PATH / "faiss.index"
    meta_path  = vector_config.PATH / "faiss_meta.pkl"

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"FAISS index loaded from: {index_path}")
        return index, metadata

    logger.warning("FAISS index not found. Run ingest.py first.")
    return None, []


# ══════════════════════════════════════════════════════════════════════════════
# Add Chunks to Vector Store (called by ingest.py)
# ══════════════════════════════════════════════════════════════════════════════

def add_chunks(chunks: list[DocumentChunk]) -> None:
    """
    Embed a list of DocumentChunks and add them to the vector store.
    Skips chunks that already exist (by chunk_id) to avoid duplicates.

    Args:
        chunks: List of DocumentChunk objects from ingest.py
    """
    if not chunks:
        logger.warning("add_chunks() called with empty list — nothing to add.")
        return

    texts      = [chunk.content for chunk in chunks]
    ids        = [chunk.chunk_id for chunk in chunks]
    metadatas  = [
        {
            "source_file": chunk.source_file,
            "page_number": str(chunk.page_number or ""),
            "section":     chunk.section or "",
            **{k: str(v) for k, v in chunk.metadata.items()}
        }
        for chunk in chunks
    ]

    logger.info(f"Embedding {len(chunks)} chunks...")
    vectors = embed(texts)

    store_type = vector_config.TYPE

    if store_type == "chroma":
        collection = _get_chroma_collection()
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB.")

    elif store_type == "faiss":
        _add_chunks_faiss(chunks, vectors, texts, metadatas)

    else:
        raise ValueError(f"Unsupported VECTOR_STORE_TYPE: '{store_type}'. Use 'chroma' or 'faiss'.")


def _add_chunks_faiss(chunks, vectors, texts, metadatas) -> None:
    """Internal: add chunks to a FAISS index and persist to disk."""
    try:
        import faiss
        import numpy as np
        import pickle
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu numpy")

    vector_config.PATH.mkdir(parents=True, exist_ok=True)
    index_path = vector_config.PATH / "faiss.index"
    meta_path  = vector_config.PATH / "faiss_meta.pkl"

    dim = len(vectors[0])

    # Load existing or create new
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            existing_meta = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(dim)   # Inner product = cosine on normalized vecs
        existing_meta = []

    # Normalize vectors for cosine similarity
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    index.add(arr)

    # Append metadata
    new_meta = [
        {"chunk_id": c.chunk_id, "text": t, **m}
        for c, t, m in zip(chunks, texts, metadatas)
    ]
    existing_meta.extend(new_meta)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(existing_meta, f)

    logger.info(f"Added {len(chunks)} chunks to FAISS index.")


# ══════════════════════════════════════════════════════════════════════════════
# Retrieve — Similarity Search (called by tools.py)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    top_k: Optional[int] = None,
    filters: Optional[dict] = None
) -> list[RetrievedChunk]:
    """
    Embed the user query and retrieve the most similar chunks
    from the vector store, filtered by similarity threshold.

    Args:
        query:   The user's question
        top_k:   Number of results to return (defaults to config TOP_K)
        filters: Optional metadata filters e.g. {"source_file": "leave_policy.pdf"}

    Returns:
        List of RetrievedChunk objects sorted by similarity (highest first).
        Returns empty list if nothing meets the similarity threshold.
    """
    top_k     = top_k or retrieval_config.TOP_K
    threshold = retrieval_config.SIMILARITY_THRESHOLD
    store_type = vector_config.TYPE

    logger.info(f"Retrieving top {top_k} chunks for query: '{query[:80]}...'")

    query_vector = embed([query])[0]

    if store_type == "chroma":
        return _retrieve_chroma(query_vector, query, top_k, threshold, filters)
    elif store_type == "faiss":
        return _retrieve_faiss(query_vector, top_k, threshold)
    else:
        raise ValueError(f"Unsupported VECTOR_STORE_TYPE: '{store_type}'.")


def _retrieve_chroma(
    query_vector: list[float],
    query: str,
    top_k: int,
    threshold: float,
    filters: Optional[dict]
) -> list[RetrievedChunk]:
    """Internal: run similarity search on ChromaDB."""
    collection = _get_chroma_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run `python ingest.py` to load HR policy documents first."
        )

    kwargs = dict(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    if filters:
        kwargs["where"] = {k: {"$eq": v} for k, v in filters.items()}

    results = collection.query(**kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Chroma cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score 0–1
        score = 1 - (dist / 2)

        if score < threshold:
            continue

        chunks.append(RetrievedChunk(
            chunk_id     = meta.get("chunk_id", ""),
            source_file  = meta.get("source_file", "unknown"),
            page_number  = int(meta["page_number"]) if meta.get("page_number") else None,
            section      = meta.get("section") or None,
            content      = doc,
            metadata     = meta,
            similarity_score = round(score, 4)
        ))

    chunks.sort(key=lambda c: c.similarity_score, reverse=True)
    logger.info(f"Retrieved {len(chunks)} chunks above threshold {threshold}.")
    return chunks


def _retrieve_faiss(
    query_vector: list[float],
    top_k: int,
    threshold: float
) -> list[RetrievedChunk]:
    """Internal: run similarity search on FAISS index."""
    try:
        import faiss
        import numpy as np
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu numpy")

    index, metadata = _get_faiss_index()

    if index is None:
        raise RuntimeError(
            "FAISS index not found. Run `python ingest.py` to load HR policy documents first."
        )

    arr = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(arr)

    scores, indices = index.search(arr, top_k)

    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        meta = metadata[idx]
        chunks.append(RetrievedChunk(
            chunk_id     = meta.get("chunk_id", ""),
            source_file  = meta.get("source_file", "unknown"),
            page_number  = int(meta["page_number"]) if meta.get("page_number") else None,
            section      = meta.get("section") or None,
            content      = meta.get("text", ""),
            metadata     = meta,
            similarity_score = round(float(score), 4)
        ))

    chunks.sort(key=lambda c: c.similarity_score, reverse=True)
    logger.info(f"Retrieved {len(chunks)} chunks above threshold {threshold}.")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def get_collection_stats() -> dict:
    """
    Return basic stats about the vector store.
    Useful for a health-check endpoint or admin page.
    """
    store_type = vector_config.TYPE

    if store_type == "chroma":
        collection = _get_chroma_collection()
        return {
            "store_type": "chroma",
            "total_chunks": collection.count(),
            "persist_path": str(vector_config.PATH)
        }

    elif store_type == "faiss":
        index, metadata = _get_faiss_index()
        return {
            "store_type": "faiss",
            "total_chunks": index.ntotal if index else 0,
            "persist_path": str(vector_config.PATH)
        }

    return {"store_type": store_type, "total_chunks": "unknown"}
