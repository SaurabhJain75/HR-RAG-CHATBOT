"""
ingest.py
---------
One-time (or re-runnable) script to load HR policy documents,
split them into chunks, and store them in the vector store.

Run this whenever you:
  - Add new HR policy documents
  - Update existing documents
  - Want to rebuild the vector store from scratch

Usage:
    python ingest.py                        # ingest all docs in HR_DOCS_PATH
    python ingest.py --file leave_policy.pdf  # ingest a single file
    python ingest.py --reset                # wipe vector store and re-ingest all
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from config import ingest_config, vector_config
from models import DocumentChunk, IngestResult, IngestSummary
from rag import add_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Document Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Run: pip install pypdf")

    reader = PdfReader(str(file_path))
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    return "\n\n".join(pages)


def load_docx(file_path: Path) -> str:
    """Extract text from a .docx Word document."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Run: pip install python-docx")

    doc   = Document(str(file_path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paras)


def load_txt(file_path: Path) -> str:
    """Load plain text or markdown files."""
    return file_path.read_text(encoding="utf-8").strip()


def load_document(file_path: Path) -> str:
    """
    Route to the correct loader based on file extension.

    Args:
        file_path: Path to the HR policy document

    Returns:
        Extracted text as a single string

    Raises:
        ValueError: If the file type is not supported
    """
    ext = file_path.suffix.lower()

    loaders = {
        ".pdf":  load_pdf,
        ".docx": load_docx,
        ".txt":  load_txt,
        ".md":   load_txt,
    }

    if ext not in loaders:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {list(loaders.keys())}"
        )

    logger.info(f"Loading: {file_path.name}")
    text = loaders[ext](file_path)

    if not text.strip():
        raise ValueError(f"No text extracted from {file_path.name} — file may be empty or scanned.")

    return text


# ══════════════════════════════════════════════════════════════════════════════
# Text Chunker
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    filename: str,
    chunk_size: int  = None,
    chunk_overlap: int = None
) -> list[DocumentChunk]:
    """
    Split a long document text into overlapping chunks.

    Uses a simple sliding window approach:
    - Each chunk is `chunk_size` characters long
    - Consecutive chunks overlap by `chunk_overlap` characters
      so sentences aren't cut off at boundaries

    Args:
        text:          Full document text
        filename:      Source filename (stored in chunk metadata)
        chunk_size:    Max characters per chunk (defaults to config)
        chunk_overlap: Overlap between chunks (defaults to config)

    Returns:
        List of DocumentChunk objects
    """
    chunk_size    = chunk_size    or ingest_config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or ingest_config.CHUNK_OVERLAP

    # Clean up excessive whitespace
    text = " ".join(text.split())

    chunks = []
    start  = 0
    index  = 0

    while start < len(text):
        end  = start + chunk_size
        part = text[start:end]

        # Try to end at a sentence boundary (. ! ?) to avoid mid-sentence cuts
        if end < len(text):
            last_period = max(
                part.rfind(". "),
                part.rfind("! "),
                part.rfind("? "),
                part.rfind("\n")
            )
            if last_period > chunk_size // 2:   # only use if boundary is in second half
                part = part[:last_period + 1]

        chunk = DocumentChunk(
            chunk_id    = f"{Path(filename).stem}_chunk_{index}",
            source_file = filename,
            page_number = None,          # page-level tracking is PDF-only (see load_pdf_with_pages)
            section     = None,          # future: detect headings
            content     = part.strip(),
            metadata    = {"filename": filename}
        )
        chunks.append(chunk)

        index += 1
        start += max((len(part) - chunk_overlap), chunk_overlap)   # slide forward with overlap

    logger.info(f"  → {len(chunks)} chunks created from '{filename}'")
    return chunks


def chunk_pdf_with_pages(file_path: Path) -> list[DocumentChunk]:
    """
    PDF-specific chunker that preserves page numbers in chunk metadata.
    Falls back to chunk_text() for non-PDFs.

    Each PDF page is chunked independently so page references stay accurate.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Run: pip install pypdf")

    reader = PdfReader(str(file_path))
    all_chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue

        # Chunk this page's text
        page_chunks = chunk_text(page_text, file_path.name)

        # Overwrite chunk_id and add page_number
        for chunk in page_chunks:
            chunk.chunk_id    = f"{file_path.stem}_p{page_num}_chunk_{chunk_index}"
            chunk.page_number = page_num
            chunk_index += 1

        all_chunks.extend(page_chunks)

    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# Ingest a Single File
# ══════════════════════════════════════════════════════════════════════════════

def ingest_file(file_path: Path) -> IngestResult:
    """
    Load, chunk, and store a single HR policy document.

    Args:
        file_path: Path to the document

    Returns:
        IngestResult with success/failure status and chunk count
    """
    filename = file_path.name

    try:
        # Use page-aware chunker for PDFs, generic for others
        if file_path.suffix.lower() == ".pdf":
            chunks = chunk_pdf_with_pages(file_path)
        else:
            text   = load_document(file_path)
            chunks = chunk_text(text, filename)

        if not chunks:
            return IngestResult(
                filename=filename,
                total_chunks=0,
                success=False,
                error_message="No chunks generated — document may be empty."
            )

        add_chunks(chunks)

        return IngestResult(
            filename=filename,
            total_chunks=len(chunks),
            success=True
        )

    except Exception as e:
        logger.error(f"Failed to ingest '{filename}': {e}")
        return IngestResult(
            filename=filename,
            total_chunks=0,
            success=False,
            error_message=str(e)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Ingest All Files in HR Docs Folder
# ══════════════════════════════════════════════════════════════════════════════

def ingest_all(docs_path: Path = None) -> IngestSummary:
    """
    Ingest every supported document in the HR docs folder.

    Args:
        docs_path: Override the default HR_DOCS_PATH from config

    Returns:
        IngestSummary with per-file results and totals
    """
    docs_path = docs_path or ingest_config.HR_DOCS_PATH
    summary   = IngestSummary()

    if not docs_path.exists():
        logger.error(f"HR docs folder not found: {docs_path}")
        logger.error("Create the folder and add your policy documents, then re-run.")
        return summary

    # Find all supported files recursively
    files = [
        f for f in docs_path.rglob("*")
        if f.is_file() and f.suffix.lower() in ingest_config.SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(f"No supported documents found in: {docs_path}")
        logger.warning(f"Supported extensions: {ingest_config.SUPPORTED_EXTENSIONS}")
        return summary

    logger.info(f"Found {len(files)} document(s) to ingest.")

    for file_path in files:
        result = ingest_file(file_path)
        summary.add_result(result)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Reset Vector Store
# ══════════════════════════════════════════════════════════════════════════════

def reset_vector_store() -> None:
    """
    Wipe the entire vector store directory.
    Use before re-ingesting if documents have changed significantly.
    """
    path = vector_config.PATH
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"Vector store wiped: {path}")
    else:
        logger.info("Vector store directory does not exist — nothing to reset.")


# ══════════════════════════════════════════════════════════════════════════════
# Print Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(summary: IngestSummary) -> None:
    """Print a clean ingestion report to the console."""
    print("\n" + "=" * 55)
    print("  HR POLICY INGESTION SUMMARY")
    print("=" * 55)
    print(f"  Total files processed : {summary.total_files}")
    print(f"  Successful            : {summary.successful_files}")
    print(f"  Failed                : {summary.failed_files}")
    print(f"  Total chunks stored   : {summary.total_chunks}")
    print("-" * 55)

    for result in summary.results:
        status = "✅" if result.success else "❌"
        msg    = f"{result.total_chunks} chunks" if result.success else result.error_message
        print(f"  {status}  {result.filename:<35} {msg}")

    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest HR policy documents into the vector store."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Ingest a single file by name (must be inside HR_DOCS_PATH)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the vector store before ingesting"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Optional reset
    if args.reset:
        logger.info("--reset flag detected. Wiping vector store...")
        reset_vector_store()

    # Single file mode
    if args.file:
        file_path = ingest_config.HR_DOCS_PATH / args.file
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        result  = ingest_file(file_path)
        summary = IngestSummary()
        summary.add_result(result)

    # All files mode (default)
    else:
        summary = ingest_all()

    print_summary(summary)

    if summary.failed_files > 0:
        sys.exit(1)   # non-zero exit so CI/CD pipelines catch failures
