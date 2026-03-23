"""
ingest.py
─────────
Ingestion pipeline for the Chemical Engineering RAG system.

Workflow:
    1. Recursively scan DATA_DIR for PDFs, grouped by topic sub-folder.
    2. Load each PDF with PyPDFLoader and tag pages with topic + source metadata.
    3. Split pages into overlapping chunks with RecursiveCharacterTextSplitter.
    4. Embed chunks with a HuggingFace sentence-transformer model.
    5. Persist the resulting FAISS index to disk for retrieval.

Usage:
    python ingest.py
    python ingest.py --data-dir /path/to/docs --vector-db ./vectorstore
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Generator

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults (override via CLI or environment variables) ─────────────────────
DEFAULT_DATA_DIR: str = os.getenv(
    "CHEM_DATA_DIR",
    r"D:\AdvancedML\chemical_rag_baseline\Documents",
)
DEFAULT_VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "vectorstore")
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100


# ── Document loading ─────────────────────────────────────────────────────────
def iter_pdf_paths(data_dir: Path) -> Generator[tuple[str, Path], None, None]:
    """
    Yield (topic_name, pdf_path) for every PDF found one level deep inside
    *data_dir*.  Non-directory entries at the top level are skipped with a
    warning.
    """
    if not data_dir.exists():
        logger.error("DATA_DIR does not exist: %s", data_dir)
        sys.exit(1)

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            logger.debug("Skipping non-directory entry: %s", entry.name)
            continue

        pdf_files = sorted(entry.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDFs found in topic folder '%s' — skipping.", entry.name)
            continue

        for pdf_path in pdf_files:
            yield entry.name, pdf_path


def load_documents(data_dir: Path) -> list[Document]:
    """
    Load all PDFs from *data_dir* and attach `topic` and `source` metadata
    to every page document.

    Returns:
        A flat list of LangChain Document objects (one per PDF page).
    """
    documents: list[Document] = []
    pdf_count = 0

    for topic, pdf_path in iter_pdf_paths(data_dir):
        logger.info("  Loading %-45s [topic: %s]", pdf_path.name, topic)
        try:
            pages: list[Document] = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:  # noqa: BLE001
            logger.error("  Failed to load '%s': %s", pdf_path.name, exc)
            continue

        for page in pages:
            page.metadata["topic"] = topic
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)
        pdf_count += 1

    logger.info("Loaded %d pages from %d PDF(s).", len(documents), pdf_count)
    return documents


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents:     Raw page-level documents from the loader.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A flat list of chunk-level Document objects.
    """
    if not documents:
        logger.error("No documents to chunk. Aborting.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,   # adds char offset metadata for traceability
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d pages → %d chunks  (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ── Vector store ─────────────────────────────────────────────────────────────
def build_and_save_vectorstore(
    chunks: list[Document],
    vector_db_dir: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """
    Embed *chunks* and persist a FAISS index to *vector_db_dir*.

    Args:
        chunks:          Chunked LangChain documents.
        vector_db_dir:   Directory where the FAISS index will be saved.
        embedding_model: HuggingFace model name for sentence embeddings.
    """
    if not chunks:
        logger.error("No chunks to embed. Aborting.")
        sys.exit(1)

    logger.info("Loading embedding model '%s'…", embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    logger.info("Embedding %d chunks — this may take a while…", len(chunks))
    t0 = time.perf_counter()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - t0
    logger.info("Embedding complete in %.1f s.", elapsed)

    vector_db_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vector_db_dir))
    logger.info("FAISS index saved to '%s'.", vector_db_dir.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into a FAISS vector store for the Chem-RAG system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Root directory containing topic sub-folders with PDF files.",
    )
    parser.add_argument(
        "--vector-db",
        type=Path,
        default=Path(DEFAULT_VECTOR_DB_DIR),
        help="Directory where the FAISS index will be persisted.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="HuggingFace sentence-transformer model name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlapping characters between consecutive chunks.",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    logger.info("═" * 55)
    logger.info("  Chem-RAG Ingestion Pipeline")
    logger.info("  Data dir  : %s", args.data_dir)
    logger.info("  Vector DB : %s", args.vector_db)
    logger.info("  Embedding : %s", args.embedding_model)
    logger.info("═" * 55)

    logger.info("Step 1/3 — Loading documents…")
    documents = load_documents(args.data_dir)

    logger.info("Step 2/3 — Chunking documents…")
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info("Step 3/3 — Building & saving vector store…")
    build_and_save_vectorstore(chunks, args.vector_db, args.embedding_model)

    logger.info("✅  Ingestion complete.")


if __name__ == "__main__":
    main()