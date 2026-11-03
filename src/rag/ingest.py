"""Pipeline d'ingestion complète (étapes 1.1 à 1.4) : documents -> fragments -> index dense + BM25."""

from pathlib import Path

from langchain_core.documents import Document

from src.config import DATA_DIR
from src.logger import get_logger, log_step
from src.rag.bm25 import build_bm25
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.vectorstore import build_vectorstore

logger = get_logger("ingest")


def run_ingestion(data_dir: Path = DATA_DIR) -> list[Document]:
    """Reconstruit les deux index persistants à partir du dossier de documents."""
    documents = load_documents(data_dir)  # étape 1.1
    chunks = chunk_documents(documents)  # étape 1.2
    build_vectorstore(chunks)  # étape 1.3
    build_bm25(chunks)  # étape 1.4
    log_step(logger, "1.4", f"Ingestion terminée : {len(chunks)} fragment(s) indexé(s)")
    return chunks

