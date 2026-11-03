"""Étape 1.3 — Index vectoriel dense (ChromaDB, persistant sur disque)."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import STORAGE_DIR
from src.logger import get_logger, log_step
from src.rag.embeddings import get_embeddings

STEP = "1.3"
logger = get_logger(__name__)

COLLECTION_NAME = "internal_docs"
CHROMA_DIR = STORAGE_DIR / "chroma_db"


def build_vectorstore(chunks: list[Document], persist_dir: Path = CHROMA_DIR) -> Chroma:
    """(Re)construit la collection : l'ancienne est supprimée pour éviter les doublons."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )
    store.delete_collection()

    log_step(logger, STEP, f"Indexation de {len(chunks)} fragment(s) dans {persist_dir}")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
    )
    log_step(logger, STEP, "Index vectoriel persisté")
    return store


def load_vectorstore(persist_dir: Path = CHROMA_DIR) -> Chroma:
    """Recharge l'index existant depuis le disque."""
    if not persist_dir.is_dir():
        raise FileNotFoundError(f"Index vectoriel absent : {persist_dir} (lancer l'ingestion)")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )

