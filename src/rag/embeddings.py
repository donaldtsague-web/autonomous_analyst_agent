"""Modèle d'embeddings local (open-source), partagé par le chunker et ChromaDB."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Charge le modèle une seule fois (téléchargement au premier appel)."""
    logger.info("Chargement du modèle d'embeddings : %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

