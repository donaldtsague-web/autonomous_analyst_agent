"""Étape 1.5 — Recherche hybride : fusion dense (ChromaDB) + lexicale (BM25)."""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from src.logger import get_logger, log_step
from src.rag.bm25 import load_bm25
from src.rag.vectorstore import load_vectorstore

STEP = "1.5"
logger = get_logger(__name__)


def get_hybrid_retriever(
    k: int = 4, dense_weight: float = 0.5, sparse_weight: float = 0.5
) -> BaseRetriever:
    """Combine les deux retrievers via Reciprocal Rank Fusion pondérée."""
    dense = load_vectorstore().as_retriever(search_kwargs={"k": k})
    sparse = load_bm25()
    sparse.k = k

    log_step(
        logger, STEP, f"Retriever hybride prêt (k={k}, poids dense/BM25 = {dense_weight}/{sparse_weight})"
    )
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[dense_weight, sparse_weight])

