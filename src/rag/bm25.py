"""Étape 1.4 — Index lexical sparse (BM25, persistant sur disque)."""

import pickle
import re
import unicodedata
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import STORAGE_DIR
from src.logger import get_logger, log_step

STEP = "1.4"
logger = get_logger(__name__)

BM25_PATH = STORAGE_DIR / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    """Minuscules, accents retirés, découpage en mots (adapté au français).

    Défini au niveau du module pour rester sérialisable par pickle.
    """
    normalized = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", without_accents)


def build_bm25(chunks: list[Document], path: Path = BM25_PATH, k: int = 4) -> BM25Retriever:
    """Construit l'index BM25 sur les fragments et le sérialise sur disque."""
    log_step(logger, STEP, f"Construction de l'index BM25 sur {len(chunks)} fragment(s)")
    retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize, k=k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(retriever, file)
    log_step(logger, STEP, f"Index BM25 sérialisé : {path}")
    return retriever


def load_bm25(path: Path = BM25_PATH) -> BM25Retriever:
    """Recharge l'index BM25 depuis le disque.

    Le fichier est produit localement par `build_bm25` : ne jamais charger
    un fichier .pkl d'origine inconnue (pickle peut exécuter du code).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Index BM25 absent : {path} (lancer l'ingestion)")
    with path.open("rb") as file:
        return pickle.load(file)

