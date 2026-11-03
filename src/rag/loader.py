"""Étape 1.1 — Chargement des documents locaux (PDF et Markdown)."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.logger import get_logger, log_step

STEP = "1.1"
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def _load_file(path: Path) -> list[Document]:
    """Charge un fichier et normalise ses métadonnées (source, file_type, page)."""
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            # PyPDFLoader indexe les pages à partir de 0 : on expose un numéro 1-based
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = path.suffix.lower().lstrip(".")

    # Écarte les pages vides (ex. pages scannées sans texte extractible)
    return [doc for doc in docs if doc.page_content.strip()]


def load_documents(data_dir: Path) -> list[Document]:
    """Charge tous les PDF/Markdown du dossier `data_dir` (non récursif)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dossier de documents introuvable : {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            if path.is_file():
                logger.warning("[étape %s] Fichier ignoré (format non supporté) : %s", STEP, path.name)
            continue
        docs = _load_file(path)
        log_step(logger, STEP, f"{path.name} : {len(docs)} segment(s) chargé(s)")
        documents.extend(docs)

    log_step(logger, STEP, f"Total : {len(documents)} segment(s) depuis {data_dir}")
    return documents
