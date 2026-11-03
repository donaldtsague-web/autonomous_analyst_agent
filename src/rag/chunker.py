"""Étape 1.2 — Découpage sémantique (Semantic Chunking)."""

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.logger import get_logger, log_step
from src.rag.embeddings import get_embeddings

STEP = "1.2"
logger = get_logger(__name__)


def chunk_documents(
    documents: list[Document],
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
) -> list[Document]:
    """Découpe les documents selon les ruptures de sens entre phrases.

    Les métadonnées (source, page, file_type) sont conservées ; un `chunk_id`
    unique est ajouté pour permettre la citation précise des sources.
    """
    kwargs = {"breakpoint_threshold_type": breakpoint_threshold_type}
    if breakpoint_threshold_amount is not None:
        kwargs["breakpoint_threshold_amount"] = breakpoint_threshold_amount
    splitter = SemanticChunker(get_embeddings(), **kwargs)

    log_step(logger, STEP, f"Découpage sémantique de {len(documents)} segment(s)...")
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('source', 'inconnu')}"
            f"#p{chunk.metadata.get('page', 0)}-c{index}"
        )

    log_step(logger, STEP, f"{len(chunks)} fragment(s) sémantique(s) créé(s)")
    return chunks

