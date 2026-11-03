"""Validation de la Phase 1 : recherche hybride sur les documents locaux (TEST-1)."""

import pytest

from src.config import DATA_DIR
from src.rag.hybrid_retriever import get_hybrid_retriever
from src.rag.ingest import run_ingestion

PDF_NAME = "plan_de_monter_en_competence.pdf"


@pytest.fixture(scope="module")
def retriever():
    if not (DATA_DIR / PDF_NAME).is_file():
        pytest.skip(f"Document de référence absent : data/{PDF_NAME} (non versionné)")
    run_ingestion()
    return get_hybrid_retriever(k=4)


def test_internal_query_returns_cited_local_sources(retriever):
    """TEST-1 : une question interne ne ramène que des fragments du PDF, avec métadonnées de citation."""
    results = retriever.invoke("Quelles sont les compétences clés du plan de montée en compétences ?")

    assert results, "la recherche hybride doit retourner au moins un fragment"
    for doc in results:
        assert doc.metadata["source"] == PDF_NAME
        assert doc.metadata["page"] >= 1
        assert doc.metadata["chunk_id"].startswith(PDF_NAME)


def test_lexical_keyword_is_found_through_bm25(retriever):
    """Un terme exact (BM25) doit remonter le fragment qui le contient."""
    results = retriever.invoke("Recherche Hybride BM25")

    assert any("BM25" in doc.page_content for doc in results)

