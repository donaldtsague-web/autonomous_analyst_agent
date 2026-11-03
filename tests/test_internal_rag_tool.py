"""Tests de l'outil RAG interne (nécessite l'index construit par `python -m src.main`)."""

import pytest

from src.rag.bm25 import BM25_PATH
from src.tools.internal_rag import TOOL_NAME, build_internal_rag_tool


@pytest.fixture(scope="module")
def tool():
    if not BM25_PATH.is_file():
        pytest.skip("Index absent : lancer `python -m src.main` d'abord")
    return build_internal_rag_tool()


def test_tool_metadata(tool):
    assert tool.name == TOOL_NAME
    assert "documents internes" in tool.description


def test_tool_returns_cited_excerpts(tool):
    output = tool.invoke({"query": "Quelles sont les compétences clés du plan ?"})

    assert "[source: plan_de_monter_en_competence.pdf" in output
    assert ", p." in output
