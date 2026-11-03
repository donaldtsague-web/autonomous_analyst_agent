"""Boîte à outils de l'agent analyste (Phase 2) : RAG interne, Web Search, Sandbox Python."""

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from src.tools.internal_rag import build_internal_rag_tool
from src.tools.python_sandbox import run_python_sandbox
from src.tools.web_search import search_web


def get_all_tools(retriever: BaseRetriever | None = None) -> list[BaseTool]:
    """Retourne les 3 outils de la Phase 2 prêts à être liés au LLM."""
    return [
        build_internal_rag_tool(retriever=retriever),
        search_web,
        run_python_sandbox,
    ]
