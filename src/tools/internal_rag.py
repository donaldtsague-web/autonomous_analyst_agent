"""Étape 2.1 — Outil de recherche interne (RAG hybride exposé à l'agent)."""

from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, create_retriever_tool

from src.logger import get_logger, log_step
from src.rag.hybrid_retriever import get_hybrid_retriever

STEP = "2.1"
logger = get_logger(__name__)

TOOL_NAME = "search_internal_documents"

# Chaque extrait est préfixé par sa source pour permettre la citation précise
_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[source: {source}, p.{page}, id: {chunk_id}]\n{page_content}"
)


def build_internal_rag_tool(retriever: BaseRetriever | None = None) -> BaseTool:
    """Encapsule le retriever hybride dans un outil appelable par le LLM."""
    retriever = retriever or get_hybrid_retriever()
    log_step(logger, STEP, f"Outil '{TOOL_NAME}' créé")
    return create_retriever_tool(
        retriever,
        name=TOOL_NAME,
        description=(
            "Recherche dans les documents internes ingérés (notes, plans, rapports PDF/Markdown). "
            "À utiliser en priorité pour toute question portant sur le contenu des documents locaux. "
            "Retourne des extraits avec leur source, leur page et leur identifiant de fragment."
        ),
        document_prompt=_DOCUMENT_PROMPT,
    )
