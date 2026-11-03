"""Étape 2.2 — Outil de recherche web résiliente (Tavily, repli DuckDuckGo)."""

from langchain_core.tools import tool

from src.config import get_tavily_api_key
from src.logger import get_logger, log_step

STEP = "2.2"
logger = get_logger(__name__)

MAX_RESULTS = 5


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    response = TavilyClient(api_key=get_tavily_api_key()).search(query, max_results=max_results)
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response.get("results", [])
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    hits = DDGS().text(query, max_results=max_results)
    return [{"title": h["title"], "url": h["href"], "content": h["body"]} for h in hits]


def search_with_fallback(query: str, max_results: int = MAX_RESULTS) -> tuple[str, list[dict]]:
    """Retourne (fournisseur utilisé, résultats). Tavily d'abord si une clé existe, sinon DuckDuckGo."""
    if get_tavily_api_key():
        try:
            results = _search_tavily(query, max_results)
            log_step(logger, STEP, f"Tavily : {len(results)} résultat(s)")
            return "Tavily", results
        except Exception as error:  # clé invalide, quota, réseau...
            logger.warning("[étape %s] Tavily indisponible (%s) : repli sur DuckDuckGo", STEP, error)

    results = _search_duckduckgo(query, max_results)
    log_step(logger, STEP, f"DuckDuckGo : {len(results)} résultat(s)")
    return "DuckDuckGo", results


def format_results(provider: str, results: list[dict]) -> str:
    """Formate les résultats avec leur URL pour permettre la citation des sources."""
    if not results:
        return f"Aucun résultat web trouvé ({provider})."
    blocks = [f"[web: {r['title']} | {r['url']}]\n{r['content']}" for r in results]
    return "\n\n".join(blocks)


@tool
def search_web(query: str) -> str:
    """Recherche sur le web pour obtenir des informations récentes ou absentes des documents internes
    (actualités, annonces, mises à jour). Retourne des extraits avec le titre et l'URL de chaque source."""
    try:
        provider, results = search_with_fallback(query)
    except Exception as error:
        logger.error("[étape %s] Recherche web en échec : %s", STEP, error)
        return f"Recherche web indisponible : {error}"
    return format_results(provider, results)

