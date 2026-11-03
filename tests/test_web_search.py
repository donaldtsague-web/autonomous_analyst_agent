"""Tests de l'outil de recherche web : choix du fournisseur et repli (réseau simulé)."""

from src.tools import web_search

TAVILY_RESULT = [{"title": "T", "url": "https://tavily.example", "content": "contenu tavily"}]
DDG_RESULT = [{"title": "D", "url": "https://ddg.example", "content": "contenu ddg"}]


def _patch(monkeypatch, tavily_key, tavily=None, ddg=None):
    monkeypatch.setattr(web_search, "get_tavily_api_key", lambda: tavily_key)
    calls = {"tavily": 0, "ddg": 0}

    def fake_tavily(query, max_results):
        calls["tavily"] += 1
        if tavily is None:
            raise RuntimeError("quota dépassé")
        return tavily

    def fake_ddg(query, max_results):
        calls["ddg"] += 1
        if ddg is None:
            raise RuntimeError("réseau coupé")
        return ddg

    monkeypatch.setattr(web_search, "_search_tavily", fake_tavily)
    monkeypatch.setattr(web_search, "_search_duckduckgo", fake_ddg)
    return calls


def test_uses_duckduckgo_without_tavily_key(monkeypatch):
    calls = _patch(monkeypatch, tavily_key=None, ddg=DDG_RESULT)

    output = web_search.search_web.invoke({"query": "actualités"})

    assert "https://ddg.example" in output
    assert calls == {"tavily": 0, "ddg": 1}


def test_prefers_tavily_when_key_is_set(monkeypatch):
    calls = _patch(monkeypatch, tavily_key="cle", tavily=TAVILY_RESULT, ddg=DDG_RESULT)

    output = web_search.search_web.invoke({"query": "actualités"})

    assert "https://tavily.example" in output
    assert calls == {"tavily": 1, "ddg": 0}


def test_falls_back_to_duckduckgo_when_tavily_fails(monkeypatch):
    calls = _patch(monkeypatch, tavily_key="cle", tavily=None, ddg=DDG_RESULT)

    output = web_search.search_web.invoke({"query": "actualités"})

    assert "https://ddg.example" in output
    assert calls == {"tavily": 1, "ddg": 1}


def test_reports_unavailability_instead_of_raising(monkeypatch):
    _patch(monkeypatch, tavily_key=None, ddg=None)

    output = web_search.search_web.invoke({"query": "actualités"})

    assert output.startswith("Recherche web indisponible")

