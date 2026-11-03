"""Configuration centrale : chemins du projet et variables d'environnement (.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"

# Charge .env (ignoré par git) sans écraser les variables déjà définies
load_dotenv(ROOT_DIR / ".env")


def get_tavily_api_key() -> str | None:
    """Clé Tavily optionnelle : sans elle, la recherche web bascule sur DuckDuckGo."""
    return os.getenv("TAVILY_API_KEY") or None
