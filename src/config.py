"""Configuration centrale : chemins du projet (étendue à l'étape 3.1 pour les LLM)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
