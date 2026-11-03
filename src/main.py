"""Point d'entrée de la pipeline d'ingestion RAG (enrichi à chaque étape de la Phase 1).

Exécution : python -m src.main
"""

from src.config import DATA_DIR
from src.logger import get_logger, log_step
from src.rag.loader import load_documents

logger = get_logger("pipeline")


def main() -> None:
    # Étape 1.1 : chargement des documents
    documents = load_documents(DATA_DIR)
    log_step(logger, "1.1", f"Chargement terminé ({len(documents)} segments)")


if __name__ == "__main__":
    main()
