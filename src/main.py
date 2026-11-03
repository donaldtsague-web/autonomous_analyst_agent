"""Point d'entrée de la pipeline d'ingestion RAG (enrichi à chaque étape de la Phase 1).

Exécution : python -m src.main
"""

from src.config import DATA_DIR
from src.logger import get_logger, log_step
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents

logger = get_logger("pipeline")


def main() -> None:
    # Étape 1.1 : chargement des documents
    documents = load_documents(DATA_DIR)
    log_step(logger, "1.1", f"Chargement terminé ({len(documents)} segments)")

    # Étape 1.2 : découpage sémantique
    chunks = chunk_documents(documents)
    log_step(logger, "1.2", f"Découpage terminé ({len(chunks)} fragments)")
    for chunk in chunks[:3]:
        preview = chunk.page_content[:80].replace("\n", " ")
        log_step(logger, "1.2", f"  {chunk.metadata['chunk_id']} : {preview}...")


if __name__ == "__main__":
    main()
