"""Point d'entrée : ingestion RAG puis démonstration de la recherche hybride.

Exécution : python -m src.main
"""

from src.logger import get_logger, log_step
from src.rag.hybrid_retriever import get_hybrid_retriever
from src.rag.ingest import run_ingestion

logger = get_logger("pipeline")

TEST_QUERY = "Quelles sont les compétences clés du plan de montée en compétences ?"


def main() -> None:
    # Étapes 1.1 à 1.4 : ingestion et indexation (dense + BM25)
    run_ingestion()

    # Étape 1.5 : recherche hybride (TEST-1 : données internes)
    retriever = get_hybrid_retriever()
    results = retriever.invoke(TEST_QUERY)
    log_step(logger, "1.5", f"Recherche hybride '{TEST_QUERY}' : {len(results)} résultat(s)")
    for doc in results:
        preview = doc.page_content[:80].replace("\n", " ")
        log_step(logger, "1.5", f"  {doc.metadata['chunk_id']} : {preview}...")


if __name__ == "__main__":
    main()
