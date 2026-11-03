"""Point d'entrée de la pipeline d'ingestion RAG (enrichi à chaque étape de la Phase 1).

Exécution : python -m src.main
"""

from src.config import DATA_DIR
from src.logger import get_logger, log_step
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.vectorstore import build_vectorstore, load_vectorstore

logger = get_logger("pipeline")

TEST_QUERY = "Quelles sont les compétences techniques clés de ce plan ?"


def main() -> None:
    # Étape 1.1 : chargement des documents
    documents = load_documents(DATA_DIR)
    log_step(logger, "1.1", f"Chargement terminé ({len(documents)} segments)")

    # Étape 1.2 : découpage sémantique
    chunks = chunk_documents(documents)
    log_step(logger, "1.2", f"Découpage terminé ({len(chunks)} fragments)")

    # Étape 1.3 : indexation vectorielle persistante, puis rechargement depuis le disque
    build_vectorstore(chunks)
    store = load_vectorstore()
    results = store.similarity_search(TEST_QUERY, k=3)
    log_step(logger, "1.3", f"Recherche dense '{TEST_QUERY}' : {len(results)} résultat(s)")
    for doc in results:
        preview = doc.page_content[:80].replace("\n", " ")
        log_step(logger, "1.3", f"  {doc.metadata['chunk_id']} : {preview}...")


if __name__ == "__main__":
    main()
