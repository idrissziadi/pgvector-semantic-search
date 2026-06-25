"""Pipeline d'ingestion : charge les donnees et les insere dans PostgreSQL."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager
from src.preprocessing import TextPreprocessor
from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    steps = [
        "Connexion a la DB",
        "Execution du schema SQL",
        "Chargement des documents",
        "Chargement des embeddings",
        "Insertion en DB",
        "Creation des index",
        "Verification finale",
    ]
    total_start = time.time()

    # Etape 1
    t = time.time()
    logger.info("Etape 1/7 - %s ...", steps[0])
    db = DatabaseManager()
    logger.info("Etape 1/7 - %s ... OK (%.1fs)", steps[0], time.time() - t)

    # Etape 2
    t = time.time()
    logger.info("Etape 2/7 - %s ...", steps[1])
    db.execute_sql_file("sql/01_schema.sql")
    logger.info("Etape 2/7 - %s ... OK (%.1fs)", steps[1], time.time() - t)

    # Etape 3
    t = time.time()
    logger.info("Etape 3/7 - %s ...", steps[2])
    preprocessor = TextPreprocessor()
    docs = preprocessor.load_processed("data/processed/documents_clean.jsonl")
    logger.info("Etape 3/7 - %s ... OK (%d docs, %.1fs)", steps[2], len(docs), time.time() - t)

    # Etape 4
    t = time.time()
    logger.info("Etape 4/7 - %s ...", steps[3])
    gen = EmbeddingGenerator()
    embeddings = gen.load_embeddings("data/processed/embeddings.npy")
    logger.info(
        "Etape 4/7 - %s ... OK (shape=%s, %.1fs)",
        steps[3], embeddings.shape, time.time() - t
    )

    # Etape 5
    t = time.time()
    logger.info("Etape 5/7 - %s ...", steps[4])
    count = db.insert_documents(docs, embeddings)
    logger.info("Etape 5/7 - %s ... OK (%d inseres, %.1fs)", steps[4], count, time.time() - t)

    # Etape 6
    t = time.time()
    logger.info("Etape 6/7 - %s ...", steps[5])
    db.execute_sql_file("sql/02_indexes.sql")
    logger.info("Etape 6/7 - %s ... OK (%.1fs)", steps[5], time.time() - t)

    # Etape 7
    t = time.time()
    logger.info("Etape 7/7 - %s ...", steps[6])
    stats = db.get_stats()
    logger.info("Etape 7/7 - %s ... OK (%.1fs)", steps[6], time.time() - t)

    total_elapsed = time.time() - total_start
    logger.info("=" * 50)
    logger.info("PIPELINE TERMINE en %.1fs", total_elapsed)
    logger.info("Documents en base : %d", stats["total_documents"])
    logger.info("Char moyen : %.0f", stats["avg_char_count"])
    logger.info("Mots moyen : %.0f", stats["avg_word_count"])
    logger.info("Index : %s", ", ".join(stats["indexes"]))
    logger.info("=" * 50)

    db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_pipeline()
