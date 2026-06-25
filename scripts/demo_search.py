"""Interface CLI interactive pour le moteur de recherche semantique."""

import logging
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import DatabaseManager
from src.embeddings import EmbeddingGenerator
from src.search import SemanticSearchEngine

logger = logging.getLogger(__name__)

TITLE = """
\033[96m╔═══════════════════════════════════════════════════╗
║   MOTEUR DE RECHERCHE SEMANTIQUE (pgvector)       ║
║   Semantic vs TF-IDF — ESI Alger 2CS              ║
╚═══════════════════════════════════════════════════╝\033[0m
"""


def format_results(results: list[dict], method: str) -> str:
    rows = []
    for i, r in enumerate(results, 1):
        title = r["title"][:50] + "..." if len(r["title"]) > 50 else r["title"]
        rows.append([i, title, r["category"], f"{r['similarity_score']:.4f}"])
    return tabulate(
        rows,
        headers=["Rang", "Titre", "Categorie", "Score"],
        tablefmt="rounded_grid",
    )


def main() -> None:
    print(TITLE)
    db = DatabaseManager()
    gen = EmbeddingGenerator()
    engine = SemanticSearchEngine(db, gen)

    try:
        while True:
            query = input("\n\033[93mVotre requete (ou 'quit') : \033[0m").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            result = engine.compare_methods(query, top_k=5)

            print(f"\n\033[92m=== Recherche Semantique ({result['semantic_time_ms']:.1f} ms) ===\033[0m")
            print(format_results(result["semantic"], "semantic"))

            print(f"\n\033[93m=== Recherche TF-IDF ({result['tfidf_time_ms']:.1f} ms) ===\033[0m")
            print(format_results(result["tfidf"], "tfidf"))

            print(f"\n\033[96mOverlap: {result['overlap']}/5 documents en commun\033[0m")

    except (KeyboardInterrupt, EOFError):
        print("\n")
    finally:
        db.close()
        print("\033[96mAu revoir !\033[0m")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
