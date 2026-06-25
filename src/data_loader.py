"""Chargement et exploration du dataset ag_news depuis Hugging Face."""

import json
import logging
import time
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

CATEGORY_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def download_ag_news(n_samples: int = 2000) -> list[dict]:
    """Telecharge le dataset ag_news et retourne les n premiers documents.

    Parameters
    ----------
    n_samples : int
        Nombre de documents a extraire.

    Returns
    -------
    list[dict]
        Documents avec id, title, text, category.
    """
    logger.info("Downloading ag_news from Hugging Face...")
    dataset = load_dataset("ag_news", split=f"train[:{n_samples}]")
    documents = []
    for i, item in enumerate(dataset):
        documents.append({
            "id": i,
            "title": item["text"].split(" - ", 1)[0] if " - " in item["text"] else item["text"][:80],
            "text": item["text"],
            "category": CATEGORY_MAP[item["label"]],
        })
    return documents


def save_documents(docs: list[dict], path: str = "data/raw/documents.jsonl") -> None:
    """Sauvegarde les documents en JSON Lines.

    Parameters
    ----------
    docs : list[dict]
        Documents a sauvegarder.
    path : str
        Chemin du fichier de sortie.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def print_statistics(docs: list[dict]) -> None:
    """Affiche les statistiques du dataset.

    Parameters
    ----------
    docs : list[dict]
        Documents a analyser.
    """
    word_lengths = [len(d["text"].split()) for d in docs]
    word_lengths.sort()
    n = len(word_lengths)
    median = word_lengths[n // 2] if n % 2 == 1 else (word_lengths[n // 2 - 1] + word_lengths[n // 2]) / 2

    logger.info("=== Dataset Statistics ===")
    logger.info("Total documents: %d", len(docs))
    logger.info("Avg words: %.1f", sum(word_lengths) / n)
    logger.info("Median words: %.1f", median)
    logger.info("Max words: %d", max(word_lengths))

    categories = {}
    for d in docs:
        cat = d["category"]
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("=== Category Distribution ===")
    for cat, count in sorted(categories.items()):
        logger.info("  %s: %d (%.1f%%)", cat, count, 100 * count / len(docs))

    logger.info("=== Sample Documents ===")
    shown = {}
    for d in docs:
        cat = d["category"]
        if cat not in shown:
            shown[cat] = 0
        if shown[cat] < 3:
            logger.info("  [%s] %s", cat, d["text"][:100])
            shown[cat] += 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()
    docs = download_ag_news(2000)
    logger.info("Download complete (%.1fs)", time.time() - t0)

    save_documents(docs, "data/raw/documents.jsonl")
    logger.info("Saved to data/raw/documents.jsonl")

    print_statistics(docs)

    sample = docs[:10]
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/sample_10.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    logger.info("Saved sample to data/processed/sample_10.json")

    with open("data/raw/documents.jsonl") as f:
        line_count = sum(1 for _ in f)
    assert line_count == 2000, f"Expected 2000 lines, got {line_count}"
    logger.info("Verification: %d lines OK", line_count)
