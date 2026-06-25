"""Preprocessing de textes pour le moteur de recherche semantique."""

import json
import logging
import re
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Pipeline de nettoyage et preparation des documents textuels.

    Parameters
    ----------
    max_chars : int, optional
        Longueur maximale des textes apres troncature.
    """

    def __init__(self, max_chars: int = Config.MAX_CHARS) -> None:
        self.max_chars = max_chars

    def clean_text(self, text: str) -> str:
        """Nettoie un texte brut.

        Parameters
        ----------
        text : str
            Texte a nettoyer.

        Returns
        -------
        str
            Texte nettoye et tronque.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected str, got {type(text)}")
        if not text:
            return ""
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:self.max_chars]

    def process_documents(self, docs: list[dict]) -> list[dict]:
        """Traite une liste de documents.

        Parameters
        ----------
        docs : list[dict]
            Documents avec champs 'title' et 'text'.

        Returns
        -------
        list[dict]
            Documents enrichis avec full_text, char_count, word_count.
        """
        processed = []
        filtered_count = 0
        for doc in docs:
            d = dict(doc)
            d["title"] = self.clean_text(d.get("title", ""))
            d["text"] = self.clean_text(d.get("text", ""))
            d["full_text"] = d["title"] + ". " + d["text"]
            d["char_count"] = len(d["full_text"])
            d["word_count"] = len(d["full_text"].split())
            if len(d["full_text"]) < 20:
                filtered_count += 1
                continue
            processed.append(d)
        if filtered_count:
            logger.info("Filtered %d documents (full_text < 20 chars)", filtered_count)
        return processed

    def save_processed(
        self, docs: list[dict], path: str = "data/processed/documents_clean.jsonl"
    ) -> None:
        """Sauvegarde les documents traites en JSON Lines.

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
        logger.info("Saved %d documents to %s", len(docs), path)

    def load_processed(self, path: str) -> list[dict]:
        """Charge des documents depuis un fichier JSON Lines.

        Parameters
        ----------
        path : str
            Chemin du fichier a charger.

        Returns
        -------
        list[dict]
            Liste de documents.
        """
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    if "title" not in doc or "text" not in doc:
                        raise ValueError(f"Invalid document structure: {doc.keys()}")
                    docs.append(doc)
        return docs
