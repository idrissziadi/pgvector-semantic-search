"""Generation d'embeddings avec Sentence-Transformers."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generateur d'embeddings via Sentence-Transformers.

    Parameters
    ----------
    model_name : str, optional
        Nom du modele. Par defaut all-MiniLM-L6-v2.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or Config.MODEL_NAME
        self.model = SentenceTransformer(self.model_name)
        self.device = str(self.model.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Model %s loaded on %s (dim=%d)",
            self.model_name, self.device, self.embedding_dim
        )

    def generate(
        self,
        texts: list[str],
        batch_size: int = Config.BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Genere les embeddings pour une liste de textes.

        Parameters
        ----------
        texts : list[str]
            Textes a vectoriser.
        batch_size : int
            Taille des batches.
        show_progress : bool
            Afficher la barre de progression.

        Returns
        -------
        np.ndarray
            Matrice (n, 384) d'embeddings normalises.
        """
        safe_texts = [t if t.strip() else " " for t in texts]
        start = time.time()
        # normalize_embeddings=True pour que la distance cosinus soit directement utilisable
        embeddings = self.model.encode(
            safe_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        elapsed = time.time() - start
        logger.info(
            "Generated %d embeddings in %.1fs (%.0f docs/s)",
            len(texts), elapsed, len(texts) / elapsed if elapsed > 0 else 0,
        )
        return np.array(embeddings)

    def generate_single(self, text: str) -> np.ndarray:
        """Vectorise une seule requete.

        Parameters
        ----------
        text : str
            Texte a vectoriser.

        Returns
        -------
        np.ndarray
            Vecteur de dimension (384,).
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        embedding = self.model.encode(
            text, normalize_embeddings=True
        )
        return np.array(embedding)

    def save_embeddings(
        self, embeddings: np.ndarray, path: str = "data/processed/embeddings.npy"
    ) -> None:
        """Sauvegarde les embeddings sur disque.

        Parameters
        ----------
        embeddings : np.ndarray
            Matrice d'embeddings.
        path : str
            Chemin du fichier de sortie.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        logger.info("Saved embeddings %s to %s (%.1f MB)", embeddings.shape, path, size_mb)

    def load_embeddings(self, path: str) -> np.ndarray:
        """Charge des embeddings depuis un fichier .npy.

        Parameters
        ----------
        path : str
            Chemin du fichier.

        Returns
        -------
        np.ndarray
            Matrice d'embeddings.
        """
        embeddings = np.load(path)
        if embeddings.ndim == 2 and embeddings.shape[1] != Config.EMBEDDING_DIM:
            raise ValueError(
                f"Expected dim {Config.EMBEDDING_DIM}, got {embeddings.shape[1]}"
            )
        return embeddings


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    from src.preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    docs = preprocessor.load_processed("data/processed/documents_clean.jsonl")

    gen = EmbeddingGenerator()
    texts = [d["full_text"] for d in docs]
    embeddings = gen.generate(texts)
    gen.save_embeddings(embeddings)
    logger.info("Final shape: %s", embeddings.shape)
