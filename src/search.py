"""Moteur de recherche semantique et comparaison avec TF-IDF."""

import logging
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database import DatabaseManager
from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Moteur de recherche combinant recherche semantique (pgvector) et TF-IDF.

    Parameters
    ----------
    db_manager : DatabaseManager
        Gestionnaire de base de donnees.
    embedding_gen : EmbeddingGenerator
        Generateur d'embeddings.
    """

    def __init__(
        self, db_manager: DatabaseManager, embedding_gen: EmbeddingGenerator
    ) -> None:
        self.db = db_manager
        self.embedding_gen = embedding_gen
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._tfidf_docs: list[dict] = []

    def search_semantic(
        self, query: str, top_k: int = 5, metric: str = "cosine"
    ) -> list[dict]:
        """Recherche semantique via pgvector.

        Parameters
        ----------
        query : str
            Requete en langage naturel.
        top_k : int
            Nombre de resultats.
        metric : str
            'cosine' ou 'l2'.

        Returns
        -------
        list[dict]
            Resultats avec id, title, content, category, similarity_score.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_embedding = self.embedding_gen.generate_single(query)
        start = time.perf_counter()

        if metric == "cosine":
            sql = """
                SELECT id, title, content, category,
                       1 - (embedding <=> %s::vector) AS similarity_score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
        else:
            sql = """
                SELECT id, title, content, category,
                       1.0 / (1.0 + (embedding <-> %s::vector)) AS similarity_score
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """

        vec = query_embedding.tolist()
        with self.db.conn.cursor() as cur:
            cur.execute(sql, (vec, vec, top_k))
            rows = cur.fetchall()

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            {
                "id": r[0],
                "title": r[1],
                "content": r[2],
                "category": r[3],
                "similarity_score": float(r[4]),
            }
            for r in rows
        ]

        try:
            with self.db.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO search_logs (query_text, query_embedding, top_k, "
                    "similarity_metric, execution_time_ms, results_count) "
                    "VALUES (%s, %s::vector, %s, %s, %s, %s)",
                    (query, vec, top_k, metric, elapsed_ms, len(results)),
                )
            self.db.conn.commit()
        except Exception:
            self.db.conn.rollback()

        return results

    def _build_tfidf_index(self) -> None:
        with self.db.conn.cursor() as cur:
            cur.execute("SELECT id, title, content, category FROM documents ORDER BY id")
            rows = cur.fetchall()

        self._tfidf_docs = [
            {"id": r[0], "title": r[1], "content": r[2], "category": r[3]}
            for r in rows
        ]
        corpus = [d["title"] + " " + d["content"] for d in self._tfidf_docs]
        self._tfidf_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        logger.info("TF-IDF index built: %d documents", len(corpus))

    def search_tfidf(self, query: str, top_k: int = 5) -> list[dict]:
        """Recherche par TF-IDF.

        Parameters
        ----------
        query : str
            Requete textuelle.
        top_k : int
            Nombre de resultats.

        Returns
        -------
        list[dict]
            Resultats avec id, title, content, category, similarity_score.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self._tfidf_vectorizer is None:
            self._build_tfidf_index()

        query_vec = self._tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self._tfidf_docs[idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "category": doc["category"],
                "similarity_score": float(scores[idx]),
            })
        return results

    def compare_methods(self, query: str, top_k: int = 5) -> dict:
        """Compare recherche semantique et TF-IDF.

        Parameters
        ----------
        query : str
            Requete a comparer.
        top_k : int
            Nombre de resultats par methode.

        Returns
        -------
        dict
            Resultats des deux methodes avec overlap.
        """
        start = time.perf_counter()
        semantic = self.search_semantic(query, top_k)
        semantic_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        tfidf = self.search_tfidf(query, top_k)
        tfidf_time = (time.perf_counter() - start) * 1000

        semantic_ids = {r["id"] for r in semantic}
        tfidf_ids = {r["id"] for r in tfidf}
        overlap = len(semantic_ids & tfidf_ids)

        return {
            "query": query,
            "semantic": semantic,
            "tfidf": tfidf,
            "overlap": overlap,
            "semantic_time_ms": semantic_time,
            "tfidf_time_ms": tfidf_time,
        }
