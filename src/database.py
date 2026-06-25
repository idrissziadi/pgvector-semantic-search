"""Gestion de la base de donnees PostgreSQL + pgvector."""

import logging
import time
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestionnaire de connexion et operations sur PostgreSQL + pgvector.

    Parameters
    ----------
    database_url : str, optional
        URL de connexion PostgreSQL.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or Config.DATABASE_URL
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = False
        try:
            register_vector(self.conn)
        except Exception:
            pass

    def _reconnect(self) -> None:
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
        except Exception:
            pass
        self._connect()

    def execute_sql_file(self, filepath: str) -> None:
        """Execute un fichier SQL complet dans une transaction.

        Parameters
        ----------
        filepath : str
            Chemin du fichier SQL.
        """
        sql = Path(filepath).read_text(encoding="utf-8")
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
            self.conn.commit()
            logger.info("Executed %s", filepath)
        except Exception as e:
            self.conn.rollback()
            logger.error("Error executing %s: %s", filepath, e)
            raise

    def insert_documents(
        self,
        docs: list[dict],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """Insere des documents avec leurs embeddings dans la base.

        Parameters
        ----------
        docs : list[dict]
            Documents a inserer.
        embeddings : np.ndarray
            Matrice d'embeddings correspondante.
        batch_size : int
            Taille des batches d'insertion.

        Returns
        -------
        int
            Nombre de lignes inserees.
        """
        from tqdm import tqdm

        total_inserted = 0
        values = []
        for i, doc in enumerate(docs):
            values.append((
                doc["title"],
                doc.get("text", doc.get("content", "")),
                doc.get("category", ""),
                doc.get("source", "ag_news"),
                doc.get("char_count", len(doc.get("text", ""))),
                doc.get("word_count", len(doc.get("text", "").split())),
                embeddings[i].tolist(),
            ))

        insert_sql = """
            INSERT INTO documents (title, content, category, source, char_count, word_count, embedding)
            VALUES %s
            ON CONFLICT (title, source) DO NOTHING
        """
        template = "(%s, %s, %s, %s, %s, %s, %s::vector)"

        with self.conn.cursor() as cur:
            for start in tqdm(range(0, len(values), batch_size), desc="Inserting"):
                batch = values[start:start + batch_size]
                psycopg2.extras.execute_values(
                    cur, insert_sql, batch, template=template, page_size=batch_size
                )
                self.conn.commit()
                total_inserted += len(batch)

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            actual_count = cur.fetchone()[0]

        duplicates = total_inserted - actual_count
        logger.info(
            "Inserted: %d / Total in DB: %d / Duplicates ignored: %d",
            total_inserted, actual_count, max(0, duplicates),
        )
        return actual_count

    def get_stats(self) -> dict:
        """Retourne les statistiques de la base de donnees.

        Returns
        -------
        dict
            Statistiques incluant total_documents, avg_char_count, avg_word_count, indexes.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), COALESCE(AVG(char_count), 0), COALESCE(AVG(word_count), 0) "
                "FROM documents"
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'documents'"
            )
            indexes = [r[0] for r in cur.fetchall()]
        return {
            "total_documents": row[0],
            "avg_char_count": float(row[1]),
            "avg_word_count": float(row[2]),
            "indexes": indexes,
        }

    def close(self) -> None:
        """Ferme la connexion a la base."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")
