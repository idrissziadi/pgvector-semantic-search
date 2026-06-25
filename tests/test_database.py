"""Tests pour le module de gestion de base de donnees."""

import numpy as np
import psycopg2
import pytest

from src.database import DatabaseManager


@pytest.fixture(scope="module")
def db(db_test_url):
    try:
        manager = DatabaseManager(db_test_url)
    except Exception:
        pytest.skip("Test database not available")
    manager.execute_sql_file("sql/01_schema.sql")
    yield manager
    try:
        with manager.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS evaluation_results CASCADE")
            cur.execute("DROP TABLE IF EXISTS search_logs CASCADE")
            cur.execute("DROP TABLE IF EXISTS documents CASCADE")
        manager.conn.commit()
    except Exception:
        pass
    manager.close()


def test_connection_status(db):
    assert db.conn.status == psycopg2.extensions.STATUS_READY


def test_schema_creation(db):
    with db.conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'documents'"
        )
        assert cur.fetchone() is not None


def test_insert_returns_count(db, sample_docs, sample_embeddings):
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM documents")
    db.conn.commit()
    docs = sample_docs[:5]
    emb = sample_embeddings[:5]
    count = db.insert_documents(docs, emb)
    assert count == 5


def test_insert_deduplication(db, sample_docs, sample_embeddings):
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM documents")
    db.conn.commit()
    docs = [sample_docs[0]]
    emb = sample_embeddings[:1]
    db.insert_documents(docs, emb)
    db.insert_documents(docs, emb)
    with db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents WHERE title = %s", (docs[0]["title"],))
        assert cur.fetchone()[0] == 1


def test_get_stats_keys(db):
    stats = db.get_stats()
    assert "total_documents" in stats
    assert "avg_char_count" in stats
    assert "avg_word_count" in stats
    assert "indexes" in stats


def test_embedding_stored_dimension(db, sample_docs, sample_embeddings):
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM documents")
    db.conn.commit()
    db.insert_documents(sample_docs[:1], sample_embeddings[:1])
    with db.conn.cursor() as cur:
        cur.execute("SELECT vector_dims(embedding) FROM documents LIMIT 1")
        dim = cur.fetchone()[0]
    assert dim == 384
