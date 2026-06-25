"""Tests pour le moteur de recherche."""

import pytest

from src.database import DatabaseManager
from src.embeddings import EmbeddingGenerator
from src.search import SemanticSearchEngine


@pytest.fixture(scope="module")
def engine(db_test_url, sample_docs, sample_embeddings):
    try:
        db = DatabaseManager(db_test_url)
    except Exception:
        pytest.skip("Test database not available")
    db.execute_sql_file("sql/01_schema.sql")
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM documents")
        cur.execute("DELETE FROM search_logs")
    db.conn.commit()
    db.insert_documents(sample_docs, sample_embeddings)
    gen = EmbeddingGenerator()
    eng = SemanticSearchEngine(db, gen)
    yield eng
    db.close()


def test_semantic_top_k(engine):
    results = engine.search_semantic("sports competition", top_k=3)
    assert len(results) == 3


def test_similarity_scores_range(engine):
    results = engine.search_semantic("technology innovation", top_k=5)
    for r in results:
        assert 0.0 <= r["similarity_score"] <= 1.0


def test_empty_query_raises_valueerror(engine):
    with pytest.raises(ValueError):
        engine.search_semantic("")


def test_compare_methods_keys(engine):
    result = engine.compare_methods("world news")
    assert "semantic" in result
    assert "tfidf" in result
    assert "overlap" in result
    assert "semantic_time_ms" in result
    assert "tfidf_time_ms" in result


def test_tfidf_non_empty(engine):
    results = engine.search_tfidf("business finance")
    assert len(results) >= 1


def test_search_logs_insertion(engine):
    with engine.db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM search_logs")
        before = cur.fetchone()[0]
    engine.search_semantic("test query for logging")
    with engine.db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM search_logs")
        after = cur.fetchone()[0]
    assert after > before
