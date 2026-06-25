"""Fixtures pytest pour les tests du moteur de recherche semantique."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def sample_docs():
    docs = []
    categories = ["World", "Sports", "Business", "Sci/Tech"]
    for i in range(10):
        cat = categories[i % 4]
        docs.append({
            "id": i,
            "title": f"Test Article {i} about {cat}",
            "text": f"This is the content of article {i} covering {cat} topics in detail with enough words.",
            "category": cat,
            "source": "ag_news",
            "full_text": f"Test Article {i} about {cat}. This is the content of article {i} covering {cat} topics in detail with enough words.",
            "char_count": 80,
            "word_count": 15,
        })
    return docs


@pytest.fixture(scope="session")
def sample_embeddings():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((10, 384))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture(scope="session")
def db_test_url():
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/semantic_search_test"
    )
