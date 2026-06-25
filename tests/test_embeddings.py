"""Tests pour le module de generation d'embeddings."""

import os
import tempfile

import numpy as np
import pytest

from src.embeddings import EmbeddingGenerator


@pytest.fixture(scope="module")
def generator():
    return EmbeddingGenerator()


def test_generate_shape(generator):
    texts = ["Hello world", "Test sentence", "Another text", "More data", "Final one"]
    embeddings = generator.generate(texts, show_progress=False)
    assert embeddings.shape == (5, 384)


def test_generate_normalized(generator):
    texts = ["Hello world", "Test sentence"]
    embeddings = generator.generate(texts, show_progress=False)
    for i in range(len(texts)):
        norm = np.linalg.norm(embeddings[i])
        assert abs(norm - 1.0) < 1e-5


def test_generate_single_shape(generator):
    embedding = generator.generate_single("Hello world")
    assert embedding.shape == (384,)


def test_generate_single_empty_raises(generator):
    with pytest.raises(ValueError):
        generator.generate_single("")


def test_save_load_roundtrip(generator):
    embeddings = generator.generate(["test text"], show_progress=False)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        path = f.name
    try:
        generator.save_embeddings(embeddings, path)
        loaded = generator.load_embeddings(path)
        np.testing.assert_array_almost_equal(embeddings, loaded)
    finally:
        os.unlink(path)
