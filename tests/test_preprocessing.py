"""Tests pour le module de preprocessing."""

import json
import os
import tempfile

import pytest

from src.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_clean_text_removes_url(preprocessor):
    text = "Visit https://arxiv.org/abs/123 for details"
    result = preprocessor.clean_text(text)
    assert "https://arxiv.org" not in result
    assert "for details" in result


def test_clean_text_removes_email(preprocessor):
    text = "Contact me at foo@bar.com for info"
    result = preprocessor.clean_text(text)
    assert "foo@bar.com" not in result
    assert "for info" in result


def test_clean_text_truncates_long_text(preprocessor):
    text = "A" * 1000
    result = preprocessor.clean_text(text)
    assert len(result) <= 512


def test_clean_text_empty_string(preprocessor):
    result = preprocessor.clean_text("")
    assert result == ""


def test_process_documents_adds_fields(preprocessor):
    docs = [{"title": "Test Title", "text": "This is test content with enough words to pass the filter"}]
    result = preprocessor.process_documents(docs)
    assert len(result) == 1
    assert "full_text" in result[0]
    assert "char_count" in result[0]
    assert "word_count" in result[0]


def test_process_documents_filters_short(preprocessor):
    docs = [{"title": "Hi", "text": "short"}]
    result = preprocessor.process_documents(docs)
    assert len(result) == 0


def test_save_and_load(preprocessor):
    docs = [
        {
            "title": "Test",
            "text": "Content here that is long enough to pass all filters",
            "full_text": "Test. Content here that is long enough to pass all filters",
            "char_count": 50,
            "word_count": 10,
        }
    ]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        preprocessor.save_processed(docs, path)
        loaded = preprocessor.load_processed(path)
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test"
    finally:
        os.unlink(path)
