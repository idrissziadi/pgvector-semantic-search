"""Schemas Pydantic pour l'API FastAPI."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    metric: Literal["cosine", "l2"] = "cosine"
    category: str | None = None


class SearchResult(BaseModel):
    id: int
    title: str
    content: str
    category: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    time_ms: float
    method: str


class CompareResponse(BaseModel):
    query: str
    semantic: list[SearchResult]
    tfidf: list[SearchResult]
    overlap_count: int
    semantic_time_ms: float
    tfidf_time_ms: float
