"""API REST FastAPI pour le moteur de recherche semantique."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingGenerator
from src.schemas import (
    CompareResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.search import SemanticSearchEngine

logger = logging.getLogger(__name__)

db_manager: DatabaseManager | None = None
embedding_gen: EmbeddingGenerator | None = None
search_engine: SemanticSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, embedding_gen, search_engine
    db_manager = DatabaseManager()
    embedding_gen = EmbeddingGenerator()
    search_engine = SemanticSearchEngine(db_manager, embedding_gen)
    logger.info("API initialized")
    yield
    if db_manager:
        db_manager.close()
    logger.info("API shutdown")


app = FastAPI(
    title="Semantic Search API",
    description="Moteur de recherche semantique avec pgvector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    connected = db_manager is not None and db_manager.conn and not db_manager.conn.closed
    count = 0
    if connected:
        try:
            with db_manager.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                count = cur.fetchone()[0]
        except Exception:
            connected = False
    return {
        "status": "ok",
        "db_connected": connected,
        "docs_count": count,
        "model": Config.MODEL_NAME,
    }


@app.post("/search/semantic", response_model=SearchResponse)
def search_semantic(req: SearchRequest):
    try:
        start = time.perf_counter()
        results = search_engine.search_semantic(req.query, req.top_k, req.metric)
        elapsed = (time.perf_counter() - start) * 1000
        return SearchResponse(
            query=req.query,
            results=[
                SearchResult(
                    id=r["id"], title=r["title"], content=r["content"],
                    category=r["category"], score=r["similarity_score"],
                )
                for r in results
            ],
            time_ms=elapsed,
            method="semantic",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/lexical", response_model=SearchResponse)
def search_lexical(req: SearchRequest):
    try:
        start = time.perf_counter()
        results = search_engine.search_tfidf(req.query, req.top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return SearchResponse(
            query=req.query,
            results=[
                SearchResult(
                    id=r["id"], title=r["title"], content=r["content"],
                    category=r["category"], score=r["similarity_score"],
                )
                for r in results
            ],
            time_ms=elapsed,
            method="tfidf",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/compare", response_model=CompareResponse)
def search_compare(req: SearchRequest):
    try:
        result = search_engine.compare_methods(req.query, req.top_k)
        return CompareResponse(
            query=req.query,
            semantic=[
                SearchResult(
                    id=r["id"], title=r["title"], content=r["content"],
                    category=r["category"], score=r["similarity_score"],
                )
                for r in result["semantic"]
            ],
            tfidf=[
                SearchResult(
                    id=r["id"], title=r["title"], content=r["content"],
                    category=r["category"], score=r["similarity_score"],
                )
                for r in result["tfidf"]
            ],
            overlap_count=result["overlap"],
            semantic_time_ms=result["semantic_time_ms"],
            tfidf_time_ms=result["tfidf_time_ms"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    try:
        stats = db_manager.get_stats()
        with db_manager.conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(AVG(execution_time_ms), 0), COUNT(*) "
                "FROM search_logs WHERE searched_at > NOW() - INTERVAL '24 hours'"
            )
            row = cur.fetchone()
        return {
            "total_documents": stats["total_documents"],
            "avg_search_time_ms": float(row[0]),
            "searches_today": row[1],
            "index_type": "HNSW",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
