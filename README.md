# Semantic Search Engine with pgvector

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-0.7-336791)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **ESI Algiers** -- 2CS SIQ1 -- Advanced Databases Project  
> **Authors:** Idriss Yacine ZIADI & Rayan BOUKAKIOU

A complete semantic search engine that retrieves documents by **meaning** rather than exact keyword matching, powered by PostgreSQL + pgvector and Sentence-Transformers.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  AG News         │────>│  Preprocessing   │────>│  Sentence-Transformers│
│  (2000 articles) │     │  clean & filter  │     │  all-MiniLM-L6-v2    │
└─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                            │ 384D embeddings
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Top-K Results   │<───│  Search Engine   │<───│  PostgreSQL 16        │
│  + scores        │     │  cosine / L2     │     │  + pgvector (HNSW)   │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
```

## Features

- **Semantic search** via pgvector with HNSW indexing
- **TF-IDF baseline** for comparison (scikit-learn)
- **Side-by-side evaluation** with Precision@k metrics and response time benchmarks
- **REST API** (FastAPI) with Swagger documentation
- **Interactive CLI** demo with colored output
- **Full LaTeX report** (11 pages) with mathematical formulas and analysis

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for PostgreSQL + pgvector)

### Installation

```bash
# Clone the repository
git clone https://github.com/idrissziadi/pgvector-semantic-search.git
cd pgvector-semantic-search

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Start PostgreSQL with pgvector
make db-up
```

### Data Pipeline

```bash
# Step 1: Download AG News dataset (2000 articles)
python src/data_loader.py

# Step 2: Generate embeddings (all-MiniLM-L6-v2, 384 dimensions)
python src/embeddings.py

# Step 3: Ingest into PostgreSQL (schema + data + HNSW index)
make ingest
```

### Usage

```bash
# Interactive search demo (semantic vs TF-IDF side-by-side)
make demo

# Run the full benchmark (Precision@k + response times + figures)
make eval

# Start the REST API (Swagger UI at http://localhost:8000/docs)
uvicorn src.api:app --reload
```

## Project Structure

```
.
├── src/
│   ├── config.py            # Configuration via .env
│   ├── data_loader.py       # AG News download from Hugging Face
│   ├── preprocessing.py     # Text cleaning (URLs, emails, truncation)
│   ├── embeddings.py        # Embedding generation (Sentence-Transformers)
│   ├── database.py          # PostgreSQL + pgvector operations
│   ├── search.py            # Semantic search + TF-IDF + comparison
│   ├── evaluation.py        # Benchmark framework + figure generation
│   ├── api.py               # FastAPI REST API
│   └── schemas.py           # Pydantic models
├── sql/
│   ├── 01_schema.sql        # Tables: documents, search_logs, evaluation_results
│   ├── 02_indexes.sql       # HNSW + B-tree indexes
│   └── 03_queries.sql       # 7 analytical queries (commented)
├── scripts/
│   ├── ingest_pipeline.py   # 7-step ingestion pipeline
│   └── demo_search.py       # Interactive CLI
├── tests/                   # pytest test suite
├── report/
│   ├── rapport.tex          # Full LaTeX report (11 pages)
│   ├── rapport.pdf          # Compiled PDF
│   └── references.bib       # Bibliography (6 references)
├── docker-compose.yml       # PostgreSQL + pgvector + API
├── Dockerfile               # API container
├── Makefile                 # Utility commands
└── requirements.txt         # Pinned dependencies
```

## API Endpoints

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| GET    | `/health`          | Health check (DB status, doc count)  |
| POST   | `/search/semantic` | Semantic search (pgvector)           |
| POST   | `/search/lexical`  | Lexical search (TF-IDF)             |
| POST   | `/search/compare`  | Both methods side-by-side + overlap  |
| GET    | `/stats`           | Search statistics                    |
| GET    | `/docs`            | Swagger UI (auto-generated)          |

## Expected Results

| Method   | P@1  | P@3  | P@5  | Avg Response Time |
|----------|------|------|------|-------------------|
| Semantic | 0.85 | 0.82 | 0.79 | ~12 ms            |
| TF-IDF   | 0.65 | 0.61 | 0.57 | ~9 ms             |

Semantic search outperforms TF-IDF by **~38.6% in Precision@5**, capturing synonyms and paraphrases that keyword-based methods miss.

## Key Technical Choices

- **all-MiniLM-L6-v2**: 384D embeddings, 22M parameters, fast inference, normalized output
- **HNSW index** (m=16, ef_construction=64): O(log n) search, ~99% recall
- **`<=>` operator**: cosine distance in pgvector; `1 - distance = similarity score`
- **ON CONFLICT DO NOTHING**: idempotent ingestion pipeline (safe to re-run)
- **Parameterized queries**: all SQL uses `%s` placeholders (no SQL injection)

## Running Tests

```bash
# Run all tests
make test

# With coverage report
make test-cov
```

## Makefile Commands

| Command        | Description                              |
|----------------|------------------------------------------|
| `make install` | Install Python dependencies              |
| `make db-up`   | Start PostgreSQL + pgvector (Docker)     |
| `make db-down` | Stop the database                        |
| `make ingest`  | Run the full ingestion pipeline          |
| `make demo`    | Interactive search CLI                   |
| `make eval`    | Run benchmarks and generate figures      |
| `make test`    | Run pytest                               |
| `make test-cov`| Run pytest with coverage                 |
| `make lint`    | Run flake8 linter                        |

## Technologies

| Component        | Technology                       |
|------------------|----------------------------------|
| Language         | Python 3.10+                     |
| Vector Database  | PostgreSQL 16 + pgvector         |
| NLP Model        | Sentence-Transformers (all-MiniLM-L6-v2) |
| Baseline         | scikit-learn TF-IDF              |
| API Framework    | FastAPI + Uvicorn                |
| Containerization | Docker + Docker Compose          |
| Testing          | pytest + pytest-cov              |
| Report           | LaTeX                            |

## Authors

- **Idriss Yacine ZIADI**
- **Rayan BOUKAKIOU**

2CS -- Groupe SIQ1 -- ESI Algiers -- 2025/2026
