"""Configuration du projet chargee depuis les variables d'environnement."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/semantic_search"
    )
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "semantic_search")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_CHARS: int = 512
    EMBEDDING_DIM: int = 384
