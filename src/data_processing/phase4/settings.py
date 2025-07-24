"""Centralised configuration & helpers (read from .env if present)."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    # --- core API keys & endpoints
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    bm25_endpoint: str = Field("http://localhost:8000/search", env="BM25_ENDPOINT")
    qdrant_collection: str = Field("quantconnect_chunks", env="QDRANT_COLLECTION")
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")

    # --- cache behaviour
    cache_ttl_hours: int = Field(24, env="CACHE_TTL_HOURS")  # 0 = infinite
    index_version: str = Field("2025-07-17", env="INDEX_VERSION")

    # --- model hierarchy
    llm_primary: str = Field("gpt-4o", env="LLM_PRIMARY")
    llm_secondary: str = Field("gpt-4o-mini", env="LLM_SECONDARY")
    llm_fallback: str = Field("gpt-3.5-turbo-0125", env="LLM_FALLBACK")

    model_timeout: int = Field(25, env="LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(2, env="LLM_RETRIES")

    # --- server
    host: str = Field("127.0.0.1", env="HOST")
    port: int = Field(8001, env="PORT")
    api_key_header: Optional[str] = Field(None, env="API_KEY_HEADER")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache() # type: ignore[call-arg]
def get_settings() -> _Settings:
    return _Settings()


# Utility: project root
PROJECT_ROOT = Path(__file__).resolve().parent

