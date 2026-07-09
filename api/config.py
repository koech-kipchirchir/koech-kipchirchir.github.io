from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from aios_core.config import AiosConfig


@dataclass
class ApiConfig:
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_prefix: str = "/v1"
    api_title: str = "AIOS API"
    api_version: str = "1.0.0"
    api_description: str = "Production-grade API for the AIOS platform"
    api_docs_url: str = "/docs"
    api_redoc_url: str = "/redoc"
    api_openapi_url: str = "/openapi.json"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    log_datefmt: str = "%Y-%m-%d %H:%M:%S"
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    rate_limit_redis_url: str = ""
    request_id_header: str = "X-Request-ID"
    max_request_size_mb: int = 10
    model: str = "gpt-4o"
    provider: str = "mock"
    api_key: str = ""
    api_base: str = ""
    enable_caching: bool = True
    health_ready_timeout_seconds: int = 30
    memory_db_path: str = "memory_data/memory.db"
    memory_vector_db_path: str = "memory_data/vectors"
    memory_embedding_model: str = "all-MiniLM-L6-v2"
    memory_max_turns: int = 50
    memory_auto_summarize: bool = True
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_vector_db_path: str = "rag_data/vectors"
    rag_similarity_top_k: int = 5
    rag_hybrid_alpha: float = 0.7
    rag_enable_chromadb: bool = True

    @classmethod
    def from_aios_config(cls, cfg: AiosConfig) -> ApiConfig:
        return cls(
            api_host=cfg.api_host,
            api_port=cfg.api_port,
            api_workers=cfg.api_workers,
            log_level=cfg.log_level,
        )

    @classmethod
    def from_env(cls) -> ApiConfig:
        return cls(
            api_host=os.getenv("AIOS_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("AIOS_API_PORT", "8000")),
            api_workers=int(os.getenv("AIOS_API_WORKERS", "1")),
            log_level=os.getenv("AIOS_LOG_LEVEL", "INFO"),
            model=os.getenv("AIOS_MODEL", "gpt-4o"),
            provider=os.getenv("AIOS_PROVIDER", "mock"),
            api_key=os.getenv("AIOS_API_KEY", ""),
            api_base=os.getenv("AIOS_API_BASE", ""),
            rate_limit_enabled=os.getenv("AIOS_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests=int(os.getenv("AIOS_RATE_LIMIT_REQUESTS", "60")),
            cors_allow_origins=os.getenv("AIOS_CORS_ORIGINS", "*").split(","),
            memory_db_path=os.getenv("AIOS_MEMORY_DB_PATH", "memory_data/memory.db"),
            memory_vector_db_path=os.getenv("AIOS_MEMORY_VECTOR_DB_PATH", "memory_data/vectors"),
            memory_embedding_model=os.getenv("AIOS_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            memory_max_turns=int(os.getenv("AIOS_MEMORY_MAX_TURNS", "50")),
            memory_auto_summarize=os.getenv("AIOS_MEMORY_AUTO_SUMMARIZE", "true").lower() == "true",
            rag_chunk_size=int(os.getenv("AIOS_RAG_CHUNK_SIZE", "512")),
            rag_chunk_overlap=int(os.getenv("AIOS_RAG_CHUNK_OVERLAP", "64")),
            rag_embedding_model=os.getenv("AIOS_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            rag_vector_db_path=os.getenv("AIOS_RAG_VECTOR_DB_PATH", "rag_data/vectors"),
            rag_similarity_top_k=int(os.getenv("AIOS_RAG_SIMILARITY_TOP_K", "5")),
            rag_hybrid_alpha=float(os.getenv("AIOS_RAG_HYBRID_ALPHA", "0.7")),
            rag_enable_chromadb=os.getenv("AIOS_RAG_ENABLE_CHROMADB", "true").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dataclass_fields__.items()
            if not k.startswith("_")
        }
