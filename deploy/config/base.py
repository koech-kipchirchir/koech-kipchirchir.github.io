"""
Base deployment settings using Pydantic-based env loading.
All settings are overridable via AIOS_* environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DeploymentConfig:
    """Top-level deployment configuration, loaded from env vars.

    Environment variable prefix: AIOS_
    Example: AIOS_HOST=0.0.0.0, AIOS_WORKERS=4
    """

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    log_level: str = "info"
    reload: bool = False

    # --- Inference ---
    provider: str = "ollama"          # openai, ollama, vllm, mock
    model: str = "llama3.1:8b"
    api_base: str = ""                # e.g. http://ollama:11434/v1
    api_key: str = ""

    # --- CORS ---
    cors_allow_origins: str = "*"

    # --- Rate Limiting ---
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window: int = 60

    # --- Memory ---
    memory_ttl_hours: int = 24
    memory_max_items: int = 1000

    # --- RAG ---
    rag_chunk_size: int = 512
    rag_top_k: int = 5

    # --- Paths ---
    data_dir: str = "/data"
    logs_dir: str = "/var/log/aios"

    # --- SSL / TLS (terminated at nginx) ---
    ssl_cert_path: str = ""
    ssl_key_path: str = ""

    # --- Monitoring ---
    enable_prometheus: bool = True
    metrics_port: int = 9090

    # --- Health ---
    health_check_interval: int = 30
    health_check_timeout: int = 10

    @classmethod
    def from_env(cls) -> DeploymentConfig:
        """Load config from AIOS_* environment variables."""
        prefix = "AIOS_"
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            env_name = f"{prefix}{field_name.upper()}"
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            field_type = cls.__dataclass_fields__[field_name].type
            if field_type is bool or field_type == "bool":
                kwargs[field_name] = raw.lower() in ("1", "true", "yes", "on")
            elif field_type is int or field_type == "int":
                kwargs[field_name] = int(raw)
            elif field_type is float or field_type == "float":
                kwargs[field_name] = float(raw)
            else:
                kwargs[field_name] = raw
        return cls(**kwargs)

    def to_app_config(self) -> dict:
        """Convert to dict suitable for patching into ApiConfig/AiosConfig."""
        return {
            "api_host": self.host,
            "api_port": self.port,
            "api_workers": self.workers,
            "log_level": self.log_level,
            "model": self.model,
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "cors_allow_origins": self.cors_allow_origins,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "memory_ttl_hours": self.memory_ttl_hours,
            "memory_max_items": self.memory_max_items,
            "rag_chunk_size": self.rag_chunk_size,
            "rag_top_k": self.rag_top_k,
        }
