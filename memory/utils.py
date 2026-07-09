from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_LOGERS: dict[str, logging.Logger] = {}


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    if name in _LOGERS:
        return _LOGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    _LOGERS[name] = logger
    return logger


def thread_safe(lock_attr: str = "_lock") -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            lock: threading.Lock = getattr(self, lock_attr)
            with lock:
                return func(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_ms() -> int:
    return int(now_utc().timestamp() * 1000)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass
class MemoryConfig:
    db_path: str | Path = "memory_data/memory.db"
    vector_db_path: str | Path = "memory_data/vectors"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_conversation_turns: int = 50
    max_memory_nodes: int = 1000
    memory_importance_threshold: float = 0.3
    similarity_top_k: int = 10
    enable_chromadb: bool = True
    enable_faiss: bool = False
    auto_summarize: bool = True
    summary_interval_turns: int = 10
    cleanup_interval_hours: int = 24
    memory_ttl_days: int = 90
    log_level: int = logging.INFO
    device: str = "cpu"
