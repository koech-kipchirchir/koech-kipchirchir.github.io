from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: str = "recursive"
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    vector_db_path: str | Path = "rag_data/vectors"
    db_path: str | Path = "rag_data/index.db"
    similarity_top_k: int = 5
    hybrid_search_alpha: float = 0.7
    rerank_top_k: int = 3
    enable_chromadb: bool = True
    enable_faiss: bool = False
    device: str = "cpu"
    batch_size: int = 32
    log_level: int = logging.INFO
