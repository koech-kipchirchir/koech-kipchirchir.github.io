"""
Knowledge system utilities: config and logging setup.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger("aios.knowledge")


def setup_logger(name: str = "aios.knowledge", level: int = logging.INFO) -> logging.Logger:
    """Get or create a knowledge system logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(level)
    return log


def structured_log(level: int, event: str, **kwargs: Any) -> None:
    """Emit a structured JSON log record."""
    record = {"event": event, "ts": time.time()}
    record.update(kwargs)
    logger.log(level, "%s", json.dumps(record, default=str))


@dataclass
class KnowledgeConfig:
    """Configuration for the knowledge system."""

    data_dir: str | Path = "knowledge_data"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_entities_per_doc: int = 200
    max_relations_per_doc: int = 500
    enable_graph: bool = True
    enable_fact_store: bool = True
    enable_semantic_search: bool = True
    enable_keyword_search: bool = True
    enable_reasoning: bool = True
    enable_validation: bool = True
    version_limit: int = 50
    similarity_top_k: int = 10
    keyword_top_k: int = 20
    hybrid_top_k: int = 10
    graph_traversal_depth: int = 3
    conf_threshold: float = 0.3
    auto_save_interval: int = 300
    extractor_batch_size: int = 50

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
