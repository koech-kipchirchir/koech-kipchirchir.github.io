"""
Production logging configuration: JSON-structured, rotated, dual-output
(console + file).
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOG_CONFIG: dict[str, Any] | None = None


def structured_json_formatter() -> logging.Formatter:
    """Return a formatter that emits JSON lines."""
    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            obj: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            if record.exc_info and record.exc_info[0]:
                obj["exception"] = self.formatException(record.exc_info)
            if hasattr(record, "request_id"):
                obj["request_id"] = record.request_id
            if hasattr(record, "duration_ms"):
                obj["duration_ms"] = record.duration_ms

            # Merge extra fields from record.__dict__
            for key in ("method", "path", "status_code", "user_id",
                        "session_id", "model_name", "token_count",
                        "tool_name", "memory_count"):
                val = getattr(record, key, None)
                if val is not None:
                    obj[key] = val
            return json.dumps(obj, default=str)

    return JSONFormatter()


def dev_formatter() -> logging.Formatter:
    """Return a human-readable formatter for development."""
    return logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging(
    log_dir: str | Path = "/var/log/aios",
    level: str = "info",
    json_output: bool = True,
    max_bytes: int = 100 * 1024 * 1024,   # 100 MB
    backup_count: int = 10,
    console_level: str = "info",
) -> None:
    """Configure the root AIOS logger for production.

    Writes JSON-structured logs to rotating files and human-readable
    logs to stdout.
    """
    global _LOG_CONFIG
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)
    console_lvl = getattr(logging, console_level.upper(), logging.INFO)

    json_fmt = structured_json_formatter()
    dev_fmt = dev_formatter()

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"format": json_fmt},
            "dev": {"format": dev_fmt},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "level": console_lvl,
                "formatter": "dev",
            },
            "file_json": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_path / "aios.json.log"),
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "level": log_level,
                "formatter": "json",
                "encoding": "utf-8",
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_path / "aios.error.log"),
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "level": logging.ERROR,
                "formatter": json_fmt if json_output else dev_fmt,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "aios": {
                "level": log_level,
                "handlers": ["console", "file_json", "file_error"],
                "propagate": False,
            },
            "aios.api": {
                "level": log_level,
                "handlers": ["console", "file_json"],
                "propagate": False,
            },
            "aios.api.middleware": {
                "level": logging.DEBUG if level == "debug" else logging.INFO,
                "handlers": ["console", "file_json"],
                "propagate": False,
            },
            "uvicorn": {
                "level": logging.WARNING,
                "handlers": ["console", "file_json"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": logging.WARNING,
                "handlers": ["console", "file_json"],
                "propagate": False,
            },
        },
        "root": {
            "level": logging.WARNING,
            "handlers": ["console", "file_error"],
        },
    }

    logging.config.dictConfig(config)
    _LOG_CONFIG = config

    logger = logging.getLogger("aios")
    logger.info("Logging configured: dir=%s level=%s json=%s",
                log_path, level, json_output)


def get_log_config() -> dict[str, Any] | None:
    """Return the current logging configuration dict."""
    return _LOG_CONFIG
