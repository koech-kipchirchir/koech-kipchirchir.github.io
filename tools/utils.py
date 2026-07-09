from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
class ToolConfig:
    permission_mode: str = "explicit"
    allowed_commands: list[str] = field(default_factory=lambda: ["ls", "cat", "pwd", "echo", "python", "pip"])
    blocked_commands: list[str] = field(default_factory=lambda: ["rm", "sudo", "chmod", "dd", "mkfs", "shutdown", "reboot"])
    sandbox_enabled: bool = True
    max_output_length: int = 10000
    request_timeout: int = 30
