"""
AIOS Fine-tuning Utilities

Shared helpers for GPU detection, logging, and common setup tasks.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_gpu() -> Tuple[bool, str]:
    """Detect a suitable GPU for training.

    Returns:
        A tuple ``(available, description)``.
    """
    if not torch.cuda.is_available():
        return False, "No CUDA-capable GPU detected."

    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    cap = torch.cuda.get_device_capability(0)

    desc = (
        f"{count}x {name} | {mem_gb:.1f} GB VRAM | "
        f"Compute {cap[0]}.{cap[1]}"
    )
    return True, desc


def require_gpu() -> str:
    """Exit gracefully if no GPU is available.

    Returns:
        A human-readable GPU description string.
    """
    available, desc = detect_gpu()
    if not available:
        print("=" * 60)
        print("  No compatible GPU detected.")
        print("  AIOS fine-tuning requires a CUDA-capable GPU with")
        print("  at least 8 GB of VRAM for QLoRA training.")
        print("=" * 60)
        sys.exit(0)
    return desc


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(
    name: str = "aios.finetuning",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name:     Logger name (hierarchical, e.g. ``"aios.finetuning"``).
        level:    Logging level.
        log_file: Optional file path to write logs to.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    """Walk upward from ``cwd`` to find the AIOS project root.

    The root is identified by the presence of an ``aios_core/`` directory
    or a ``settings.gradle.kts`` file.

    Returns:
        Absolute ``Path`` to the project root.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "aios_core").is_dir() or (parent / "settings.gradle.kts").exists():
            return parent
    return cwd


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def get_compute_dtype(dtype_str: str = "auto") -> torch.dtype:
    """Resolve a string dtype to a torch dtype.

    Args:
        dtype_str: One of ``"auto"``, ``"float16"``, ``"bfloat16"``,
                   ``"float32"``.

    Returns:
        The corresponding ``torch.dtype``.
    """
    mapping = {
        "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.float16)


def format_timestamp(ts: Optional[datetime] = None) -> str:
    """Return a compact, filesystem-safe timestamp string."""
    dt = ts or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")
