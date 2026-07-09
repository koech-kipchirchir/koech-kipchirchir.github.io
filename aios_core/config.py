"""
AIOS Configuration System

Provides a production-ready, dataclass-based configuration for the AIOS
project.  Automatically detects the project root, resolves all subdirectory
paths, and creates required directories on initialization.

Supports multiple environments (development, staging, production) and is
designed to accommodate the future AIOS-7B model with minimal changes.

Typical usage::

    from aios_core.config import AiosConfig

    cfg = AiosConfig()
    cfg.setup_directories()
    print(cfg.llm_dir)       # /path/to/aios/llm
    print(cfg.api_host)      # 0.0.0.0
    print(cfg.log_level)     # INFO
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_project_root() -> Path:
    """Walk upward from this file's directory looking for the AIOS project root.

    Detection strategy (checked in order):
    1.  The directory containing an ``aios_core`` package.
    2.  The directory containing the root ``build.gradle.kts`` (Android project).
    3.  Current working directory as a last resort.

    Returns:
        Absolute ``Path`` to the project root.
    """
    marker = Path(__file__).resolve().parent.parent  # aios_core/..   (once)
    # -- climb up at most 5 levels looking for aios_core as a sibling
    for candidate in [marker] + list(marker.parents[:5]):
        if (candidate / "aios_core").is_dir() and (candidate / "aios_core" / "__init__.py").exists():
            return candidate.resolve()
        if (candidate / "build.gradle.kts").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


_PROJECT_ROOT: Path = _detect_project_root()

# Default directory name for each subsystem (under the project root).
_SUBDIR_NAMES: Dict[str, str] = {
    "llm":          "llm",
    "datasets":     "datasets",
    "memory":       "memory",
    "logs":         "logs",
    "configs":      "configs",
    "scripts":      "scripts",
    "tests":        "tests",
    "docs":         "docs",
    "agents":       "agents",
    "tools":        "tools",
    "rag":          "rag",
    "voice":        "voice",
    "vision":       "vision",
    "inference":    "inference",
    "finetuning":   "finetuning",
    "benchmarks":   "benchmarks",
    "api":          "api",
    "desktop":      "desktop",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AiosConfig:
    """Central configuration for an AIOS instance.

    All file-system paths are derived from the auto-detected project root
    unless explicitly overridden.

    Attributes:
        app_name:          Human-friendly application name.
        app_version:       Semantic version string.
        environment:       Deployment environment (``development``, ``staging``,
                           ``production``).
        device:            Device identifier (defaults to hostname).
        model_name:        Active LLM model identifier.
        log_level:         Logging verbosity (e.g. ``"DEBUG"``, ``"INFO"``).
        api_host:          Bind address for the HTTP API server.
        api_port:          Bind port for the HTTP API server.
        api_workers:       Number of uvicorn workers (production only).
        max_upload_mb:     Maximum file upload size in megabytes.
        rate_limit:        Max requests per minute per client.
        secret_key:        Signing key for sessions / tokens (set via env var
                           in production).
        allowed_origins:   CORS allowed origins (``["*"]`` in development).

    Path attributes (all resolved to absolute :class:`pathlib.Path`):

        root_dir:          Project root (auto-detected).
        llm_dir:           LLM models, tokenizers, training code.
        datasets_dir:      Training/evaluation datasets.
        memory_dir:        Persistent memory / knowledge base.
        logs_dir:          Application and request logs.
        configs_dir:       User-editable configuration files.
        scripts_dir:       Utility and automation scripts.
        tests_dir:         Test suites.
        docs_dir:          Documentation.
        agents_dir:        Agent definitions.
        tools_dir:         Tool implementations.
        rag_dir:           Retrieval-Augmented Generation assets.
        voice_dir:         Voice/speech models and assets.
        vision_dir:        Vision model assets.
        inference_dir:     Inference engine artifacts.
        finetuning_dir:    Fine-tuning checkpoints and configs.
        benchmarks_dir:    Benchmark results and scripts.
        api_dir:           API specification / client generation.
        desktop_dir:       Desktop application resources.
        checkpoint_dir:    Default model checkpoint directory (under ``llm_dir``).
    """

    # -- Basic metadata -----------------------------------------------------
    app_name: str = "AIOS"
    app_version: str = "2.0.0"
    environment: str = "development"
    device: str = field(default_factory=platform.node)
    model_name: str = "google/flan-t5-small"

    # -- Logging ------------------------------------------------------------
    log_level: str = "INFO"

    # -- API server ---------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    max_upload_mb: int = 10
    rate_limit: int = 60
    secret_key: str = "change-me-in-production"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

    # -- Paths (auto-resolved) ----------------------------------------------
    root_dir: Path = field(default_factory=lambda: _PROJECT_ROOT)

    # Sub-directories – populated by :meth:`_resolve_paths`.
    llm_dir:          Path = field(init=False)
    datasets_dir:     Path = field(init=False)
    memory_dir:       Path = field(init=False)
    logs_dir:         Path = field(init=False)
    configs_dir:      Path = field(init=False)
    scripts_dir:      Path = field(init=False)
    tests_dir:        Path = field(init=False)
    docs_dir:         Path = field(init=False)
    agents_dir:       Path = field(init=False)
    tools_dir:        Path = field(init=False)
    rag_dir:          Path = field(init=False)
    voice_dir:        Path = field(init=False)
    vision_dir:       Path = field(init=False)
    inference_dir:    Path = field(init=False)
    finetuning_dir:   Path = field(init=False)
    benchmarks_dir:   Path = field(init=False)
    api_dir:          Path = field(init=False)
    desktop_dir:      Path = field(init=False)
    checkpoint_dir:   Path = field(init=False)

    # -- Look-up for backward compat / programmatic access ------------------
    _subdir_names: ClassVar[Dict[str, str]] = _SUBDIR_NAMES

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Resolve all path fields and normalise the config."""
        self._resolve_paths()
        self._normalise()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def setup_directories(self, exist_ok: bool = True) -> None:
        """Create every required directory on disk if it does not exist.

        Args:
            exist_ok: If ``True`` (default), no error is raised when a
                      directory already exists.

        Returns:
            ``None``.
        """
        dirs = (
            self.llm_dir,
            self.datasets_dir,
            self.memory_dir,
            self.logs_dir,
            self.configs_dir,
            self.scripts_dir,
            self.tests_dir,
            self.docs_dir,
            self.agents_dir,
            self.tools_dir,
            self.rag_dir,
            self.voice_dir,
            self.vision_dir,
            self.inference_dir,
            self.finetuning_dir,
            self.benchmarks_dir,
            self.api_dir,
            self.desktop_dir,
            self.checkpoint_dir,
        )
        for d in dirs:
            d.mkdir(parents=True, exist_ok=exist_ok)

    def is_production(self) -> bool:
        """Return ``True`` when the current environment is ``"production"``."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return ``True`` when the current environment is ``"development"``."""
        return self.environment.lower() == "development"

    def to_dict(self) -> Dict[str, object]:
        """Export the configuration as a plain dictionary.

        Only init-able fields are included so the dict can be unpacked to
        create a new config instance.  Path objects are kept as ``Path``
        (callers that need strings should convert explicitly).

        Returns:
            A flat dictionary of public constructor attributes.
        """
        result: Dict[str, object] = {}
        for field_def in self.__dataclass_fields__.values():
            if field_def.name.startswith("_") or not field_def.init:
                continue
            result[field_def.name] = getattr(self, field_def.name)
        return result

    def for_model(self, model_name: str) -> "AiosConfig":
        """Return a copy of this config tuned for a specific model.

        This is the extension point for future AIOS-7B support::

            cfg = AiosConfig().for_model("aios-7b")

        The method will adjust batch sizes, context windows, checkpoint
        paths, etc. based on the requested model identifier.

        Args:
            model_name:  Model identifier (e.g. ``"aios-7b"``,
                         ``"google/flan-t5-small"``).

        Returns:
            A new ``AiosConfig`` instance with model-specific overrides.
        """
        overrides: Dict[str, object] = {"model_name": model_name}

        model_lower = model_name.lower().replace("-", "_").replace("/", "_")

        if "7b" in model_lower or "7_b" in model_lower:
            overrides.update(
                {
                    "app_version": "3.0.0",
                    "log_level": "INFO",
                }
            )

        return AiosConfig(**{**self.to_dict(), **overrides})  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> None:
        """Set every ``*_dir`` field to its absolute path under root."""
        root = self.root_dir.resolve()
        for attr, subdir in self._subdir_names.items():
            path = root / subdir
            setattr(self, f"{attr}_dir", path)
        # Special case: checkpoints live inside llm/
        self.checkpoint_dir = self.llm_dir / "checkpoints"

    def _normalise(self) -> None:
        """Coerce string values to their canonical forms."""
        self.environment = self.environment.lower().strip()
        self.log_level = self.log_level.upper().strip()
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            self.log_level = "INFO"
        if not 1 <= self.api_port <= 65535:
            self.api_port = 8000
        if self.api_workers < 1:
            self.api_workers = 1

    def __repr__(self) -> str:
        parts = []
        for k, v in sorted(self.to_dict().items()):
            if isinstance(v, Path):
                v = str(v)
            parts.append(f"{k}={v!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_config: Optional[AiosConfig] = None


def get_config(reload: bool = False) -> AiosConfig:
    """Return the global :class:`AiosConfig` singleton.

    Args:
        reload: If ``True``, discard the cached instance and build a new one.

    Returns:
        The active ``AiosConfig`` instance.
    """
    global _config
    if _config is None or reload:
        _config = AiosConfig()
        _config.setup_directories()
    return _config


def configure_logging(cfg: Optional[AiosConfig] = None) -> logging.Logger:
    """Configure the root logger according to *cfg*.

    Args:
        cfg:  A configuration instance.  Falls back to :func:`get_config`.

    Returns:
        The root logger.
    """
    if cfg is None:
        cfg = get_config()

    level = getattr(logging, cfg.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("aios")
    logger.info("AIOS v%s [%s] — %s", cfg.app_version, cfg.environment, cfg.device)
    return logger
