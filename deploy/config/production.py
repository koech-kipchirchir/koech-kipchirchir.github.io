"""
Production-specific deployment configuration overrides.
"""

from __future__ import annotations

from deploy.config.base import DeploymentConfig


def production_config() -> DeploymentConfig:
    """Return a DeploymentConfig tuned for production."""
    cfg = DeploymentConfig.from_env()

    # Sensible production defaults
    cfg.reload = False
    cfg.workers = max(cfg.workers, 2)

    # Production must have a proper log level
    if cfg.log_level not in ("debug", "info", "warning", "error", "critical"):
        cfg.log_level = "info"

    # Require an API key for production
    if not cfg.api_key and cfg.provider in ("openai", "anthropic"):
        import warnings
        warnings.warn("AIOS_API_KEY is not set — authentication may fail")

    return cfg
