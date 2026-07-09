from __future__ import annotations

import logging
from typing import Annotated, Any, AsyncIterator

from fastapi import Request

from api.config import ApiConfig
from api.exceptions import ServiceUnavailableError

logger = logging.getLogger("aios.api.dependencies")


async def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def get_config(request: Request) -> ApiConfig:
    config: ApiConfig | None = getattr(request.app.state, "config", None)
    if config is None:
        raise ServiceUnavailableError(detail="API configuration not loaded")
    return config


async def get_engine(request: Request) -> Any:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise ServiceUnavailableError(detail="AI engine not initialized")
    return engine


async def get_token_manager(request: Request) -> Any:
    tm = getattr(request.app.state, "token_manager", None)
    if tm is None:
        tm = getattr(request.app.state, "token_counter", None)
    if tm is None:
        raise ServiceUnavailableError(detail="Token manager not available")
    return tm


async def get_memory_manager(request: Request) -> Any:
    mm = getattr(request.app.state, "memory_manager", None)
    if mm is None:
        raise ServiceUnavailableError(detail="Memory manager not initialized")
    return mm


async def get_readiness(request: Request) -> dict[str, Any]:
    ready = getattr(request.app.state, "ready", False)
    return {"ready": ready}
