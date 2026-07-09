from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import get_config, get_engine, get_readiness
from api.config import ApiConfig

logger = logging.getLogger("aios.api.health")

router = APIRouter(tags=["Health"])


@dataclass
class HealthStatus:
    status: str = "ok"
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    engine_ready: bool = False
    model: str = ""
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp,
            "engine": {
                "ready": self.engine_ready,
                "model": self.model,
                "provider": self.provider,
            },
        }


@router.get("/health", summary="Health check")
async def health(
    config: ApiConfig = Depends(get_config),
    engine: Any = Depends(get_engine),
) -> JSONResponse:
    start = time.time()
    uptime = time.time() - getattr(engine, "_start_time", start)
    status = HealthStatus(
        version=config.api_version,
        uptime_seconds=uptime,
        engine_ready=True,
        model=getattr(engine, "config", None).model if getattr(engine, "config", None) else config.model,
        provider=getattr(engine, "config", None).provider if getattr(engine, "config", None) else config.provider,
    )
    return JSONResponse(content=status.to_dict())


@router.get("/ready", summary="Readiness check")
async def readiness(
    ready: dict[str, Any] = Depends(get_readiness),
) -> JSONResponse:
    if ready.get("ready"):
        return JSONResponse(content={"status": "ready", "ready": True})
    return JSONResponse(
        status_code=503,
        content={"status": "not ready", "ready": False},
    )


@router.get("/live", summary="Liveness check")
async def liveness() -> JSONResponse:
    return JSONResponse(content={"status": "alive"})
