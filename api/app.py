from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from api.config import ApiConfig
from api.exceptions import APIError
from api.chat import router as chat_router
from api.health import router as health_router
from api.memory import router as memory_router
from api.model import router as model_router
from api.rag import router as rag_router
from api.tools import router as tools_router
from api.middleware import configure_middleware
from api.startup import startup_event
from api.shutdown import shutdown_event

logger = logging.getLogger("aios.api")


def create_app(config: ApiConfig | None = None) -> FastAPI:
    if config is None:
        config = ApiConfig.from_env()

    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description=config.api_description,
        docs_url=config.api_docs_url,
        redoc_url=config.api_redoc_url,
        openapi_url=config.api_openapi_url,
        lifespan=_lifespan,
    )

    app.state.config = config

    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS + request ID + logging + rate limit
    configure_middleware(app, config)

    # Routers
    app.include_router(chat_router, prefix=config.api_prefix)
    app.include_router(health_router, prefix=config.api_prefix)
    app.include_router(model_router, prefix=config.api_prefix)
    app.include_router(memory_router, prefix=config.api_prefix)
    app.include_router(rag_router, prefix=config.api_prefix)
    app.include_router(tools_router, prefix=config.api_prefix)

    # Global exception handler
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "status": 500,
                }
            },
        )

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        return {
            "name": config.api_title,
            "version": config.api_version,
            "docs": config.api_docs_url,
            "openapi": config.api_openapi_url,
        }

    return app


async def _lifespan(app: FastAPI):
    config: ApiConfig = getattr(app.state, "config", ApiConfig.from_env())
    await startup_event(app, config)
    yield
    await shutdown_event(app)
