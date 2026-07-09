from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import ApiConfig
from api.exceptions import RateLimitError as APIRateLimitError

logger = logging.getLogger("aios.api.middleware")


def configure_cors(app: FastAPI, config: ApiConfig) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allow_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=config.cors_allow_methods,
        allow_headers=config.cors_allow_headers,
    )
    logger.info(
        "CORS configured (origins=%s)", config.cors_allow_origins
    )


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        header_name: str = "X-Request-ID",
    ) -> None:
        super().__init__(app)
        self._header_name = header_name

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(self._header_name) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[self._header_name] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        request_id = getattr(request.state, "request_id", "-")
        logger.info(
            "%s %s -> %s (%.1fms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
        return response


class RateLimiter:
    def __init__(self, config: ApiConfig) -> None:
        self._enabled = config.rate_limit_enabled
        self._max_requests = config.rate_limit_requests
        self._window = config.rate_limit_window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._logger = logging.getLogger("aios.api.rate_limiter")

    async def check(self, request: Request) -> None:
        if not self._enabled:
            return
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self._window
        bucket = self._buckets[client_ip]
        bucket[:] = [t for t in bucket if t > window_start]
        if len(bucket) >= self._max_requests:
            oldest = bucket[0] if bucket else now
            retry_after = int(self._window - (now - oldest))
            self._logger.warning(
                "Rate limit exceeded for %s (%d/%d)",
                client_ip, len(bucket), self._max_requests,
            )
            raise APIRateLimitError(retry_after=max(1, retry_after))
        bucket.append(now)

    async def cleanup(self) -> None:
        now = time.time()
        window_start = now - self._window * 2
        expired = [
            ip for ip, times in self._buckets.items()
            if all(t < window_start for t in times)
        ]
        for ip in expired:
            del self._buckets[ip]


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, config: ApiConfig) -> None:
        super().__init__(app)
        self._limiter = RateLimiter(config)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        await self._limiter.check(request)
        return await call_next(request)


def configure_middleware(app: FastAPI, config: ApiConfig) -> None:
    configure_cors(app, config)
    app.add_middleware(RequestIDMiddleware, header_name=config.request_id_header)
    app.add_middleware(RequestLoggingMiddleware)
    if config.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware, config=config)
    logger.info(
        "Middleware configured (rate_limit=%s)", config.rate_limit_enabled
    )
