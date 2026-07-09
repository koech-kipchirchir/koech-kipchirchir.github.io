"""
Prometheus metrics integration for AIOS FastAPI application.

Usage in api/app.py::

    from deploy.monitoring.prometheus_middleware import setup_prometheus_metrics
    setup_prometheus_metrics(app)
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest,
    CONTENT_TYPE_LATEST, REGISTRY,
)

# ---- Metrics ----

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

HTTP_REQUESTS_ACTIVE = Gauge(
    "http_requests_active",
    "Number of active HTTP requests",
)

AIOS_TOKENS_TOTAL = Counter(
    "aios_tokens_total",
    "Total tokens processed",
    ["type"],  # prompt, completion
)

AIOS_MEMORY_ITEMS = Gauge(
    "aios_memory_items",
    "Number of items in memory store",
)

AIOS_MODEL_LOADED = Gauge(
    "aios_model_loaded",
    "Whether a model is currently loaded (1=yes, 0=no)",
)

AIOS_GPU_MEMORY_USED_BYTES = Gauge(
    "aios_gpu_memory_used_bytes",
    "GPU memory used in bytes",
)

AIOS_GPU_MEMORY_TOTAL_BYTES = Gauge(
    "aios_gpu_memory_total_bytes",
    "Total GPU memory in bytes",
)


class PrometheusMiddleware:
    """FastAPI middleware that records request metrics."""

    async def __call__(self, request: Request, call_next: Any) -> Response:
        method = request.method
        path = request.url.path

        HTTP_REQUESTS_ACTIVE.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception as exc:
            status = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
            HTTP_REQUESTS_ACTIVE.dec()


def setup_prometheus_metrics(app: FastAPI) -> None:
    """Add Prometheus metrics endpoint and middleware to a FastAPI app."""

    # Add middleware (before other middleware if possible)
    app.add_middleware(PrometheusMiddleware)

    # Add /metrics endpoint
    @app.get("/metrics", include_in_schema=False, tags=["monitoring"])
    async def metrics() -> Response:
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Add /health endpoint for Prometheus scraping health
    @app.get("/health", include_in_schema=False, tags=["monitoring"])
    async def health() -> dict:
        return {"status": "ok", "service": "aios-api"}


# ---- Helper functions for other parts of the app ----

def record_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    if prompt_tokens > 0:
        AIOS_TOKENS_TOTAL.labels(type="prompt").inc(prompt_tokens)
    if completion_tokens > 0:
        AIOS_TOKENS_TOTAL.labels(type="completion").inc(completion_tokens)


def set_memory_items(count: int) -> None:
    AIOS_MEMORY_ITEMS.set(count)


def set_model_loaded(loaded: bool) -> None:
    AIOS_MODEL_LOADED.set(1 if loaded else 0)


def set_gpu_memory(used_bytes: float, total_bytes: float) -> None:
    AIOS_GPU_MEMORY_USED_BYTES.set(used_bytes)
    AIOS_GPU_MEMORY_TOTAL_BYTES.set(total_bytes)
