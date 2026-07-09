from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineMetrics:
    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    rate_limit_hits: int = 0
    cache_hits: int = 0
    tool_calls: int = 0
    agent_routes: int = 0
    rag_queries: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def avg_tokens_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens / self.total_requests

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_errors": self.total_errors,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 1),
            "error_rate": round(self.error_rate, 4),
            "rate_limit_hits": self.rate_limit_hits,
            "cache_hits": self.cache_hits,
            "tool_calls": self.tool_calls,
            "agent_routes": self.agent_routes,
            "rag_queries": self.rag_queries,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }

    def __str__(self) -> str:
        d = self.to_dict()
        return (
            f"Metrics(req={d['total_requests']}, tokens={d['total_tokens']}, "
            f"avg_latency={d['avg_latency_ms']}ms, errors={d['total_errors']}, "
            f"uptime={d['uptime_seconds']}s)"
        )


class MetricsCollector:
    def __init__(self) -> None:
        self._metrics = EngineMetrics()
        self._lock = threading.Lock()

    def record_request(
        self,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
    ) -> None:
        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.total_latency_ms += latency_ms
            self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)
            self._metrics.total_prompt_tokens += prompt_tokens
            self._metrics.total_completion_tokens += completion_tokens
            self._metrics.total_tokens += prompt_tokens + completion_tokens
            if not success:
                self._metrics.total_errors += 1

    def record_rate_limit(self) -> None:
        with self._lock:
            self._metrics.rate_limit_hits += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._metrics.cache_hits += 1

    def record_tool_call(self) -> None:
        with self._lock:
            self._metrics.tool_calls += 1

    def record_agent_route(self) -> None:
        with self._lock:
            self._metrics.agent_routes += 1

    def record_rag_query(self) -> None:
        with self._lock:
            self._metrics.rag_queries += 1

    def get_metrics(self) -> EngineMetrics:
        with self._lock:
            return self._metrics

    def reset(self) -> None:
        with self._lock:
            self._metrics = EngineMetrics()
