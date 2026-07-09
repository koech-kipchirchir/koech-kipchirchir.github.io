"""
Base classes for all benchmarks.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("aios.benchmarks")


@dataclass
class BenchmarkItem:
    """A single test item within a benchmark."""

    id: str = ""
    prompt: str = ""
    expected: Any = None
    choices: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkPrediction:
    """A model's prediction for a single item."""

    item_id: str = ""
    output: str = ""
    correct: bool | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    confidence: float | None = None
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated result of running a benchmark."""

    name: str = ""
    description: str = ""
    version: str = "1.0"
    total_items: int = 0
    correct: int = 0
    accuracy: float = 0.0
    total_latency_s: float = 0.0
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_prompt_tokens: int = 0
    avg_completion_tokens: int = 0
    token_throughput_sec: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_util_pct: float = 0.0
    estimated_cost_usd: float = 0.0
    error_count: int = 0
    predictions: list[BenchmarkPrediction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.total_items, 1)


class Benchmark(ABC):
    """Abstract base for all benchmarks."""

    def __init__(self, name: str = "", description: str = "", version: str = "1.0") -> None:
        self.name = name or self.__class__.__name__
        self.description = description
        self._version = version
        self._items: list[BenchmarkItem] = []

    @property
    def items(self) -> list[BenchmarkItem]:
        return list(self._items)

    @abstractmethod
    async def load(self, **kwargs: Any) -> int:
        """Load the dataset. Returns item count."""
        pass

    @abstractmethod
    async def evaluate(
        self,
        predictions: list[BenchmarkPrediction],
    ) -> BenchmarkResult:
        """Score predictions and produce a BenchmarkResult."""
        pass

    async def run(
        self,
        model_fn: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Convenience: load, infer, evaluate in one call."""
        count = await self.load(**kwargs)
        logger.info("Running %s on %d items", self.name, count)
        predictions = await self._infer_all(model_fn)
        return await self.evaluate(predictions)

    async def _infer_all(self, model_fn: Any) -> list[BenchmarkPrediction]:
        results: list[BenchmarkPrediction] = []
        for item in self._items:
            pred = await self._infer_one(item, model_fn)
            results.append(pred)
        return results

    async def _infer_one(self, item: BenchmarkItem, model_fn: Any) -> BenchmarkPrediction:
        start = time.perf_counter()
        prompt_tokens = 0
        completion_tokens = 0
        output = ""
        error = None

        try:
            response = await model_fn(item.prompt, choices=item.choices)
            if isinstance(response, dict):
                output = response.get("text", str(response))
                prompt_tokens = response.get("prompt_tokens", 0)
                completion_tokens = response.get("completion_tokens", 0)
            else:
                output = str(response)
        except Exception as e:
            error = str(e)
            logger.warning("Inference failed for item %s: %s", item.id, error)

        elapsed = time.perf_counter() - start
        return BenchmarkPrediction(
            item_id=item.id,
            output=output,
            latency_s=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
        )

    def _accuracy(self, predictions: list[BenchmarkPrediction]) -> float:
        correct = sum(1 for p in predictions if p.correct)
        return correct / max(len(predictions), 1)

    def _compute_latency_stats(self, predictions: list[BenchmarkPrediction]) -> dict[str, float]:
        latencies = sorted([p.latency_s for p in predictions if p.error is None])
        if not latencies:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "total": 0.0}
        n = len(latencies)
        return {
            "avg": sum(latencies) / n,
            "p50": latencies[n // 2] if n > 0 else 0.0,
            "p95": latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": latencies[int(n * 0.99)] if n > 0 else 0.0,
            "total": sum(latencies),
        }
