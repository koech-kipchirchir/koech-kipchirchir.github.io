"""
Resource and performance metrics collection during benchmark runs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from benchmarks.config import MODEL_PRICING

logger = logging.getLogger("aios.benchmarks.metrics")


@dataclass
class ResourceSample:
    """A single resource usage snapshot."""

    timestamp: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_util_pct: float = 0.0


class ResourceMonitor:
    """Asynchronously samples system resources during a benchmark run.

    Tracks memory (RSS) and optionally GPU memory/utilization
    by polling at a configurable interval.
    """

    def __init__(self, interval_s: float = 0.5, track_gpu: bool = True) -> None:
        self._interval = interval_s
        self._track_gpu = track_gpu
        self._samples: list[ResourceSample] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling in a background thread."""
        if self._running:
            return
        self._running = True
        self._samples.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.debug("Resource monitor started (interval=%.2fs)", self._interval)

    def stop(self) -> None:
        """Stop sampling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        logger.debug("Resource monitor stopped (%d samples)", len(self._samples))

    @property
    def samples(self) -> list[ResourceSample]:
        return list(self._samples)

    @property
    def peak_memory_mb(self) -> float:
        return max((s.memory_mb for s in self._samples), default=0.0)

    @property
    def avg_memory_mb(self) -> float:
        vals = [s.memory_mb for s in self._samples]
        return sum(vals) / max(len(vals), 1)

    @property
    def peak_gpu_memory_mb(self) -> float:
        return max((s.gpu_memory_mb for s in self._samples if s.gpu_memory_mb > 0), default=0.0)

    @property
    def avg_gpu_util_pct(self) -> float:
        vals = [s.gpu_util_pct for s in self._samples if s.gpu_util_pct > 0]
        return sum(vals) / max(len(vals), 1)

    def _sample_loop(self) -> None:
        while self._running:
            sample = ResourceSample(timestamp=time.time())
            try:
                sample.memory_mb = self._get_memory_mb()
            except Exception:
                pass
            if self._track_gpu:
                try:
                    gpu = self._get_gpu_stats()
                    sample.gpu_memory_mb = gpu.get("memory_mb", 0.0)
                    sample.gpu_util_pct = gpu.get("util_pct", 0.0)
                except Exception:
                    pass
            self._samples.append(sample)
            time.sleep(self._interval)

    def _get_memory_mb(self) -> float:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    def _get_gpu_stats(self) -> dict[str, float]:
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "memory_mb": float(parts[0]) if len(parts) > 0 else 0.0,
                    "util_pct": float(parts[1]) if len(parts) > 1 else 0.0,
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return {"memory_mb": 0.0, "util_pct": 0.0}


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str = "",
    input_rate: float = 0.0,
    output_rate: float = 0.0,
) -> float:
    """Estimate USD cost for a given token usage.

    Uses MODEL_PRICING if model_name is known, otherwise falls
    back to the provided rates.
    """
    if model_name and model_name in MODEL_PRICING:
        pricing = MODEL_PRICING[model_name]
        input_rate = pricing["input"]
        output_rate = pricing["output"]
    input_cost = (prompt_tokens / 1000) * input_rate
    output_cost = (completion_tokens / 1000) * output_rate
    return round(input_cost + output_cost, 6)


def estimate_batch_cost(
    predictions: list,
    model_name: str = "",
    input_rate: float = 0.0,
    output_rate: float = 0.0,
) -> float:
    """Estimate total cost for a list of BenchmarkPredictions."""
    total = 0.0
    for p in predictions:
        total += estimate_cost(
            p.prompt_tokens,
            p.completion_tokens,
            model_name=model_name,
            input_rate=input_rate,
            output_rate=output_rate,
        )
    return round(total, 6)


# ---------------------------------------------------------------------------
# Token throughput
# ---------------------------------------------------------------------------

def token_throughput(
    total_tokens: int,
    total_time_s: float,
) -> float:
    """Tokens per second."""
    return total_tokens / max(total_time_s, 0.001)
