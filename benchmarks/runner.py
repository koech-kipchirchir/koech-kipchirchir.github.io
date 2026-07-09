"""
Benchmark runner: orchestrates loading, inference, evaluation,
resource monitoring, and report generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from benchmarks.base import Benchmark, BenchmarkResult
from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import ResourceMonitor, estimate_batch_cost
from benchmarks.reporter import ReportGenerator

logger = logging.getLogger("aios.benchmarks.runner")


@dataclass
class RunReport:
    """Complete report of a benchmark run session."""

    model_name: str = ""
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    results: list[BenchmarkResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    duration_s: float = 0.0
    output_paths: dict[str, str] = field(default_factory=dict)


class BenchmarkRunner:
    """Orchestrates benchmark execution.

    Usage::

        runner = BenchmarkRunner(config)
        report = await runner.run(
            model_fn=my_inference_function,
            benchmarks=[MMLUBenchmark(), GSM8KBenchmark()],
        )
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self._config = config or BenchmarkConfig()
        self._reporter = ReportGenerator(self._config.output_dir)

    @property
    def config(self) -> BenchmarkConfig:
        return self._config

    async def run(
        self,
        model_fn: Callable,
        benchmarks: list[Benchmark],
        model_name: str = "",
    ) -> RunReport:
        """Run a list of benchmarks with a model function.

        The model_fn should accept ``(prompt: str, **kwargs) -> str | dict``.
        If returning a dict, use keys ``text``, ``prompt_tokens``,
        ``completion_tokens``.
        """
        start = datetime.now(timezone.utc)
        start_ts = time.perf_counter()
        results: list[BenchmarkResult] = []

        logger.info("Starting benchmark run for model=%s with %d benchmark(s)",
                     model_name or "custom", len(benchmarks))

        for bench in benchmarks:
            result = await self._run_single(bench, model_fn, model_name)
            results.append(result)

        total_duration = time.perf_counter() - start_ts
        report = RunReport(
            model_name=model_name,
            config=self._config,
            results=results,
            start_time=start.isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            duration_s=round(total_duration, 2),
        )

        # Generate reports
        formats = self._config.formats
        paths = self._reporter.generate(results, model_name=model_name, formats=formats)
        report.output_paths = {fmt: str(p) for fmt, p in paths.items()}

        self._log_summary(report)
        return report

    async def _run_single(
        self,
        bench: Benchmark,
        model_fn: Callable,
        model_name: str,
    ) -> BenchmarkResult:
        logger.info("Running %s...", bench.name)

        # Load with optional sample limit
        kwargs = {}
        if self._config.sample_limit > 0:
            kwargs["sample_limit"] = self._config.sample_limit

        try:
            n = await bench.load(**kwargs)
        except Exception as e:
            logger.error("Failed to load %s: %s", bench.name, e)
            return BenchmarkResult(name=bench.name, error_count=1, metadata={"load_error": str(e)})

        if n == 0:
            logger.warning("No items loaded for %s", bench.name)
            return BenchmarkResult(name=bench.name)

        # Run inference with resource monitoring
        monitor = ResourceMonitor(
            interval_s=self._config.gpu_sample_interval,
            track_gpu=self._config.track_gpu,
        )
        monitor.start()

        predictions = []
        for item in bench.items:
            pred = await bench._infer_one(item, model_fn)
            predictions.append(pred)

        monitor.stop()

        # Evaluate
        result = await bench.evaluate(predictions)

        # Attach resource metrics
        result.peak_memory_mb = monitor.peak_memory_mb
        result.avg_memory_mb = monitor.avg_memory_mb
        result.peak_gpu_memory_mb = monitor.peak_gpu_memory_mb
        result.avg_gpu_util_pct = monitor.avg_gpu_util_pct

        # Estimate cost
        result.estimated_cost_usd = estimate_batch_cost(
            predictions,
            model_name=model_name,
            input_rate=self._config.cost_per_input_token,
            output_rate=self._config.cost_per_output_token,
        )

        logger.info(
            "%s: accuracy=%.2f%%, avg_latency=%.3fs, cost=$%.4f",
            bench.name,
            result.accuracy * 100,
            result.avg_latency_s,
            result.estimated_cost_usd,
        )
        return result

    def _log_summary(self, report: RunReport) -> None:
        logger.info("=" * 60)
        logger.info("Benchmark Run Complete")
        logger.info("  Model:       %s", report.model_name or "custom")
        logger.info("  Benchmarks:  %d", len(report.results))
        logger.info("  Duration:    %.1fs", report.duration_s)
        for r in report.results:
            logger.info("  %-15s acc=%.2f%% lat=%.3fs cost=$%.4f",
                        r.name, r.accuracy * 100, r.avg_latency_s, r.estimated_cost_usd)
        for fmt, path in report.output_paths.items():
            logger.info("  %-5s report: %s", fmt.upper(), path)
        logger.info("=" * 60)
