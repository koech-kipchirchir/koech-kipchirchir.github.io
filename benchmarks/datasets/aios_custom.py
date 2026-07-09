"""
Custom AIOS benchmarks: agent task performance, tool-use accuracy,
memory retrieval, workflow execution, and multi-agent coordination.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from benchmarks.base import Benchmark, BenchmarkItem, BenchmarkPrediction, BenchmarkResult

logger = logging.getLogger("aios.benchmarks.aios_custom")


class AIOSAgentBenchmark(Benchmark):
    """End-to-end benchmark for AIOS agent capabilities.

    Evaluates:
    - Tool call accuracy (correct tool + parameters)
    - Task completion success rate
    - Multi-step reasoning accuracy
    - Agent coordination quality
    - Memory recall precision
    """

    def __init__(self) -> None:
        super().__init__(
            name="AIOS-Custom",
            description="AIOS agent capabilities: tool use, task completion, memory, reasoning",
        )

    async def load(self, data_dir: str | Path = "", **kwargs: Any) -> int:
        path = Path(data_dir) if data_dir else Path(__file__).parent.parent / "sample_data"
        filepath = path / "aios_custom_samples.json"

        if filepath.exists():
            data = json.loads(filepath.read_text("utf-8"))
        else:
            data = self._generate_samples(6)

        items: list[BenchmarkItem] = []
        for i, sample in enumerate(data):
            items.append(BenchmarkItem(
                id=f"aios:{i}",
                prompt=sample.get("prompt", ""),
                expected=sample.get("expected", ""),
                choices=sample.get("choices"),
                metadata={
                    "category": sample.get("category", "general"),
                    "expected_tool": sample.get("expected_tool", ""),
                    "expected_parameters": sample.get("expected_parameters", {}),
                    "check_fn": sample.get("check_fn", "exact_match"),
                },
            ))

        self._items = items
        logger.info("AIOS-Custom loaded %d items across categories", len(items))
        return len(items)

    async def evaluate(self, predictions: list[BenchmarkPrediction]) -> BenchmarkResult:
        for pred in predictions:
            item = next((i for i in self._items if i.id == pred.item_id), None)
            if item is None:
                continue
            if pred.error:
                pred.correct = False
                continue
            pred.correct = self._evaluate_prediction(pred, item)

        correct = sum(1 for p in predictions if p.correct)
        n = len(predictions)
        lat_stats = self._compute_latency_stats(predictions)
        total_tokens = sum(p.prompt_tokens + p.completion_tokens for p in predictions)

        return BenchmarkResult(
            name=self.name,
            description=self.description,
            version="1.0",
            total_items=n,
            correct=correct,
            accuracy=correct / max(n, 1),
            total_latency_s=lat_stats["total"],
            avg_latency_s=lat_stats["avg"],
            p50_latency_s=lat_stats["p50"],
            p95_latency_s=lat_stats["p95"],
            p99_latency_s=lat_stats["p99"],
            total_prompt_tokens=sum(p.prompt_tokens for p in predictions),
            total_completion_tokens=sum(p.completion_tokens for p in predictions),
            avg_prompt_tokens=sum(p.prompt_tokens for p in predictions) // max(n, 1),
            avg_completion_tokens=sum(p.completion_tokens for p in predictions) // max(n, 1),
            token_throughput_sec=total_tokens / max(lat_stats["total"], 0.001),
            error_count=sum(1 for p in predictions if p.error),
            predictions=predictions,
        )

    def _evaluate_prediction(self, pred: BenchmarkPrediction, item: BenchmarkItem) -> bool:
        check_fn = item.metadata.get("check_fn", "exact_match")
        expected = item.expected
        output = pred.output.strip()

        if check_fn == "exact_match":
            return str(expected).strip() in output or output in str(expected).strip()

        if check_fn == "tool_call":
            tool = item.metadata.get("expected_tool", "")
            params = item.metadata.get("expected_parameters", {})
            return (tool.lower() in output.lower()
                    and all(str(v).lower() in output.lower() for v in params.values()))

        if check_fn == "contains_all":
            parts = str(expected).split(",") if isinstance(expected, str) else [str(expected)]
            return all(p.strip().lower() in output.lower() for p in parts if p.strip())

        if check_fn == "json_valid":
            import json
            try:
                data = json.loads(output)
                return isinstance(data, dict) and len(data) > 0
            except (json.JSONDecodeError, ValueError):
                return False

        if check_fn == "code_runs":
            try:
                compile(output, "<string>", "exec")
                return True
            except SyntaxError:
                return False

        # Default: check if expected is a substring
        return str(expected).lower() in output.lower()

    def _generate_samples(self, count: int) -> list[dict]:
        samples = [
            {
                "category": "tool_use",
                "prompt": "Call the weather tool to check the temperature in Tokyo.",
                "expected": "get_weather",
                "expected_tool": "get_weather",
                "expected_parameters": {"location": "Tokyo"},
                "check_fn": "tool_call",
            },
            {
                "category": "reasoning",
                "prompt": "If Alice has 3 cats and each cat eats 2 cans of food per day, "
                         "how many cans do they eat in a week?",
                "expected": "42",
                "check_fn": "exact_match",
            },
            {
                "category": "memory",
                "prompt": "Recall what the user said about their favorite programming language "
                         "and summarize it.",
                "expected": "python",
                "check_fn": "contains_all",
            },
            {
                "category": "code_gen",
                "prompt": "Write a Python function that checks if a string is a palindrome.",
                "expected": "def is_palindrome",
                "check_fn": "code_runs",
            },
            {
                "category": "multi_agent",
                "prompt": "Coordinate with the research agent and the coding agent to "
                         "implement a binary search tree in Python.",
                "expected": "class TreeNode,insert,search",
                "check_fn": "contains_all",
            },
            {
                "category": "structured_output",
                "prompt": "Return a JSON object with keys: name, age, city for a person named "
                         "John who is 30 and lives in New York.",
                "expected": '{"name": "John", "age": 30, "city": "New York"}',
                "check_fn": "json_valid",
            },
        ]
        return samples[:count]
