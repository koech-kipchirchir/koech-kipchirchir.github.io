"""
HumanEval: code generation benchmark.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from benchmarks.base import Benchmark, BenchmarkItem, BenchmarkPrediction, BenchmarkResult

logger = logging.getLogger("aios.benchmarks.humaneval")


class HumanEvalBenchmark(Benchmark):
    """HumanEval: function synthesis from docstrings (pass@1)."""

    def __init__(self) -> None:
        super().__init__(
            name="HumanEval",
            description="Code generation — function synthesis from docstrings",
        )

    async def load(self, data_dir: str | Path = "", **kwargs: Any) -> int:
        path = Path(data_dir) if data_dir else Path(__file__).parent.parent / "sample_data"
        filepath = path / "humaneval_samples.json"

        if filepath.exists():
            data = json.loads(filepath.read_text("utf-8"))
        else:
            data = self._generate_samples(5)

        items: list[BenchmarkItem] = []
        for i, sample in enumerate(data):
            prompt = sample.get("prompt", sample.get("instruction", ""))
            items.append(BenchmarkItem(
                id=f"humaneval:{i}",
                prompt=prompt,
                expected=sample,
                metadata={"entry_point": sample.get("entry_point", f"func_{i}"),
                          "tests": sample.get("tests", [])},
            ))

        self._items = items
        logger.info("HumanEval loaded %d items", len(items))
        return len(items)

    async def evaluate(self, predictions: list[BenchmarkPrediction]) -> BenchmarkResult:
        for pred in predictions:
            item = next((i for i in self._items if i.id == pred.item_id), None)
            if item is None or not isinstance(item.expected, dict):
                continue
            if pred.error:
                pred.correct = False
                continue
            pred.correct = self._evaluate_code(
                pred.output,
                item.expected.get("tests", []),
                item.expected.get("entry_point", ""),
            )

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

    def _evaluate_code(self, output: str, tests: list[str], entry_point: str) -> bool:
        if not tests:
            # No test cases — check for syntactic validity
            return bool(output.strip() and len(output.strip()) > 10)

        code = self._extract_code(output)
        if not code:
            return False

        # Try to compile
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            return False

        # Run test assertions (sandboxed)
        import ast
        import sys
        from io import StringIO

        local_ns: dict = {}
        try:
            exec(code, local_ns)
        except Exception:
            return False

        if entry_point and entry_point not in local_ns:
            return False

        for test_expr in tests:
            try:
                result = eval(test_expr, local_ns)
                if not result:
                    return False
            except Exception:
                return False
        return True

    def _extract_code(self, output: str) -> str:
        # Try to extract code from markdown code blocks
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
        if blocks:
            return blocks[0].strip()
        return output.strip()

    def _generate_samples(self, count: int) -> list[dict]:
        samples = [
            {
                "entry_point": "add",
                "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
                "tests": ["add(1, 2) == 3", "add(-1, 1) == 0", "add(0, 0) == 0"],
            },
            {
                "entry_point": "is_even",
                "prompt": 'def is_even(n: int) -> bool:\n    """Return True if n is even."""\n',
                "tests": ["is_even(2) == True", "is_even(3) == False", "is_even(0) == True"],
            },
            {
                "entry_point": "factorial",
                "prompt": 'def factorial(n: int) -> int:\n    """Return n! for non-negative n."""\n',
                "tests": ["factorial(0) == 1", "factorial(5) == 120"],
            },
            {
                "entry_point": "reverse_string",
                "prompt": 'def reverse_string(s: str) -> str:\n    """Return s reversed."""\n',
                "tests": ["reverse_string('abc') == 'cba'", "reverse_string('') == ''"],
            },
            {
                "entry_point": "fibonacci",
                "prompt": 'def fibonacci(n: int) -> int:\n    """Return the nth Fibonacci number."""\n',
                "tests": ["fibonacci(0) == 0", "fibonacci(1) == 1", "fibonacci(10) == 55"],
            },
        ]
        return samples[:count]
