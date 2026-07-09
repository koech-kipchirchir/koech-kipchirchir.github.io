"""
GSM8K: grade-school math word problems benchmark.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from benchmarks.base import Benchmark, BenchmarkItem, BenchmarkPrediction, BenchmarkResult

logger = logging.getLogger("aios.benchmarks.gsm8k")


class GSM8KBenchmark(Benchmark):
    """GSM8K: grade-school math word problems (chain-of-thought)."""

    def __init__(self) -> None:
        super().__init__(
            name="GSM8K",
            description="Grade-school math word problems",
        )

    async def load(self, data_dir: str | Path = "", **kwargs: Any) -> int:
        path = Path(data_dir) if data_dir else Path(__file__).parent.parent / "sample_data"
        filepath = path / "gsm8k_samples.json"

        if filepath.exists():
            data = json.loads(filepath.read_text("utf-8"))
        else:
            data = self._generate_samples(5)

        items: list[BenchmarkItem] = []
        for i, sample in enumerate(data):
            prompt = self._format_prompt(sample.get("question", ""))
            items.append(BenchmarkItem(
                id=f"gsm8k:{i}",
                prompt=prompt,
                expected=sample.get("answer", ""),
                metadata={"question": sample.get("question", ""),
                          "answer_number": self._extract_answer_number(sample.get("answer", ""))},
            ))

        self._items = items
        logger.info("GSM8K loaded %d items", len(items))
        return len(items)

    async def evaluate(self, predictions: list[BenchmarkPrediction]) -> BenchmarkResult:
        for pred in predictions:
            item = next((i for i in self._items if i.id == pred.item_id), None)
            if item is None:
                continue
            if pred.error:
                pred.correct = False
                continue
            expected_num = item.metadata.get("answer_number")
            predicted_num = self._extract_answer_number(pred.output)
            pred.correct = (predicted_num is not None and expected_num is not None
                            and abs(predicted_num - expected_num) < 0.01)

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

    def _format_prompt(self, question: str) -> str:
        return (
            f"Solve the following math problem step by step.\n"
            f"After reasoning, output the final answer as: #### <number>\n\n"
            f"{question}\n"
        )

    def _extract_answer_number(self, text: str) -> float | None:
        if not text:
            return None
        # Look for #### <number> pattern
        m = re.search(r"####\s*(-?\d+\.?\d*)", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        # Fallback: last number in the text
        nums = re.findall(r"-?\d+\.?\d*", text)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass
        return None

    def _generate_samples(self, count: int) -> list[dict]:
        samples = [
            {"question": "Janet has 3 apples. She buys 5 more. How many apples does she have?",
             "answer": "#### 8"},
            {"question": "A train travels 120 miles in 2 hours. What is its average speed?",
             "answer": "#### 60"},
            {"question": "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
             "answer": "#### 20"},
            {"question": "The sum of two numbers is 15. One number is 7. What is the other?",
             "answer": "#### 8"},
            {"question": "A recipe calls for 2 cups of flour for 12 cookies. How much flour for 30 cookies?",
             "answer": "#### 5"},
        ]
        return samples[:count]
