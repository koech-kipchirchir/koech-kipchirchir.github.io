"""
ARC (AI2 Reasoning Challenge) benchmark.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from benchmarks.base import Benchmark, BenchmarkItem, BenchmarkPrediction, BenchmarkResult

logger = logging.getLogger("aios.benchmarks.arc")


class ARCBenchmark(Benchmark):
    """ARC (AI2 Reasoning Challenge): science multiple-choice QA.

    Supports both ARC-Challenge (hard) and ARC-Easy (easy) subsets.
    """

    LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

    def __init__(self, challenge_only: bool = True) -> None:
        super().__init__(
            name="ARC" if challenge_only else "ARC-Easy",
            description="AI2 Reasoning Challenge — science QA",
        )
        self._challenge_only = challenge_only

    async def load(self, data_dir: str | Path = "", **kwargs: Any) -> int:
        path = Path(data_dir) if data_dir else Path(__file__).parent.parent / "sample_data"
        subset = "challenge" if self._challenge_only else "easy"
        filepath = path / f"arc_{subset}.json"

        if filepath.exists():
            data = json.loads(filepath.read_text("utf-8"))
        else:
            data = self._generate_samples(10)

        items: list[BenchmarkItem] = []
        for i, record in enumerate(data):
            question = record.get("question", "")
            choices = record.get("choices", ["", "", "", ""])
            answer_key = record.get("answerKey", record.get("answer", ""))
            prompt = self._format_prompt(question, choices)
            items.append(BenchmarkItem(
                id=f"arc:{i}",
                prompt=prompt,
                expected=answer_key,
                choices=choices,
                metadata={"subset": subset, "question": question},
            ))

        self._items = items
        logger.info("ARC-%s loaded %d items", subset, len(items))
        return len(items)

    async def evaluate(self, predictions: list[BenchmarkPrediction]) -> BenchmarkResult:
        for pred in predictions:
            item = next((i for i in self._items if i.id == pred.item_id), None)
            if item is None:
                continue
            if pred.error:
                pred.correct = False
                continue
            pred.correct = self._check_answer(pred.output, item.expected)

        correct = sum(1 for p in predictions if p.correct)
        n = len(predictions)
        lat_stats = self._compute_latency_stats(predictions)
        total_tokens = sum(p.prompt_tokens + p.completion_tokens for p in predictions)

        by_subject: dict[str, int] = {}
        by_subject_total: dict[str, int] = {}

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

    def _format_prompt(self, question: str, choices: list[str]) -> str:
        labels = ["A", "B", "C", "D"]
        lines = [question, ""]
        for label, choice in zip(labels, choices):
            if choice:
                lines.append(f"{label}. {choice}")
        lines.append("")
        lines.append("Answer the correct letter (A, B, C, or D):")
        return "\n".join(lines)

    def _check_answer(self, output: str, expected: str) -> bool:
        output = output.strip().upper()
        expected = expected.strip().upper()
        if output == expected:
            return True
        for prefix in ["ANSWER:", "THE ANSWER IS", "THE CORRECT ANSWER IS"]:
            if prefix in output:
                parts = output.split(prefix)
                if len(parts) > 1:
                    candidate = parts[-1].strip().rstrip(".").strip()
                    if candidate == expected:
                        return True
        if len(output) == 1 and output in "ABCD":
            return output == expected
        return False

    def _generate_samples(self, count: int) -> list[dict]:
        samples: list[dict] = []
        questions = [
            "Which of the following is a renewable energy source?",
            "What is the chemical symbol for gold?",
            "Which planet is known as the Red Planet?",
            "What force keeps planets in orbit around the sun?",
            "What is the smallest unit of life?",
            "Which gas makes up the majority of Earth's atmosphere?",
            "What is the process by which plants make their own food?",
            "Which type of rock is formed from cooled magma?",
            "What is the speed of light approximately?",
            "Which of the following is not a state of matter?",
        ]
        for i, q in enumerate(questions[:count]):
            samples.append({
                "id": i,
                "question": q,
                "choices": [
                    f"Choice A for Q{i}",
                    f"Choice B for Q{i}",
                    f"Choice C for Q{i}",
                    f"Choice D for Q{i}",
                ],
                "answerKey": random.choice(["A", "B", "C", "D"]),
            })
        return samples
