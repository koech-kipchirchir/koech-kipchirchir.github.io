"""
MMLU (Massive Multitask Language Understanding) benchmark.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from benchmarks.base import Benchmark, BenchmarkItem, BenchmarkPrediction, BenchmarkResult

logger = logging.getLogger("aios.benchmarks.mmlu")

MMLU_CATEGORIES = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


class MMLUBenchmark(Benchmark):
    """MMLU: 57-subject multiple-choice QA benchmark."""

    LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

    def __init__(self, subjects: list[str] | None = None) -> None:
        super().__init__(
            name="MMLU",
            description="Massive Multitask Language Understanding — 57 subjects",
        )
        self._subjects = subjects or []

    async def load(self, data_dir: str | Path = "", **kwargs: Any) -> int:
        path = Path(data_dir) if data_dir else Path(__file__).parent.parent / "sample_data"
        items: list[BenchmarkItem] = []

        subjects = self._subjects or MMLU_CATEGORIES
        for subj in subjects:
            filepath = path / f"mmlu_{subj}.json"
            if filepath.exists():
                data = json.loads(filepath.read_text("utf-8"))
            else:
                data = self._generate_sample(subj, 5)

            for record in data:
                choices = record.get("choices", ["", "", "", ""])
                prompt = self._format_prompt(record.get("question", ""), choices)
                items.append(BenchmarkItem(
                    id=f"{subj}:{record.get('id', len(items))}",
                    prompt=prompt,
                    expected=record.get("answer", ""),
                    choices=choices,
                    metadata={"subject": subj, "question": record.get("question", "")},
                ))

        self._items = items
        logger.info("MMLU loaded %d items across %d subjects", len(items), len(subjects))
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

    def _check_answer(self, output: str, expected: Any) -> bool:
        output = output.strip().upper()
        expected_str = str(expected).strip().upper()
        if output == expected_str:
            return True
        for prefix in ["ANSWER:", "ANSWER", "THE ANSWER IS", "THE CORRECT ANSWER IS"]:
            if prefix in output:
                parts = output.split(prefix)
                if len(parts) > 1:
                    candidate = parts[-1].strip().rstrip(".").strip()
                    if candidate == expected_str:
                        return True
        if len(output) == 1 and output in "ABCD":
            return output == expected_str
        label = self.LABEL_MAP.get(expected_str)
        if label is not None and expected_str in output:
            return True
        return False

    def _generate_sample(self, subject: str, count: int) -> list[dict]:
        import random
        samples: list[dict] = []
        for i in range(count):
            samples.append({
                "id": i,
                "question": f"Sample {subject} question #{i}?",
                "choices": [f"Option A for #{i}", f"Option B for #{i}",
                            f"Option C for #{i}", f"Option D for #{i}"],
                "answer": random.choice(["A", "B", "C", "D"]),
            })
        return samples
