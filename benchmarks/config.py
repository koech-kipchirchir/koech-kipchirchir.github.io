"""
Benchmark system configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""

    output_dir: str | Path = "benchmark_reports"
    sample_limit: int = 0           # 0 = full dataset
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 120
    retry_attempts: int = 2
    parallel_tasks: int = 1
    seed: int = 42

    # Per-benchmark subsets (empty = all)
    mmlu_subjects: list[str] = field(default_factory=list)
    arc_challenge_only: bool = True

    # Resource tracking
    track_memory: bool = True
    track_gpu: bool = True
    gpu_sample_interval: float = 0.5

    # Cost estimation (USD per 1K tokens)
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0

    # Reports to generate
    formats: list[str] = field(default_factory=lambda: ["md", "html", "csv"])

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


# Well-known model pricing (USD per 1K tokens) — populated by model registry
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":              {"input": 0.01,   "output": 0.03},
    "gpt-4o-mini":         {"input": 0.0015, "output": 0.006},
    "gpt-4-turbo":         {"input": 0.01,   "output": 0.03},
    "gpt-4":               {"input": 0.03,   "output": 0.06},
    "gpt-3.5-turbo":       {"input": 0.001,  "output": 0.002},
    "claude-3-opus":       {"input": 0.015,  "output": 0.075},
    "claude-3-sonnet":     {"input": 0.003,  "output": 0.015},
    "claude-3-haiku":      {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet":   {"input": 0.003,  "output": 0.015},
    "llama-3-8b":          {"input": 0.0002, "output": 0.0002},
    "llama-3-70b":         {"input": 0.001,  "output": 0.001},
    "mixtral-8x7b":        {"input": 0.0005, "output": 0.0005},
    "gemini-1.5-pro":      {"input": 0.0035, "output": 0.0105},
    "gemini-1.5-flash":    {"input": 0.0005, "output": 0.0015},
    "deepseek-v2":         {"input": 0.0005, "output": 0.0015},
    "command-r-plus":      {"input": 0.003,  "output": 0.015},
}
