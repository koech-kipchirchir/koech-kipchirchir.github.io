"""
AIOS Benchmarking Framework
===========================

Production-grade benchmark suite for evaluating LLMs and AIOS agents.

Supported benchmarks:
- **MMLU** (57-subject multiple-choice QA)
- **HumanEval** (code generation / function synthesis)
- **GSM8K** (grade-school math word problems)
- **TruthfulQA** (truthfulness / common misconceptions)
- **ARC** (AI2 Reasoning Challenge — science QA)
- **AIOS-Custom** (agent tool-use, task completion, memory, reasoning)

Report formats: Markdown, HTML, CSV

Metrics tracked: accuracy, latency, memory, GPU usage,
token throughput, cost estimation.
"""

from __future__ import annotations

from benchmarks.base import (
    Benchmark, BenchmarkItem, BenchmarkPrediction, BenchmarkResult,
)
from benchmarks.config import BenchmarkConfig, MODEL_PRICING
from benchmarks.datasets.mmlu import MMLUBenchmark, MMLU_CATEGORIES
from benchmarks.datasets.humaneval import HumanEvalBenchmark
from benchmarks.datasets.gsm8k import GSM8KBenchmark
from benchmarks.datasets.truthfulqa import TruthfulQABenchmark
from benchmarks.datasets.arc import ARCBenchmark
from benchmarks.datasets.aios_custom import AIOSAgentBenchmark
from benchmarks.metrics import ResourceMonitor, estimate_cost, estimate_batch_cost
from benchmarks.models.registry import (
    list_models, list_providers, get_pricing, get_provider,
    get_model_info, register_model, get_models_by_provider,
)
from benchmarks.reporter import ReportGenerator
from benchmarks.runner import BenchmarkRunner, RunReport

__all__ = [
    "ARCBenchmark",
    "AIOSAgentBenchmark",
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkItem",
    "BenchmarkPrediction",
    "BenchmarkResult",
    "BenchmarkRunner",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "MMLUBenchmark",
    "MMLU_CATEGORIES",
    "MODEL_PRICING",
    "ReportGenerator",
    "ResourceMonitor",
    "RunReport",
    "TruthfulQABenchmark",
    "estimate_batch_cost",
    "estimate_cost",
    "get_model_info",
    "get_models_by_provider",
    "get_pricing",
    "get_provider",
    "list_models",
    "list_providers",
    "register_model",
]

__version__ = "0.1.0"
