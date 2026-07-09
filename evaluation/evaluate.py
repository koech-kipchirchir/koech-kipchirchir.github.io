import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training.utils import get_logger, get_gpu_memory

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    accuracy: Optional[float] = None
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    total_tokens: int = 0
    num_samples: int = 0
    num_correct: int = 0
    num_errors: int = 0
    per_sample: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "Evaluation Results:",
            f"  Accuracy:            {self.accuracy:.4f}" if self.accuracy is not None else "  Accuracy:            N/A",
            f"  Avg Latency:         {self.avg_latency_ms:.2f} ms",
            f"  P50 Latency:         {self.p50_latency_ms:.2f} ms",
            f"  P95 Latency:         {self.p95_latency_ms:.2f} ms",
            f"  P99 Latency:         {self.p99_latency_ms:.2f} ms",
            f"  Throughput:          {self.throughput_tokens_per_sec:.1f} tok/s",
            f"  Peak Memory:         {self.peak_memory_mb:.1f} MB",
            f"  Total Tokens:        {self.total_tokens}",
            f"  Samples:             {self.num_samples}",
            f"  Correct:             {self.num_correct}",
            f"  Errors:              {self.num_errors}",
        ]
        return "\n".join(lines)


class ModelEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        prompts: List[str],
        ground_truth: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        batch_size: int = 1,
        correct_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> EvaluationResult:
        peak_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak_memory = get_gpu_memory() or 0

        latencies = []
        tokens_generated = 0
        num_correct = 0
        num_errors = 0
        per_sample: List[Dict[str, Any]] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_truth = ground_truth[i : i + batch_size] if ground_truth else [None] * len(batch_prompts)

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).to(self.device)

            start = time.perf_counter()
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                elapsed = time.perf_counter() - start
            except Exception as e:
                logger.error("Generation error on sample %d: %s", i, e)
                num_errors += len(batch_prompts)
                for j, prompt in enumerate(batch_prompts):
                    per_sample.append({
                        "prompt": prompt,
                        "prediction": None,
                        "ground_truth": batch_truth[j],
                        "correct": False,
                        "error": str(e),
                    })
                continue

            latencies.append(elapsed)

            for j, (prompt, output_ids) in enumerate(zip(batch_prompts, outputs)):
                input_len = inputs["input_ids"].shape[1]
                generated = output_ids[input_len:]
                n_tokens = len(generated)
                tokens_generated += n_tokens

                prediction = self.tokenizer.decode(generated, skip_special_tokens=True)
                gt = batch_truth[j]

                correct = False
                if gt is not None and correct_fn:
                    correct = correct_fn(prediction, gt)
                elif gt is not None:
                    correct = prediction.strip() == gt.strip()

                if correct:
                    num_correct += 1

                per_sample.append({
                    "prompt": prompt,
                    "prediction": prediction,
                    "ground_truth": gt,
                    "correct": correct,
                    "latency_ms": elapsed * 1000,
                    "num_tokens": n_tokens,
                })

        if torch.cuda.is_available():
            peak_memory = (get_gpu_memory() or 0) - peak_memory

        latencies_ms = [l * 1000 for l in latencies]
        latencies_ms.sort()
        n = len(latencies_ms)

        result = EvaluationResult(
            accuracy=num_correct / len(prompts) if len(prompts) > 0 else None,
            avg_latency_ms=sum(latencies_ms) / n if n > 0 else 0,
            p50_latency_ms=latencies_ms[n // 2] if n > 0 else 0,
            p95_latency_ms=latencies_ms[int(n * 0.95)] if n > 0 else 0,
            p99_latency_ms=latencies_ms[int(n * 0.99)] if n > 0 else 0,
            throughput_tokens_per_sec=tokens_generated / sum(latencies) if sum(latencies) > 0 else 0,
            peak_memory_mb=peak_memory,
            total_tokens=tokens_generated,
            num_samples=len(prompts),
            num_correct=num_correct,
            num_errors=num_errors,
            per_sample=per_sample,
        )

        logger.info("\n%s", result)
        return result
