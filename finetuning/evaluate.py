"""
AIOS Fine-tuning Evaluation

Evaluates a trained LoRA adapter (or merged model) on a held-out validation
set and reports loss, perplexity, and optional generation samples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset as HFDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from finetuning.config import MODEL_REGISTRY, list_supported_models  # noqa: E402
from finetuning.dataset import load_chat_dataset  # noqa: E402
from finetuning.model import load_for_inference  # noqa: E402
from finetuning.utils import setup_logger  # noqa: E402

logger = setup_logger("aios.finetuning.evaluate")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_perplexity(model, tokenizer, dataset: HFDataset) -> float:
    """Compute the perplexity of *model* on *dataset*.

    Args:
        model:     The model to evaluate.
        tokenizer: Corresponding tokenizer.
        dataset:   HuggingFace Dataset with a ``"text"`` column.

    Returns:
        Perplexity score (lower is better).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length or 2048,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            n_tokens = (labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    return perplexity


def generate_samples(
    model,
    tokenizer,
    prompts: List[str],
    max_new: int = 256,
    temperature: float = 0.7,
) -> List[str]:
    """Generate responses for a list of prompts.

    Args:
        model:      The model to use for generation.
        tokenizer:  Corresponding tokenizer.
        prompts:    List of input strings.
        max_new:    Maximum number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        List of generated responses.
    """
    model.eval()
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AIOS Fine-tuning — Evaluate a trained model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter directory or merged model",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="qwen3-8b",
        choices=list_supported_models(),
        help="Model architecture key",
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        default="datasets/valid.jsonl",
        help="Validation dataset path",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=2048,
        help="Maximum sequence length for evaluation",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        default=True,
        help="Compute perplexity on the validation set",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of generation samples to show",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write evaluation results as JSON",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    model_info = MODEL_REGISTRY[args.model_key]

    model, tokenizer = load_for_inference(
        model_path=args.model_path,
        max_seq_length=args.max_seq,
        load_in_4bit=False,
    )

    results: Dict[str, Any] = {
        "model_path": args.model_path,
        "model_key": args.model_key,
        "max_seq": args.max_seq,
    }

    # -- Perplexity ------------------------------------------------------
    if args.perplexity:
        valid_path = Path(args.valid_file)
        if valid_path.exists():
            logger.info("Loading validation set: %s", valid_path)
            dataset = load_chat_dataset(
                valid_path, model_info.chat_template, max_samples=args.samples * 3
            )
            logger.info("Computing perplexity ...")
            ppl = compute_perplexity(model, tokenizer, dataset)
            logger.info("Perplexity: %.4f", ppl)
            results["perplexity"] = round(ppl, 4)
        else:
            logger.warning("Validation file not found: %s", valid_path)

    # -- Generation samples ----------------------------------------------
    if args.samples > 0:
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What is the capital of France?",
            "How do I train a neural network?",
        ]
        logger.info("Generating %s sample responses ...", min(args.samples, len(test_prompts)))
        responses = generate_samples(
            model, tokenizer, test_prompts[:args.samples]
        )
        samples = []
        for prompt, response in zip(test_prompts[:args.samples], responses):
            samples.append({"prompt": prompt, "response": response})
            logger.info("")
            logger.info("PROMPT: %s", prompt)
            logger.info("RESPONSE: %s", response[:300])
        results["samples"] = samples

    # -- Save results ----------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
