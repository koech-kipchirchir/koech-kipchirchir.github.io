"""
AIOS Fine-tuning Merge

Merges LoRA adapter weights into the base model and saves the full-precision
or quantised merged model to disk.

This is a prerequisite for GGUF export and for deploying the model without
runtime adapter loading.

Usage::

    python -m finetuning.merge \\
        --model-path checkpoints_v2/run-name/final \\
        --model-key qwen3-8b \\
        --output-dir exports/merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from finetuning.config import list_supported_models  # noqa: E402
from finetuning.model import load_for_inference, merge_and_unload as merge_adapters, save_model  # noqa: E402
from finetuning.utils import setup_logger  # noqa: E402

logger = setup_logger("aios.finetuning.merge")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_and_save(
    adapter_path: Path,
    output_dir: Path,
    model_key: str,  # noqa: ARG001  (kept for CLI compatibility)
    max_seq: int,
    save_mode: str,
    quantization: Optional[str],
) -> None:
    """Merge LoRA adapters into the base model and save.

    Args:
        adapter_path: Path to the saved LoRA adapter directory.
        output_dir:   Destination directory for the merged model.
        model_key:    Model architecture key (ignored; inferred from adapter).
        max_seq:      Maximum sequence length.
        save_mode:    ``"huggingface"`` or ``"gguf"``.
        quantization: Optional quantization method (``"q4_k_m"``, ``"q8_0"``, etc.).
    """
    logger.info("Loading model from adapter path (merging) ...")
    model, tokenizer = load_for_inference(
        model_path=adapter_path,
        max_seq_length=max_seq,
        load_in_4bit=False,
    )

    merged = merge_adapters(model)
    save_model(merged, tokenizer, output_dir, save_mode, quantization)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AIOS Fine-tuning — Merge LoRA adapters into base model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter directory",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="qwen3-8b",
        choices=list_supported_models(),
        help="Model architecture key",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/merged",
        help="Output directory for the merged model",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--save-mode",
        type=str,
        choices=["huggingface", "gguf"],
        default="huggingface",
        help="Output format for the merged model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="GGUF quantization type (e.g. 'q4_k_m', 'q8_0'); used only when --save-mode=gguf",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    merge_and_save(
        adapter_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        model_key=args.model_key,
        max_seq=args.max_seq,
        save_mode=args.save_mode,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
