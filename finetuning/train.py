"""
AIOS Fine-tuning Entry Point

Trains a model using QLoRA (or LoRA / full fine-tuning) with the
configuration specified in ``configs/train.yaml``.

Features:
    - Automatic checkpoint resume (finds latest checkpoint in output dir).
    - TensorBoard logging.
    - Best-model tracking with automatic save.
    - Graceful exception handling with actionable messages.

Usage::

    python -m finetuning.train                              # uses configs/train.yaml
    python -m finetuning.train --config my_config.yaml       # custom config
    python -m finetuning.train --overwrite                   # force fresh start
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from finetuning.config import TrainConfig  # noqa: E402
from finetuning.dataset import load_datasets  # noqa: E402
from finetuning.model import build_model, log_model_summary  # noqa: E402
from finetuning.utils import require_gpu, setup_logger  # noqa: E402

logger = setup_logger("aios.finetuning.train")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class MetricsLoggerCallback(TrainerCallback):
    """Log training metrics to the console at each logging step."""

    def __init__(self, log_eval: bool = True) -> None:
        self.log_eval = log_eval

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.log_history:
            return
        metrics = state.log_history[-1]

        loss = metrics.get("loss")
        eval_loss = metrics.get("eval_loss")
        lr = metrics.get("learning_rate", 0.0)
        grad_norm = metrics.get("grad_norm")
        epoch = metrics.get("epoch", 0.0)

        parts = [f"Step {state.global_step}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if eval_loss is not None and self.log_eval:
            parts.append(f"val_loss={eval_loss:.4f}")
        if lr:
            parts.append(f"lr={lr:.2e}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if epoch:
            parts.append(f"epoch={epoch:.2f}")

        logger.info("  ".join(parts))


class ProgressCallback(TrainerCallback):
    """Print a progress bar summary every N steps."""

    def __init__(self, total_steps: int, log_interval: int = 50) -> None:
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time: Optional[float] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step % self.log_interval != 0 or step == 0:
            return

        elapsed = time.time() - (self.start_time or time.time())
        steps_per_sec = step / max(elapsed, 1e-6)
        remaining_steps = self.total_steps - step if self.total_steps > 0 else 0
        eta_secs = remaining_steps / max(steps_per_sec, 1e-6)

        def _fmt_secs(s: float) -> str:
            m, s = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        logger.info(
            "  Progress: %s / %s steps (%s%%) | %s steps/s | ETA %s",
            step,
            self.total_steps if self.total_steps > 0 else "?",
            f"{100.0 * step / self.total_steps:.1f}" if self.total_steps > 0 else "?",
            f"{steps_per_sec:.2f}",
            _fmt_secs(eta_secs) if self.total_steps > 0 else "?",
        )


class BestModelSaverCallback(TrainerCallback):
    """Log a message when a new best model is saved."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.best_model_checkpoint:
            logger.info(
                "Best model updated: %s (metric=%.4f)",
                state.best_model_checkpoint,
                state.best_metric,
            )


# ---------------------------------------------------------------------------
# Auto-resume helpers
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Scan *output_dir* for HuggingFace checkpoint directories.

    Returns the path of the checkpoint with the highest step number, or
    ``None`` if no checkpoint is found.

    Args:
        output_dir: The run output directory.

    Returns:
        Path to the latest checkpoint directory, or ``None``.
    """
    if not output_dir.is_dir():
        return None

    ckpt_dirs: List[Path] = []
    for entry in output_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            try:
                step = int(entry.name.split("-")[1])
                ckpt_dirs.append((step, entry))
            except (IndexError, ValueError):
                continue

    if not ckpt_dirs:
        return None

    ckpt_dirs.sort(key=lambda x: x[0])
    latest = str(ckpt_dirs[-1][1])
    return latest


def _has_checkpoint(output_dir: Path) -> bool:
    """Check if *output_dir* contains at least one checkpoint."""
    return _find_latest_checkpoint(output_dir) is not None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig, overwrite: bool = False) -> str:
    """Run the full fine-tuning loop.

    The pipeline:
        1. Validate environment (GPU, config).
        2. Prepare output directory (detect resume or overwrite).
        3. Seed everything for reproducibility.
        4. Save a copy of the config.
        5. Build model and tokenizer.
        6. Load and tokenise datasets.
        7. Configure ``TrainingArguments``, ``DataCollator``, ``Trainer``.
        8. Resume from latest checkpoint (or clean start).
        9. Train and save the final adapter.

    Args:
        cfg:       Training configuration.
        overwrite: If ``True``, delete the output directory before starting.

    Returns:
        Path to the final saved adapter directory.

    Raises:
        SystemExit: If no GPU is available.
        ValueError: If the configuration is invalid.
    """
    # -- Device -----------------------------------------------------------
    gpu_desc = require_gpu()
    bf16_supported = _bf16_supported()
    logger.info("GPU: %s", gpu_desc)
    logger.info("BF16 supported: %s", bf16_supported)

    # -- Output directory ------------------------------------------------
    output_dir = cfg.resolved_output_dir

    if overwrite and output_dir.exists():
        import shutil

        logger.warning("Overwriting output directory: %s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # -- Seed ------------------------------------------------------------
    from transformers import set_seed

    set_seed(cfg.seed)
    logger.info("Random seed: %s", cfg.seed)

    # -- Save config for reproducibility --------------------------------
    cfg.to_yaml(output_dir / "train_config.yaml")
    logger.info("Config saved to %s", output_dir / "train_config.yaml")

    # -- Model -----------------------------------------------------------
    logger.info("Building model ...")
    model, tokenizer = build_model(cfg)
    log_model_summary(model)

    # -- Dataset ---------------------------------------------------------
    logger.info("Loading datasets ...")
    datasets = load_datasets(cfg, tokenizer)

    n_train = len(datasets["train"])
    n_val = len(datasets["validation"])
    logger.info("Dataset sizes — train: %s, validation: %s", n_train, n_val)

    # -- Auto-resume detection -------------------------------------------
    resume_path: Optional[str] = None
    if cfg.resume_from_checkpoint is not None:
        resume_path = str(Path(cfg.resume_from_checkpoint).resolve())
        if not os.path.isdir(resume_path):
            logger.warning("Specified resume path not found: %s", resume_path)
            resume_path = None
        else:
            logger.info("Resuming from specified checkpoint: %s", resume_path)

    if resume_path is None and not overwrite:
        latest = _find_latest_checkpoint(output_dir)
        if latest is not None:
            resume_path = latest
            logger.info("Auto-resume: continuing from latest checkpoint: %s", latest)
        else:
            logger.info("No existing checkpoint found — starting fresh.")

    # -- Training arguments ----------------------------------------------
    max_steps = cfg.max_steps if cfg.max_steps > 0 else -1
    total_steps = max_steps if max_steps > 0 else _estimate_total_steps(cfg, n_train)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=cfg.run_name,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        max_steps=max_steps,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        tf32=True,
        gradient_checkpointing=cfg.use_gradient_checkpointing,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if n_val > 0 else "no",
        eval_steps=cfg.eval_steps if n_val > 0 else None,
        logging_steps=cfg.logging_steps,
        report_to=["tensorboard"] if cfg.use_tensorboard else [],
        load_best_model_at_end=cfg.load_best_model_at_end and n_val > 0,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=cfg.seed,
        data_seed=cfg.seed,
        ddp_find_unused_parameters=False,
        optim="adamw_8bit",
        disable_tqdm=False,
    )

    # -- Data collator ---------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # -- Callbacks -------------------------------------------------------
    callbacks: List[TrainerCallback] = [
        MetricsLoggerCallback(),
        ProgressCallback(total_steps=total_steps, log_interval=max(1, cfg.logging_steps * 5)),
    ]
    if n_val > 0:
        callbacks.append(BestModelSaverCallback())

    # -- Trainer ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"] if n_val > 0 else None,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # -- Print training plan ---------------------------------------------
    logger.info("=" * 62)
    logger.info("TRAINING PLAN")
    logger.info("  Model:              %s", cfg.model_info.hf_path)
    logger.info("  Method:             %s", cfg.training_method.upper())
    logger.info("  Run name:           %s", cfg.run_name)
    logger.info("  Output dir:         %s", output_dir)
    logger.info("  Epochs:             %s", cfg.num_epochs)
    logger.info("  Max steps:          %s", "auto" if max_steps <= 0 else max_steps)
    logger.info("  Batch (per device): %s", cfg.batch_size)
    logger.info("  Gradient accum:     %s", cfg.gradient_accumulation_steps)
    logger.info("  Effective batch:    %s", cfg.batch_size * cfg.gradient_accumulation_steps)
    logger.info("  Learning rate:      %s", cfg.learning_rate)
    logger.info("  Scheduler:          %s", cfg.lr_scheduler_type)
    logger.info("  Warmup ratio:       %s", cfg.warmup_ratio)
    logger.info("  Weight decay:       %s", cfg.weight_decay)
    logger.info("  Max seq length:     %s", cfg.max_seq_length)
    logger.info("  Seed:               %s", cfg.seed)
    logger.info("  Train samples:      %s", n_train)
    logger.info("  Validation samples: %s", n_val)
    logger.info("  Checkpoint every:   %s steps", cfg.save_steps)
    logger.info("  Eval every:         %s steps" if n_val > 0 else "  Eval:               disabled", cfg.eval_steps)
    logger.info("  Log every:          %s steps", cfg.logging_steps)
    logger.info("  Resume:             %s", resume_path or "no")
    logger.info("=" * 62)

    # -- Train -----------------------------------------------------------
    train_result = trainer.train(resume_from_checkpoint=resume_path)

    # -- Save final adapter ----------------------------------------------
    final_ckpt = str(output_dir / "final")
    logger.info("Saving final model to %s", final_ckpt)

    trainer.save_model(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)
    logger.info("Adapters saved to %s", final_ckpt)

    # -- Save training metrics -------------------------------------------
    metrics_path = output_dir / "train_results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2, default=str)
    logger.info("Training metrics saved to %s", metrics_path)

    # -- Save config again (final copy) ----------------------------------
    cfg.to_yaml(output_dir / "train_config.yaml")

    # -- Save loss curve (simple extract) --------------------------------
    if hasattr(trainer, "state") and trainer.state.log_history:
        loss_curve = [
            {
                "step": h.get("step", i),
                "loss": h.get("loss"),
                "eval_loss": h.get("eval_loss"),
                "learning_rate": h.get("learning_rate"),
                "epoch": h.get("epoch"),
            }
            for i, h in enumerate(trainer.state.log_history)
            if "loss" in h or "eval_loss" in h
        ]
        curve_path = output_dir / "loss_curve.json"
        with open(curve_path, "w", encoding="utf-8") as f:
            json.dump(loss_curve, f, indent=2)
        logger.info("Loss curve saved to %s", curve_path)

    logger.info("=" * 62)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best checkpoint:  %s", getattr(trainer.state, "best_model_checkpoint", "N/A"))
    logger.info("  Best metric:      %.4f" % getattr(trainer.state, "best_metric", 0.0)
                 if getattr(trainer.state, "best_metric", None) is not None
                 else "  Best metric:      N/A")
    logger.info("  Final adapter:    %s", final_ckpt)
    logger.info("  Output directory: %s", output_dir)
    logger.info("=" * 62)

    return final_ckpt


def _bf16_supported() -> bool:
    """Check if the hardware supports BF16 mixed precision.

    Uses Unsloth's detection when available, falls back to PyTorch.
    """
    try:
        from unsloth import is_bfloat16_supported

        return is_bfloat16_supported()
    except ImportError:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _estimate_total_steps(cfg: TrainConfig, n_train: int) -> int:
    """Estimate the total number of training steps.

    Uses the same formula as HuggingFace ``Trainer``::

        steps = ceil(epochs * n_train / (batch * accum))

    Args:
        cfg:     Training configuration.
        n_train: Number of training examples.

    Returns:
        Estimated total steps.
    """
    eff_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(n_train / eff_batch))
    return cfg.num_epochs * steps_per_epoch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="AIOS Fine-tuning with Unsloth + QLoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML configuration file (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a specific checkpoint directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before starting (fresh run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without training",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point with top-level exception handling."""
    args = parse_args()

    # --- Resolve config path ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # --- Load config ---
    try:
        cfg = TrainConfig.from_yaml(str(config_path))
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_dir():
            print(f"Error: resume path not found: {resume_path}", file=sys.stderr)
            sys.exit(1)
        cfg.resume_from_checkpoint = str(resume_path)

    # --- Dry run ---
    if args.dry_run:
        print("Configuration valid.")
        print(f"  Model:         {cfg.model_info.hf_path}")
        print(f"  Method:        {cfg.training_method}")
        print(f"  Output dir:    {cfg.resolved_output_dir}")
        print(f"  Train file:    {cfg.resolved_train_file}")
        print(f"  Valid file:    {cfg.resolved_valid_file}")
        print(f"  Max seq:       {cfg.max_seq_length}")
        print(f"  Batch:         {cfg.batch_size}")
        print(f"  Accumulation:  {cfg.gradient_accumulation_steps}")
        print(f"  Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps}")
        print(f"  Epochs:        {cfg.num_epochs}")
        print(f"  LR:            {cfg.learning_rate}")
        print(f"  LoRA r:        {cfg.lora_r}")
        print(f"  Seed:          {cfg.seed}")
        print(f"  Resume:        {cfg.resume_from_checkpoint or 'none'}")
        return

    # --- Train ---
    try:
        train(cfg, overwrite=args.overwrite)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.critical("Training failed with an unexpected error:")
        traceback.print_exc()
        print(file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        print("  TROUBLESHOOTING", file=sys.stderr)
        print(file=sys.stderr)
        print("  Common issues:", file=sys.stderr)
        print("  1. Out of memory (OOM) — reduce batch_size or max_seq_length", file=sys.stderr)
        print("  2. CUDA error — check nvidia-smi and driver version", file=sys.stderr)
        print("  3. Missing dependency — pip install -r requirements.txt", file=sys.stderr)
        print("  4. Disk space — free up space in the output directory", file=sys.stderr)
        print(file=sys.stderr)
        print("  Check logs/aios.finetuning.train.log for details.", file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
