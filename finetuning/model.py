"""
AIOS Fine-tuning Model Module

Loads Qwen3-8B-Instruct (or any model in the registry) for fine-tuning.
Uses Unsloth when available for memory-efficient QLoRA, falling back to
standard HuggingFace Transformers + PEFT + BitsAndBytes.

Usage::

    from finetuning.model import build_model, load_for_inference

    # Training — uses Unsloth if installed, otherwise HF Transformers
    model, tokenizer = build_model(cfg)

    # Inference
    model, tokenizer = load_for_inference("checkpoints_v2/run/final")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from finetuning.config import TrainConfig
from finetuning.utils import detect_gpu, get_compute_dtype, setup_logger

logger = setup_logger("aios.finetuning.model")


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_HAS_UNSLOTH: bool = False
_UNSLOTH_ERR: str = ""

try:
    from unsloth import FastLanguageModel  # noqa: F401

    _HAS_UNSLOTH = True
except ImportError as exc:
    _UNSLOTH_ERR = str(exc)


def unsloth_available() -> bool:
    """Check whether the ``unsloth`` package is installed.

    Returns:
        ``True`` if Unsloth can be imported.
    """
    return _HAS_UNSLOTH


def _require_cuda() -> str:
    """Detect GPU and exit with a clear error if none is available.

    Returns:
        GPU description string.
    """
    available, desc = detect_gpu()
    if not available:
        print("=" * 62, file=sys.stderr)
        print("  No compatible GPU detected.", file=sys.stderr)
        print("  AIOS fine-tuning requires a CUDA-capable GPU.", file=sys.stderr)
        print(file=sys.stderr)
        print("  Requirements:", file=sys.stderr)
        print("  - NVIDIA GPU with 8+ GB VRAM for QLoRA", file=sys.stderr)
        print("  - CUDA Toolkit 11.8+ and compatible drivers", file=sys.stderr)
        print("  - PyTorch compiled with CUDA support", file=sys.stderr)
        print(file=sys.stderr)
        print("  Verify with:", file=sys.stderr)
        print("    python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        sys.exit(1)
    return desc


def _check_flash_attention() -> bool:
    """Check if Flash Attention 2 is available.

    Returns:
        ``True`` if flash-attn is installed and a compatible GPU is present.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Unsloth backend
# ---------------------------------------------------------------------------


def _build_unsloth(cfg: TrainConfig) -> Tuple[Any, Any]:
    """Build model using Unsloth's ``FastLanguageModel``.

    Supports QLoRA (4-bit), LoRA, and full fine-tuning with gradient
    checkpointing and Flash Attention.

    Args:
        cfg: Training configuration.

    Returns:
        A tuple ``(model, tokenizer)``.
    """
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model_info = cfg.model_info
    max_seq = cfg.max_seq_length
    dtype = get_compute_dtype(cfg.model_dtype)
    is_full = cfg.training_method == "full"

    logger.info("Backend: Unsloth %s", getattr(FastLanguageModel, "__version__", "unknown"))
    logger.info("Loading base model: %s", model_info.hf_path)
    logger.info("Max sequence length: %s", max_seq)

    if cfg.load_in_4bit:
        logger.info("Quantisation: 4-bit (NF4, double_quant=%s)", cfg.bnb_4bit_use_double_quant)

    if cfg.use_flash_attention:
        if _check_flash_attention():
            logger.info("Flash Attention 2: enabled")
        else:
            logger.warning("Flash Attention 2 requested but not available — falling back to eager mode")

    model_kwargs: Dict[str, Any] = {
        "model_name": model_info.hf_path,
        "max_seq_length": max_seq,
        "dtype": dtype,
        "load_in_4bit": cfg.load_in_4bit,
        "device_map": "auto",
        "token": None,
        "cache_dir": None,
        "trust_remote_code": True,
    }

    if cfg.use_flash_attention and _check_flash_attention():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    # Patch tokenizer for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_full:
        logger.info("Full fine-tuning — no LoRA adapters applied.")
        model = _enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)
        return model, tokenizer

    logger.info(
        "Applying LoRA (r=%s, alpha=%s, dropout=%s, rslora=%s, dora=%s) ...",
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.lora_dropout,
        cfg.use_rslora,
        cfg.use_dora,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=model_info.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
        loftq_config=None,
    )

    return model, tokenizer


def _load_unsloth(
    model_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    dtype: torch.dtype,
) -> Tuple[Any, Any]:
    """Load a model (adapters or merged) via Unsloth for inference.

    Args:
        model_path:     Path or HF identifier.
        max_seq_length: Maximum sequence length.
        load_in_4bit:   Load in 4-bit.
        dtype:          Torch dtype.

    Returns:
        A tuple ``(model, tokenizer)``.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )

    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend (fallback)
# ---------------------------------------------------------------------------


def _build_hf(cfg: TrainConfig) -> Tuple[Any, Any]:
    """Build model using standard HuggingFace Transformers + PEFT + BitsAndBytes.

    Used when Unsloth is not installed.

    Args:
        cfg: Training configuration.

    Returns:
        A tuple ``(model, tokenizer)``.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    model_info = cfg.model_info
    max_seq = cfg.max_seq_length
    dtype = get_compute_dtype(cfg.model_dtype)
    is_full = cfg.training_method == "full"

    logger.info("Backend: HuggingFace Transformers (Unsloth not installed)")
    logger.info("Loading base model: %s", model_info.hf_path)
    logger.info("Max sequence length: %s", max_seq)

    # --- Tokenizer ---
    tokenizer_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_info.hf_path,
        "trust_remote_code": True,
        "use_fast": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Quantisation ---
    quantization_config = None
    if cfg.load_in_4bit:
        logger.info("Quantisation: 4-bit (NF4, double_quant=%s)", cfg.bnb_4bit_use_double_quant)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=get_compute_dtype(cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
    elif cfg.training_method == "lora":
        logger.info("Quantisation: none (full-precision LoRA)")

    # --- Model ---
    attn_implementation = None
    if cfg.use_flash_attention:
        if _check_flash_attention():
            logger.info("Flash Attention 2: enabled")
            attn_implementation = "flash_attention_2"
        else:
            logger.warning("Flash Attention 2 requested but not available — using eager")

    model = AutoModelForCausalLM.from_pretrained(
        model_info.hf_path,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    # --- Gradient checkpointing ---
    model = _enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)

    # --- LoRA ---
    if is_full:
        logger.info("Full fine-tuning — no LoRA adapters applied.")
        return model, tokenizer

    logger.info(
        "Applying LoRA (r=%s, alpha=%s, dropout=%s, rslora=%s, dora=%s) ...",
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.lora_dropout,
        cfg.use_rslora,
        cfg.use_dora,
    )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_info.target_modules,
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
    )

    model = get_peft_model(model, peft_config)

    return model, tokenizer


def _load_hf(
    model_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    dtype: torch.dtype,
) -> Tuple[Any, Any]:
    """Load a model via HuggingFace Transformers for inference.

    Args:
        model_path:     Path or HF identifier.
        max_seq_length: Maximum sequence length.
        load_in_4bit:   Load in 4-bit.
        dtype:          Torch dtype.

    Returns:
        A tuple ``(model, tokenizer)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Gradient checkpointing helper
# ---------------------------------------------------------------------------


def _enable_gradient_checkpointing(model: Any, enabled: bool) -> Any:
    """Enable or disable gradient checkpointing on the model.

    Args:
        model:   The PyTorch model.
        enabled: Whether to enable checkpointing.

    Returns:
        The model (modified in-place).
    """
    if enabled:
        logger.info("Gradient checkpointing: enabled")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        logger.info("Gradient checkpointing: disabled")
    return model


# ---------------------------------------------------------------------------
# Public API — build_model
# ---------------------------------------------------------------------------


def build_model(cfg: TrainConfig) -> Tuple[Any, Any]:
    """Load a base model and prepare it for fine-tuning.

    Auto-detects the best backend:

    1. **Unsloth** (preferred) — memory-efficient QLoRA with 4-bit
       quantisation, Flash Attention, and fast LoRA kernels.
    2. **HuggingFace Transformers** (fallback) — standard PEFT + BitsAndBytes
       when Unsloth is not installed.

    Supported training methods (``cfg.training_method``):

    * ``"qlora"`` — 4-bit quantised base model + LoRA adapters.
    * ``"lora"``  — full-precision base model + LoRA adapters.
    * ``"full"``  — full fine-tuning (no adapters).

    Args:
        cfg: Training configuration.

    Returns:
        A tuple ``(model, tokenizer)`` ready for training.

    Raises:
        SystemExit: If no CUDA-capable GPU is detected.
    """
    _require_cuda()

    if cfg.max_seq_length <= 0:
        raise ValueError(
            f"max_seq_length must be > 0, got {cfg.max_seq_length}. "
            "Set it in your config or use a model_key that provides a default."
        )

    use_unsloth = unsloth_available()

    if not use_unsloth:
        logger.warning(
            "Unsloth not installed (%s). Falling back to HuggingFace Transformers. "
            "For better performance and memory efficiency, install Unsloth:\n"
            "  pip install unsloth",
            _UNSLOTH_ERR,
        )

    if use_unsloth:
        model, tokenizer = _build_unsloth(cfg)
    else:
        model, tokenizer = _build_hf(cfg)

    n_trainable = count_trainable_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable: %s / %s (%.2f%%)",
        f"{n_trainable:,}",
        f"{n_total:,}",
        100.0 * n_trainable / n_total if n_total else 0.0,
    )

    return model, tokenizer


def load_for_inference(
    model_path: str | Path,
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
    dtype: str = "auto",
) -> Tuple[Any, Any]:
    """Load a model (adapters or merged) for inference.

    Prefers Unsloth when available; falls back to HF Transformers.

    Args:
        model_path:     Path to the saved adapter directory or merged model.
        max_seq_length: Maximum sequence length.
        load_in_4bit:   Load in 4-bit for memory saving.
        dtype:          Model dtype string.

    Returns:
        A tuple ``(model, tokenizer)`` with the model in eval mode.
    """
    model_path = str(model_path)
    dtype_ = get_compute_dtype(dtype)

    logger.info("Loading model for inference from %s ...", model_path)
    logger.info("max_seq_length=%s, load_in_4bit=%s", max_seq_length, load_in_4bit)

    if unsloth_available():
        try:
            model, tokenizer = _load_unsloth(model_path, max_seq_length, load_in_4bit, dtype_)
        except Exception as exc:
            logger.warning("Unsloth inference load failed (%s). Falling back to HF.", exc)
            model, tokenizer = _load_hf(model_path, max_seq_length, load_in_4bit, dtype_)
    else:
        model, tokenizer = _load_hf(model_path, max_seq_length, load_in_4bit, dtype_)

    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


def load_for_training(cfg: TrainConfig) -> Tuple[Any, Any]:
    """Convenience wrapper around :func:`build_model`.

    Args:
        cfg: Training configuration.

    Returns:
        A tuple ``(model, tokenizer)``.
    """
    return build_model(cfg)


# ---------------------------------------------------------------------------
# Merge & save
# ---------------------------------------------------------------------------


def merge_and_unload(model: Any) -> Any:
    """Merge LoRA adapter weights into the base model and unload adapters.

    Works with both Unsloth PEFT models and standard HF PEFT models.

    Args:
        model: A PEFT model with LoRA adapters attached.

    Returns:
        The merged base model (no adapter wrappers).
    """
    logger.info("Merging LoRA adapters into base model ...")
    try:
        merged = model.merge_and_unload()
        logger.info("Merge complete.")
        return merged
    except AttributeError:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            merged = model.merge_and_unload()
            logger.info("Merge complete (HF PEFT).")
            return merged
        logger.warning("Model does not support merge_and_unload — returning as-is.")
        return model


def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    save_mode: str = "huggingface",
    quantization: Optional[str] = None,
) -> None:
    """Save model and tokenizer to disk.

    Args:
        model:        The model to save.
        tokenizer:    The tokenizer to save.
        output_dir:   Destination directory.
        save_mode:    ``"huggingface"`` or ``"gguf"``.
        quantization: GGUF quantization type (e.g. ``"q4_k_m"``).

    Raises:
        RuntimeError: If GGUF export is requested but Unsloth is not installed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_mode == "gguf":
        if not unsloth_available():
            raise RuntimeError(
                "GGUF export requires Unsloth.\n"
                "  pip install unsloth\n"
                "  Alternatively, use save_mode='huggingface' and convert manually."
            )
        from unsloth import save_to_gguf

        quantization = quantization or "q4_k_m"
        logger.info("Saving to GGUF (quantization: %s) ...", quantization)
        save_to_gguf(
            model=model,
            tokenizer=tokenizer,
            save_path=str(output_dir / f"model-{quantization}.gguf"),
            quantization=quantization,
        )
    else:
        logger.info("Saving in HuggingFace format to %s ...", output_dir)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    logger.info("Model saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def count_trainable_params(model: Any) -> int:
    """Count trainable (requires_grad) parameters.

    Args:
        model: A PyTorch model (may be PEFT-wrapped).

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_summary(model: Any, log: Optional[logging.Logger] = None) -> None:
    """Log a summary of the model architecture and parameter counts.

    Args:
        model: A PyTorch model.
        log:   Logger instance; uses module logger if ``None``.
    """
    log = log or logger

    n_trainable = count_trainable_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = n_total - n_trainable

    log.info("Model parameter summary:")
    log.info("  Total parameters:     %s", f"{n_total:,}")
    log.info("  Trainable parameters: %s", f"{n_trainable:,}")
    log.info("  Frozen parameters:    %s", f"{n_frozen:,}")
    log.info("  Trainable ratio:      %.2f%%", 100.0 * n_trainable / n_total if n_total else 0.0)
