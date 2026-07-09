"""
AIOS Fine-tuning Package

End-to-end fine-tuning pipeline for AIOS using Unsloth + QLoRA.
"""

from __future__ import annotations

from finetuning.config import (
    LoRAConfig,
    MODEL_REGISTRY,
    ModelInfo,
    TrainConfig,
    list_supported_models,
)
from finetuning.dataset import (
    ChatExample,
    Turn,
    format_example,
    load_dataset,
    load_datasets,
    pack_sequences,
    tokenize_with_labels,
    validate_example,
)
from finetuning.model import (
    build_model,
    count_trainable_params,
    load_for_inference,
    load_for_training,
    log_model_summary,
    merge_and_unload,
    save_model,
)
from finetuning.utils import (
    detect_gpu,
    find_project_root,
    format_timestamp,
    get_compute_dtype,
    require_gpu,
    setup_logger,
)

__all__ = [
    # config
    "LoRAConfig",
    "MODEL_REGISTRY",
    "ModelInfo",
    "TrainConfig",
    "list_supported_models",
    # dataset
    "ChatExample",
    "Turn",
    "format_example",
    "load_dataset",
    "load_datasets",
    "pack_sequences",
    "tokenize_with_labels",
    "validate_example",
    # model
    "build_model",
    "count_trainable_params",
    "load_for_inference",
    "load_for_training",
    "log_model_summary",
    "merge_and_unload",
    "save_model",
    # utils
    "detect_gpu",
    "find_project_root",
    "format_timestamp",
    "get_compute_dtype",
    "require_gpu",
    "setup_logger",
]

__version__ = "0.2.0"
