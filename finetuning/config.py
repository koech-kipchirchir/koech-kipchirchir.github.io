"""
AIOS Fine-tuning Configuration

Provides a validated, dataclass-based configuration loaded from a YAML file
with sensible defaults for QLoRA fine-tuning with Unsloth.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingMethod(str, Enum):
    """Supported fine-tuning methods."""
    QLORA = "qlora"
    LORA = "lora"
    FULL = "full"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Model registry — extend here to support additional architectures
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Metadata for a supported base model."""

    hf_path: str
    """HuggingFace model identifier."""

    chat_template: str
    """Name of the chat template (e.g. ``"qwen2.5"``, ``"llama3"``)."""

    max_seq_length: int
    """Default maximum sequence length for this architecture."""

    target_modules: List[str]
    """LoRA target module name patterns."""


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "qwen3-8b": ModelInfo(
        hf_path="Qwen/Qwen3-8B-Instruct",
        chat_template="qwen2.5",
        max_seq_length=8192,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
    "qwen3-1.7b": ModelInfo(
        hf_path="Qwen/Qwen3-1.7B-Instruct",
        chat_template="qwen2.5",
        max_seq_length=8192,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
    "llama-3.2-3b": ModelInfo(
        hf_path="meta-llama/Llama-3.2-3B-Instruct",
        chat_template="llama3",
        max_seq_length=8192,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
    "gemma-2-2b": ModelInfo(
        hf_path="google/gemma-2-2b-it",
        chat_template="gemma",
        max_seq_length=8192,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
    "mistral-7b": ModelInfo(
        hf_path="mistralai/Mistral-7B-Instruct-v0.3",
        chat_template="mistral",
        max_seq_length=8192,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
}


def list_supported_models() -> List[str]:
    """Return a sorted list of supported model keys."""
    return sorted(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# LoRA configuration (standalone)
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """LoRA-specific hyper-parameters that can be loaded from ``lora.yaml``.

    When provided alongside a ``TrainConfig``, these values take precedence
    over the ``TrainConfig`` defaults for the LoRA-related fields.
    """

    r: int = 16
    """LoRA rank."""

    alpha: int = 32
    """LoRA alpha scaling factor."""

    dropout: float = 0.0
    """LoRA dropout probability."""

    bias: str = "none"
    """LoRA bias setting (``"none"``, ``"all"``, ``"lora_only"``)."""

    use_rslora: bool = True
    """Use rank-stabilised LoRA."""

    use_dora: bool = False
    """Use weight-decomposed LoRA."""

    target_modules: List[str] = field(default_factory=list)
    """Override target module names. Empty = use model defaults."""

    load_in_4bit: bool = True
    """Load base model in 4-bit."""

    bnb_4bit_quant_type: str = "nf4"
    """4-bit quantisation type (``"nf4"`` or ``"fp4"``)."""

    bnb_4bit_compute_dtype: str = "bfloat16"
    """Compute dtype for 4-bit layers."""

    bnb_4bit_use_double_quant: bool = True
    """Enable double quantisation."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoRAConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A populated ``LoRAConfig`` instance.
        """
        import yaml

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        import yaml

        path = Path(path)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def apply_to(self, cfg: "TrainConfig") -> None:
        """Overlay these LoRA values onto a ``TrainConfig``.

        Args:
            cfg: A ``TrainConfig`` instance to update in-place.
        """
        cfg.lora_r = self.r
        cfg.lora_alpha = self.alpha
        cfg.lora_dropout = self.dropout
        cfg.lora_bias = self.bias
        cfg.use_rslora = self.use_rslora
        cfg.use_dora = self.use_dora
        cfg.load_in_4bit = self.load_in_4bit
        cfg.bnb_4bit_quant_type = self.bnb_4bit_quant_type
        cfg.bnb_4bit_compute_dtype = self.bnb_4bit_compute_dtype
        cfg.bnb_4bit_use_double_quant = self.bnb_4bit_use_double_quant
        if self.target_modules:
            cfg.model_info.target_modules = self.target_modules


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All hyper-parameters and paths for a fine-tuning run.

    Supports two YAML layouts:

    * **Flat** (legacy) — all keys at the top level, matching attribute names.
    * **Nested** — grouped under ``model``, ``training``, ``lora``, ``export``.
    """

    # -- Model -----------------------------------------------------------------
    model_key: str = "qwen3-8b"
    """Key into :data:`MODEL_REGISTRY`."""

    model_provider: str = "huggingface"
    """Model provider (``"huggingface"``, ``"local"``)."""

    model_dtype: str = "auto"
    """Model dtype (``"auto"``, ``"float16"``, ``"bfloat16"``, ``"float32"``)."""

    load_in_4bit: bool = True
    """Load base model in 4-bit (QLoRA)."""

    use_flash_attention: bool = False
    """Enable Flash Attention 2 (requires compatible GPU)."""

    # -- Training method -------------------------------------------------------
    training_method: str = "qlora"
    """Fine-tuning method (``"qlora"``, ``"lora"``, ``"full"``)."""

    # -- LoRA ------------------------------------------------------------------
    lora_r: int = 16
    """LoRA rank."""

    lora_alpha: int = 16
    """LoRA alpha scaling factor."""

    lora_dropout: float = 0.0
    """LoRA dropout probability."""

    lora_bias: str = "none"
    """LoRA bias setting (``"none"``, ``"all"``, ``"lora_only"``)."""

    use_rslora: bool = True
    """Use :math:`R^2`-normalised LoRA (rank-stabilised)."""

    use_dora: bool = False
    """Use DoRA (weight-decomposed LoRA)."""

    # -- QLoRA ----------------------------------------------------------------
    bnb_4bit_quant_type: str = "nf4"
    """4-bit quantisation type (``"nf4"`` or ``"fp4"``)."""

    bnb_4bit_compute_dtype: str = "bfloat16"
    """Compute dtype for 4-bit layers."""

    bnb_4bit_use_double_quant: bool = True
    """Enable double quantisation for additional memory saving."""

    # -- Training --------------------------------------------------------------
    max_seq_length: int = 0
    """Override for max sequence length (0 = use model default)."""

    batch_size: int = 2
    """Per-device training batch size."""

    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps."""

    learning_rate: float = 2e-4
    """Peak learning rate."""

    lr_scheduler_type: str = "cosine"
    """LR scheduler (``"linear"``, ``"cosine"``, ``"constant"``, …)."""

    warmup_ratio: float = 0.03
    """Fraction of total steps used for linear warmup."""

    num_epochs: int = 3
    """Number of training epochs."""

    max_steps: int = -1
    """Hard limit on training steps (``-1`` = use epochs)."""

    weight_decay: float = 0.01
    """AdamW weight decay."""

    adam_beta1: float = 0.9
    """Adam beta1."""

    adam_beta2: float = 0.95
    """Adam beta2."""

    adam_epsilon: float = 1e-8
    """Adam epsilon."""

    max_grad_norm: float = 1.0
    """Gradient clipping norm."""

    seed: int = 42
    """Random seed for reproducibility."""

    # -- Optimisations ---------------------------------------------------------
    use_gradient_checkpointing: bool = True
    """Enable gradient checkpointing to reduce VRAM."""

    use_mixed_precision: bool = True
    """Enable ``bf16`` mixed-precision training."""

    packing: bool = False
    """Pack multiple short sequences into one (improves throughput)."""

    # -- Checkpointing & logging -----------------------------------------------
    output_dir: str = "outputs"
    """Root directory for checkpoints and logs."""

    run_name: str = ""
    """Optional run name (defaults to ``{model_key}-{timestamp}``)."""

    save_steps: int = 100
    """Save checkpoint every N steps."""

    eval_steps: int = 100
    """Run evaluation every N steps."""

    logging_steps: int = 10
    """Log metrics every N steps."""

    save_total_limit: int = 3
    """Keep at most this many checkpoints (oldest deleted)."""

    load_best_model_at_end: bool = True
    """Restore the best checkpoint when training finishes."""

    metric_for_best_model: str = "eval_loss"
    """Metric used to determine the best checkpoint."""

    greater_is_better: bool = False
    """Whether a higher metric value is better."""

    resume_from_checkpoint: Optional[str] = None
    """Path to a checkpoint directory to resume from."""

    # -- TensorBoard -----------------------------------------------------------
    use_tensorboard: bool = True
    """Log metrics to TensorBoard."""

    # -- Dataset ---------------------------------------------------------------
    train_file: str = "datasets/train.jsonl"
    """Path to the training dataset file."""

    valid_file: str = "datasets/valid.jsonl"
    """Path to the validation dataset file."""

    dataset_size: int = 0
    """Limit dataset to first N examples (0 = use all)."""

    # -- Save / export ---------------------------------------------------------
    save_adapters: bool = True
    """Save LoRA adapters at the end of training."""

    merge_model: bool = True
    """Merge LoRA adapters into the base model after training."""

    export_gguf: bool = True
    """Export the merged model to GGUF format."""

    export_ollama: bool = True
    """Generate an Ollama Modelfile for the exported model."""

    push_to_hub: bool = False
    """Push the final model to HuggingFace Hub."""

    hub_model_id: str = ""
    """HuggingFace Hub repository name."""

    # -- Internal (set by trainer) ---------------------------------------------
    _project_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    _device: str = ""
    _hf_path_override: str = ""

    # --------------------------------------------------------------------------
    # Resolve model key from name
    # --------------------------------------------------------------------------

    @staticmethod
    def _resolve_model_key(name: str) -> str:
        """Resolve a model name or path to a registry key.

        Priority:
        1. Exact match against registry keys.
        2. Match against registry ``hf_path`` values.
        3. Treat as a direct HuggingFace path (returns key *and* sets override).

        Args:
            name: Model name (key) or HuggingFace path.

        Returns:
            The registry key (or ``"custom"`` if not found).

        Side effect:
            If the name is not a registry key, stores it in ``_hf_path_override``
            so the caller can retrieve it.
        """
        name_lower = name.lower().replace("_", "-")

        # 1. Exact match against keys
        for key in MODEL_REGISTRY:
            if key.lower() == name_lower:
                return key

        # 2. Match against hf_path values
        for key, info in MODEL_REGISTRY.items():
            if info.hf_path.lower() == name_lower:
                return key

        # 3. HF path contains a slash — treat as direct path
        if "/" in name:
            raise ValueError(
                f"Model {name!r} is not in the registry. "
                f"Supported models: {list_supported_models()}. "
                f"Use a registry key or add the model to MODEL_REGISTRY."
            )

        raise ValueError(
            f"Unknown model_key={name!r}. "
            f"Supported: {list_supported_models()}"
        )

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Normalise and validate configuration values."""
        if self.training_method not in ("qlora", "lora", "full"):
            sys.exit(f"Unknown training_method={self.training_method!r}. Expected: qlora, lora, full")

        if self.training_method == "full":
            self.load_in_4bit = False
            self.lora_r = 0

        if self.max_seq_length == 0:
            self.max_seq_length = MODEL_REGISTRY.get(self.model_key, ModelInfo(
                hf_path="", chat_template="", max_seq_length=4096, target_modules=[]
            )).max_seq_length

        if not self.run_name:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_key}-{ts}"

    @property
    def model_info(self) -> ModelInfo:
        """Shortcut to the resolved :class:`ModelInfo`.

        Returns the cached override if one was set (e.g. with custom
        ``target_modules`` or ``hf_path``), otherwise returns the registry
        entry for ``model_key``.
        """
        override: Optional[ModelInfo] = getattr(self, "_model_info_override", None)
        if override is not None:
            return override

        info = MODEL_REGISTRY.get(self.model_key)
        if info is None:
            return ModelInfo(
                hf_path=self._hf_path_override or self.model_key,
                chat_template="qwen2.5",
                max_seq_length=self.max_seq_length or 4096,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
        if self._hf_path_override:
            from dataclasses import replace
            return replace(info, hf_path=self._hf_path_override)
        return info

    @property
    def resolved_output_dir(self) -> Path:
        """Absolute path to the output directory."""
        return self._project_root / self.output_dir / self.run_name

    @property
    def resolved_train_file(self) -> Path:
        """Absolute path to the training file."""
        return self._project_root / self.train_file

    @property
    def resolved_valid_file(self) -> Path:
        """Absolute path to the validation file."""
        return self._project_root / self.valid_file

    # --------------------------------------------------------------------------
    # Serialisation
    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a YAML file.

        Supports both flat and nested (grouped) YAML layouts.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A populated ``TrainConfig`` instance.
        """
        import yaml

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """Build a ``TrainConfig`` from a dict that may be flat or nested.

        A dict is considered **nested** when at least one top-level value is
        itself a ``dict`` (e.g. ``model: {name: …}``).  Nested keys are
        flattened before constructing the dataclass.

        Args:
            data: Configuration dictionary.

        Returns:
            A populated ``TrainConfig`` instance.
        """
        has_nested = any(isinstance(v, dict) for v in data.values())

        if not has_nested:
            # Flat format — pass through directly
            kwargs = dict(data)
            cls._validate_model_key(kwargs)
            return cls(**kwargs)

        # Nested format — flatten
        flat: Dict[str, Any] = {}

        # --- model section ---
        model_cfg = data.get("model", {}) or {}
        provider = model_cfg.get("provider", "huggingface")
        name = model_cfg.get("name", "")

        if name:
            try:
                key = cls._resolve_model_key(name)
            except ValueError:
                # Not in registry — use as-is; model_info will provide defaults
                key = name.replace("/", "-").replace("_", "-").lower()
                flat["_hf_path_override"] = name
            flat["model_key"] = key
        else:
            flat["model_key"] = "qwen3-8b"

        flat["model_provider"] = provider
        flat["model_dtype"] = model_cfg.get("dtype", "auto")
        flat["load_in_4bit"] = model_cfg.get("load_in_4bit", True)
        flat["use_flash_attention"] = model_cfg.get("use_flash_attention", False)

        # --- training section ---
        train_cfg = data.get("training", {}) or {}
        flat["training_method"] = train_cfg.get("method", "qlora")
        flat["num_epochs"] = train_cfg.get("epochs", 3)
        flat["learning_rate"] = train_cfg.get("learning_rate", 2e-4)
        flat["batch_size"] = train_cfg.get("batch_size", 1)
        flat["gradient_accumulation_steps"] = train_cfg.get("gradient_accumulation", 16)
        flat["warmup_ratio"] = train_cfg.get("warmup_ratio", 0.03)
        flat["weight_decay"] = train_cfg.get("weight_decay", 0.01)
        flat["max_seq_length"] = train_cfg.get("max_seq_length", 4096)
        flat["save_steps"] = train_cfg.get("save_steps", 100)
        flat["eval_steps"] = train_cfg.get("eval_steps", 100)
        flat["logging_steps"] = train_cfg.get("logging_steps", 10)
        flat["output_dir"] = train_cfg.get("output_dir", "outputs")
        flat["seed"] = train_cfg.get("seed", 42)
        # Pass through extra training keys
        for extra in ("lr_scheduler_type", "max_steps", "adam_beta1", "adam_beta2",
                       "adam_epsilon", "max_grad_norm", "packing",
                       "use_gradient_checkpointing", "save_total_limit",
                       "load_best_model_at_end", "metric_for_best_model",
                       "greater_is_better", "resume_from_checkpoint",
                       "use_tensorboard", "save_adapters"):
            if extra in train_cfg:
                flat[extra] = train_cfg[extra]

        # --- lora section ---
        lora_cfg = data.get("lora", {}) or {}
        flat["lora_r"] = lora_cfg.get("r", 16)
        flat["lora_alpha"] = lora_cfg.get("alpha", 32)
        flat["lora_dropout"] = lora_cfg.get("dropout", 0.0)
        flat["lora_bias"] = lora_cfg.get("bias", "none")
        flat["use_rslora"] = lora_cfg.get("use_rslora", True)
        flat["use_dora"] = lora_cfg.get("use_dora", False)
        # target_modules are consumed during model build; store in bnb overrides
        tm = lora_cfg.get("target_modules", [])
        if tm:
            flat["_target_modules_override"] = tm

        # --- export section ---
        export_cfg = data.get("export", {}) or {}
        flat["save_adapters"] = export_cfg.get("save_adapter", True)
        flat["merge_model"] = export_cfg.get("merge_model", True)
        flat["export_gguf"] = export_cfg.get("export_gguf", True)
        flat["export_ollama"] = export_cfg.get("export_ollama", True)

        # Pop internal-use keys before constructing the dataclass
        target_modules_override = flat.pop("_target_modules_override", None)

        cfg = cls(**flat)

        # Apply target module overrides
        if target_modules_override:
            import copy
            info = copy.deepcopy(cfg.model_info)
            info.target_modules = tm
            # Monkey-patch the model_info property for this instance
            object.__setattr__(cfg, "_model_info_override", info)

        return cfg

    @staticmethod
    def _validate_model_key(kwargs: Dict[str, Any]) -> None:
        """Validate that ``model_key`` is present and known."""
        mk = kwargs.get("model_key", "qwen3-8b")
        if mk not in MODEL_REGISTRY:
            # Try resolving as hf_path
            for key, info in MODEL_REGISTRY.items():
                if info.hf_path == mk:
                    kwargs["model_key"] = key
                    return

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        import yaml

        path = Path(path)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
