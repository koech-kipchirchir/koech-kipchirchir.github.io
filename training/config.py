import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal


@dataclass
class TrainingConfig:
    model_name: str = "microsoft/phi-2"
    """HuggingFace model identifier or local path."""

    dataset_name: str = ""
    """HuggingFace dataset identifier or local path."""

    dataset_config: Optional[str] = None
    """Dataset configuration / subset name."""

    dataset_split: str = "train"
    """Dataset split to use for training."""

    dataset_text_field: str = "text"
    """Column name containing the training text."""

    dataset_val_split: Optional[float] = 0.05
    """Fraction of training data to hold out for validation."""

    output_dir: str = "./outputs"
    """Root directory for saving model checkpoints and logs."""

    run_name: str = "run"
    """Human-readable name for this run (used in logging)."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ------------------------------------------------------------------
    #  Training hyper-parameters
    # ------------------------------------------------------------------
    num_train_epochs: int = 3
    """Number of training epochs."""

    per_device_train_batch_size: int = 4
    """Batch size per device during training."""

    per_device_eval_batch_size: int = 4
    """Batch size per device during evaluation."""

    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients before an optimizer step."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    learning_rate: float = 2e-4
    """Peak learning rate."""

    weight_decay: float = 0.01
    """Weight decay (L2 regularisation)."""

    adam_beta1: float = 0.9
    """Adam beta1."""

    adam_beta2: float = 0.999
    """Adam beta2."""

    adam_epsilon: float = 1e-8
    """Adam epsilon."""

    max_steps: int = -1
    """If > 0, training stops after this many steps (overrides epochs)."""

    warmup_ratio: float = 0.03
    """Fraction of total steps used for linear warmup."""

    lr_scheduler_type: str = "cosine"
    """LR scheduler type (linear, cosine, cosine_with_restarts, constant, etc.)."""

    # ------------------------------------------------------------------
    #  Mixed precision & performance
    # ------------------------------------------------------------------
    fp16: bool = False
    """Enable FP16 training."""

    bf16: bool = False
    """Enable BF16 training (preferred on Ampere+ GPUs)."""

    tf32: bool = True
    """Allow TF32 matrix multiplications on Ampere GPUs."""

    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save VRAM."""

    dataloader_num_workers: int = 0
    """Number of dataloader sub-processes (0 = main process)."""

    dataloader_pin_memory: bool = True
    """Pin memory in dataloader for faster GPU transfer."""

    # ------------------------------------------------------------------
    #  PEFT (LoRA / QLoRA)
    # ------------------------------------------------------------------
    use_peft: bool = True
    """Enable Parameter-Efficient Fine-Tuning."""

    peft_method: Literal["lora", "qlora"] = "lora"
    """PEFT method: 'lora' or 'qlora'."""

    lora_r: int = 8
    """LoRA rank."""

    lora_alpha: int = 16
    """LoRA alpha scaling."""

    lora_dropout: float = 0.05
    """LoRA dropout rate."""

    lora_target_modules: Optional[List[str]] = None
    """Modules to apply LoRA to (None = auto-detect)."""

    qlora_bits: Literal[4, 8] = 4
    """Quantisation bits for QLoRA."""

    qlora_double_quant: bool = True
    """Nested quantisation (Double Quantisation)."""

    qlora_quant_type: Literal["nf4", "fp4"] = "nf4"
    """Quantisation data type."""

    # ------------------------------------------------------------------
    #  Checkpointing
    # ------------------------------------------------------------------
    save_strategy: Literal["steps", "epoch", "no"] = "steps"
    """When to save checkpoints."""

    save_steps: int = 500
    """Save checkpoint every N steps (when save_strategy='steps')."""

    save_total_limit: int = 3
    """Keep at most this many checkpoints (older ones are deleted)."""

    load_best_model_at_end: bool = True
    """Load the best checkpoint (by eval loss) at the end of training."""

    resume_from_checkpoint: Optional[str] = None
    """Path to a specific checkpoint to resume from."""

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    evaluation_strategy: Literal["steps", "epoch", "no"] = "steps"
    """When to run evaluation."""

    eval_steps: int = 500
    """Evaluate every N steps."""

    eval_accumulation_steps: Optional[int] = None
    """Number of steps to accumulate eval outputs (None = same as grad accum)."""

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    logging_strategy: Literal["steps", "epoch", "no"] = "steps"
    """When to log metrics."""

    logging_steps: int = 25
    """Log every N steps."""

    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    """Logging backends: 'tensorboard', 'wandb', 'none'."""

    project_name: str = "aios-trainer"
    """Project name for WandB / MLflow."""

    # ------------------------------------------------------------------
    #  Google Drive / Colab
    # ------------------------------------------------------------------
    gdrive_mount: bool = False
    """Mount Google Drive for saving checkpoints (Colab)."""

    gdrive_output_dir: Optional[str] = None
    """Drive path override (defaults to 'MyDrive/<project_name>')."""

    # ------------------------------------------------------------------
    #  Misc
    # ------------------------------------------------------------------
    trust_remote_code: bool = False
    """Trust remote code when loading models (use only for trusted sources)."""

    max_seq_length: int = 2048
    """Maximum sequence length for tokenization."""

    packing: bool = False
    """Pack multiple short sequences into one sample for efficiency."""

    # ------------------------------------------------------------------
    #  Internal (populated during training)
    # ------------------------------------------------------------------
    device: str = "auto"
    """Device override ('auto', 'cuda', 'cpu')."""

    world_size: int = 1
    """Number of distributed processes (populated by Accelerate)."""

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self) -> None:
        if self.bf16 and self.fp16:
            raise ValueError("Only one of bf16 / fp16 can be enabled.")
        if self.dataset_val_split is not None and not (0 < self.dataset_val_split < 1):
            raise ValueError("dataset_val_split must be in (0, 1) or None.")
