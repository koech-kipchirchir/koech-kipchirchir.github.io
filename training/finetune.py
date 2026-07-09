import os
import sys
import json
from typing import Optional, Dict, Any, List, Literal

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)

from training.config import TrainingConfig
from training.trainer import Trainer
from training.tokenizer import TokenizerManager
from training.utils import get_logger, detect_device, set_seed
from training.checkpoint import CheckpointManager

logger = get_logger(__name__)


SUPPORTED_MODEL_FAMILIES = {
    "gemma": ["google/gemma-2b", "google/gemma-7b", "google/gemma-2-2b", "google/gemma-2-7b"],
    "llama": [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-3-8b",
        "meta-llama/Llama-3-70b",
        "NousResearch/Llama-2-7b-hf",
    ],
    "qwen": [
        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen2-1.5B",
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-72B",
    ],
    "mistral": [
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-v0.1",
    ],
    "phi": [
        "microsoft/phi-1_5",
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-small-8k-instruct",
    ],
    "deepseek": [
        "deepseek-ai/deepseek-llm-7b-base",
        "deepseek-ai/deepseek-llm-67b-base",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
    ],
}


MODEL_FAMILY_PATTERNS: Dict[str, List[str]] = {
    "gemma": ["gemma"],
    "llama": ["llama"],
    "qwen": ["qwen", "qwen2"],
    "mistral": ["mistral", "mixtral"],
    "phi": ["phi"],
    "deepseek": ["deepseek"],
}


def detect_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    for family, patterns in MODEL_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return family
    return "unknown"


def get_model_kwargs(model_family: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,
    }

    if model_family == "gemma":
        pass
    elif model_family == "llama":
        pass
    elif model_family == "qwen":
        kwargs["trust_remote_code"] = True
    elif model_family == "mistral":
        pass
    elif model_family == "phi":
        kwargs["trust_remote_code"] = True
    elif model_family == "deepseek":
        kwargs["trust_remote_code"] = True

    return kwargs


class FineTuner:
    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        self.model_family: str = ""
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.trainer: Optional[Trainer] = None

        set_seed(config.seed)

    def setup(self) -> None:
        self.model_family = detect_model_family(self.config.model_name)
        logger.info("Detected model family: %s", self.model_family)

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        self._apply_peft()

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
            or detect_model_family(self.config.model_name) in ("qwen", "phi", "deepseek"),
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.chat_template is None:
            logger.info("No chat template; using default completion format.")

        logger.info(
            "Tokenizer loaded (vocab_size=%d, pad_token='%s')",
            len(tokenizer),
            tokenizer.pad_token,
        )
        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        model_kwargs = get_model_kwargs(self.model_family)

        if self.config.trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        if self.config.use_peft and self.config.peft_method == "qlora":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.config.qlora_bits == 4,
                load_in_8bit=self.config.qlora_bits == 8,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=self.config.qlora_double_quant,
                bnb_4bit_quant_type=self.config.qlora_quant_type,
            )

        if self.config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        device = detect_device()
        if device.type == "cpu":
            model_kwargs["torch_dtype"] = torch.float32

        logger.info(
            "Loading model '%s' (family=%s, device=%s, dtype=%s)",
            self.config.model_name,
            self.model_family,
            device,
            model_kwargs["torch_dtype"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            logger.info("Gradient checkpointing enabled.")

        model = model.to(device)
        model.config.use_cache = not self.config.gradient_checkpointing

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model loaded: %s total params, %s trainable params",
            f"{n_params:,}",
            f"{n_trainable:,}",
        )

        return model

    def _apply_peft(self) -> None:
        if not self.config.use_peft:
            logger.info("PEFT disabled — training full model.")
            return

        from peft import (
            LoraConfig,
            get_peft_model,
            TaskType,
            prepare_model_for_kbit_training,
        )

        if self.config.peft_method == "qlora":
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("Model prepared for k-bit (QLoRA) training.")

        target_modules = self.config.lora_target_modules
        if target_modules is None:
            target_modules = self._auto_detect_target_modules()

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info(
            "LoRA applied (r=%d, alpha=%d, dropout=%.2f, targets=%s)",
            self.config.lora_r,
            self.config.lora_alpha,
            self.config.lora_dropout,
            target_modules,
        )

    def _auto_detect_target_modules(self) -> List[str]:
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        if self.model_family == "gemma":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif self.model_family == "llama":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif self.model_family == "qwen":
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif self.model_family == "mistral":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif self.model_family == "phi":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif self.model_family == "deepseek":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            logger.warning("Unknown model family '%s'; using common attention modules.", self.model_family)
            return common

    def train(self) -> Dict[str, Any]:
        self.setup()

        self.trainer = Trainer(
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        result = self.trainer.train()
        logger.info("Fine-tuning complete.")
        return result

    def resume(self, checkpoint_path: str) -> Dict[str, Any]:
        self.config.resume_from_checkpoint = checkpoint_path
        return self.train()

    def save_pretrained(self, output_dir: Optional[str] = None) -> str:
        path = output_dir or os.path.join(self.config.output_dir, "final_model")
        os.makedirs(path, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

        logger.info("Model saved to %s", path)
        return path


def create_finetuning_config(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    peft_method: Literal["lora", "qlora"] = "qlora",
    output_dir: str = "./finetune_output",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 2e-4,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    **kwargs,
) -> TrainingConfig:
    return TrainingConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        use_peft=True,
        peft_method=peft_method,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs,
    )
