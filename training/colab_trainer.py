import os
import sys
import subprocess
import json
import time
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch

from training.config import TrainingConfig
from training.trainer import Trainer
from training.utils import (
    get_logger,
    detect_device,
    is_colab,
    is_kaggle,
    get_gpu_memory,
)
from training.checkpoint import CheckpointManager

logger = get_logger(__name__)


class ColabTrainer(Trainer):
    def __init__(
        self,
        config: TrainingConfig,
        auto_setup: bool = True,
        gpu_memory_threshold_mb: int = 14000,
    ) -> None:
        self.gpu_memory_threshold_mb = gpu_memory_threshold_mb
        self._session_recovered = False

        if auto_setup:
            config = self._configure_for_environment(config)

        super().__init__(config)

    def _configure_for_environment(self, config: TrainingConfig) -> TrainingConfig:
        config = self._detect_and_configure_gpu(config)
        config = self._configure_colab_drive(config)
        config = self._configure_memory_optimizations(config)
        return config

    def _detect_and_configure_gpu(self, config: TrainingConfig) -> TrainingConfig:
        if not torch.cuda.is_available():
            logger.info("No GPU detected. Falling back to CPU.")
            config.device = "cpu"
            config.use_peft = False
            config.gradient_checkpointing = False
            config.per_device_train_batch_size = max(1, config.per_device_train_batch_size // 4)
            config.per_device_eval_batch_size = max(1, config.per_device_eval_batch_size // 4)
            return config

        gpu_name = torch.cuda.get_device_name(0).lower()
        total_mem = get_gpu_memory()
        logger.info("Detected GPU: %s | VRAM: %s MB", gpu_name, total_mem)

        config.device = "cuda"

        if "t4" in gpu_name:
            logger.info("T4 GPU detected — applying T4 optimizations.")
            config.bf16 = True
            config.fp16 = False
            config.per_device_train_batch_size = min(4, config.per_device_train_batch_size)
            config.gradient_accumulation_steps = max(4, config.gradient_accumulation_steps)
            config.qlora_bits = 4
            config.peft_method = "qlora"

        elif "l4" in gpu_name:
            logger.info("L4 GPU detected — applying L4 optimizations.")
            config.bf16 = True
            config.fp16 = False
            config.per_device_train_batch_size = min(8, config.per_device_train_batch_size)
            config.qlora_bits = 4
            config.peft_method = "qlora"

        elif "a100" in gpu_name:
            logger.info("A100 GPU detected — applying A100 optimizations.")
            config.bf16 = True
            config.fp16 = False
            config.tf32 = True
            config.per_device_train_batch_size = min(16, config.per_device_train_batch_size)

        elif "v100" in gpu_name:
            logger.info("V100 GPU detected — applying V100 optimizations.")
            config.fp16 = True
            config.bf16 = False
            config.per_device_train_batch_size = min(8, config.per_device_train_batch_size)

        elif "a10" in gpu_name or "a16" in gpu_name:
            logger.info("A-series GPU detected.")
            config.bf16 = True
            config.per_device_train_batch_size = min(8, config.per_device_train_batch_size)

        elif "p100" in gpu_name:
            logger.info("P100 GPU detected.")
            config.fp16 = True
            config.per_device_train_batch_size = min(8, config.per_device_train_batch_size)

        elif "k80" in gpu_name:
            logger.info("K80 GPU detected — very limited VRAM.")
            config.fp16 = True
            config.use_peft = True
            config.peft_method = "qlora"
            config.qlora_bits = 4
            config.per_device_train_batch_size = 1
            config.gradient_accumulation_steps = max(8, config.gradient_accumulation_steps)
            config.max_seq_length = min(1024, config.max_seq_length)

        else:
            logger.info("Unknown GPU '%s'. Using conservative defaults.", gpu_name)
            if total_mem and total_mem < 8000:
                config.peft_method = "qlora"
                config.per_device_train_batch_size = 1
                config.gradient_accumulation_steps = max(4, config.gradient_accumulation_steps)

        if total_mem and total_mem < 12000:
            config.max_seq_length = min(1024, config.max_seq_length)
            logger.info("VRAM <12 GB: limiting seq_length to %d", config.max_seq_length)

        return config

    def _configure_colab_drive(self, config: TrainingConfig) -> TrainingConfig:
        if not is_colab():
            return config

        try:
            from google.colab import drive

            if not os.path.exists("/content/drive"):
                logger.info("Mounting Google Drive...")
                drive.mount("/content/drive")
                logger.info("Google Drive mounted successfully.")

            project_dir = config.gdrive_output_dir or f"MyDrive/{config.project_name}"
            gdrive_path = os.path.join("/content/drive", project_dir)
            os.makedirs(gdrive_path, exist_ok=True)

            config.output_dir = gdrive_path
            logger.info("Checkpoints will be saved to: %s", gdrive_path)

        except ImportError:
            logger.warning("google.colab not available; skipping Drive mount.")
        except Exception as e:
            logger.warning("Failed to mount Google Drive: %s", e)

        return config

    def _configure_memory_optimizations(self, config: TrainingConfig) -> TrainingConfig:
        if not torch.cuda.is_available():
            return config

        total_mem = get_gpu_memory() or 16000

        if total_mem < 12000:
            config.gradient_checkpointing = True
            config.gradient_accumulation_steps = max(
                config.gradient_accumulation_steps, 4
            )
            config.per_device_train_batch_size = min(2, config.per_device_train_batch_size)
            logger.info("Low VRAM (%d MB): applied aggressive memory optimizations.", total_mem)

        torch.backends.cudnn.benchmark = True

        if config.tf32 and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled.")
            except AttributeError:
                pass

        return config

    def train(
        self,
        resume_last_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        if resume_last_checkpoint and not self.config.resume_from_checkpoint:
            last_ckpt = self._find_last_checkpoint()
            if last_ckpt:
                logger.info("Auto-resuming from checkpoint: %s", last_ckpt)
                self.config.resume_from_checkpoint = last_ckpt
                self._session_recovered = True

        if not self._session_recovered:
            self._install_missing_deps()

        self._log_environment_summary()

        return super().train()

    def _find_last_checkpoint(self) -> Optional[str]:
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return None

        checkpoint_manager = CheckpointManager(
            output_dir=ckpt_dir,
            save_total_limit=self.config.save_total_limit,
        )
        best = checkpoint_manager.get_best_checkpoint()
        if best:
            return best

        latest = checkpoint_manager.get_latest_checkpoint()
        return latest

    def _install_missing_deps(self) -> None:
        required = [
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "bitsandbytes",
            "trl",
            "torch",
        ]
        for pkg in required:
            try:
                importlib.import_module(pkg)
            except ImportError:
                logger.info("Installing missing dependency: %s", pkg)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", pkg]
                )

    def _log_environment_summary(self) -> None:
        lines = [
            "=" * 56,
            "Environment Summary",
            "=" * 56,
            f"Python:           {sys.version}",
            f"Platform:         {sys.platform}",
            f"Colab:            {is_colab()}",
            f"Kaggle:           {is_kaggle()}",
            f"CUDA available:   {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            lines += [
                f"GPU:              {torch.cuda.get_device_name(0)}",
                f"VRAM:             {get_gpu_memory()} MB",
                f"CUDA version:     {torch.version.cuda}",
                f"cuDNN version:    {torch.backends.cudnn.version()}",
            ]
        lines += [
            f"Torch version:    {torch.__version__}",
            f"Model:            {self.config.model_name}",
            f"PEFT:             {self.config.use_peft} ({self.config.peft_method})",
            f"Mixed precision:  {'bf16' if self.config.bf16 else 'fp16' if self.config.fp16 else 'fp32'}",
            f"Batch size:       {self.config.per_device_train_batch_size}",
            f"Grad accum:       {self.config.gradient_accumulation_steps}",
            f"Max seq length:   {self.config.max_seq_length}",
            f"Output dir:       {self.config.output_dir}",
            "=" * 56,
        ]
        for line in lines:
            logger.info(line)

        print("\n" + "\n".join(lines) + "\n")

    def clear_gpu_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared.")

    def get_training_loop_progress(self) -> Dict[str, Any]:
        return {
            "current_step": self.metrics.global_step,
            "total_steps": self._total_steps,
            "current_epoch": self.metrics.epoch,
            "total_epochs": self.config.num_train_epochs,
            "loss": self.metrics.get_average_loss(),
            "elapsed_seconds": self.metrics.get_elapsed_time(),
            "throughput_tokens_per_sec": self.metrics.get_throughput(),
            "best_eval_loss": self.metrics.best_eval_loss,
        }
