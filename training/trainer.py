import os
import time
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training.config import TrainingConfig
from training.dataset import DatasetManager
from training.optimizer import OptimizerManager
from training.scheduler import SchedulerManager
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker
from training.logger import LoggerManager
from training.callbacks import CallbackManager, Callback, EarlyStoppingCallback, ProgressCallback
from training.utils import get_logger, detect_device, set_seed

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = detect_device()

        set_seed(config.seed)

        self.metrics = MetricsTracker()
        self.callback_manager = CallbackManager()

        self._model_loaded_externally = model is not None
        self._total_steps: int = 0

    def _prepare(self) -> None:
        if self.model is None:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16 if self.config.fp16 else torch.float32,
            )

        if self.tokenizer is None:
            from training.tokenizer import TokenizerManager

            self.tokenizer = TokenizerManager(self.config).load()

        self.model = self.model.to(self.device)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            self.model.config.use_cache = False

        if self.config.use_peft:
            self._apply_peft()

        dataset_manager = DatasetManager(self.config, self.tokenizer)
        train_dataset, eval_dataset = dataset_manager.load()

        total_train_samples = len(train_dataset) if train_dataset else 0
        self._total_steps = SchedulerManager.build_num_training_steps(
            self.config, max(1, total_train_samples)
        )

        self.train_loader = dataset_manager.build_dataloader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
        )

        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = dataset_manager.build_dataloader(
                eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
            )

        self.optimizer = OptimizerManager(self.config, self.model).build()
        self.scheduler = SchedulerManager(self.config, self.optimizer).build(
            self._total_steps
        )

        self.checkpoint_manager = CheckpointManager(
            output_dir=os.path.join(self.config.output_dir, "checkpoints"),
            save_total_limit=self.config.save_total_limit,
        )

        self.logger_manager = LoggerManager(
            log_dir=os.path.join(self.config.output_dir, "logs"),
            backends=self.config.report_to,
        )

        self.callback_manager.add(ProgressCallback(total_steps=self._total_steps))
        self.logger_manager.log_hyperparams(self.config.__dict__)

        if self.config.resume_from_checkpoint:
            self._resume()

    def _apply_peft(self) -> None:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        if self.config.peft_method == "qlora":
            self.model = prepare_model_for_kbit_training(self.model)

        target_modules = self.config.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]

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

    def _resume(self) -> None:
        ckpt_path = self.config.resume_from_checkpoint
        if not os.path.isdir(ckpt_path):
            logger.warning("Resume path not found: %s", ckpt_path)
            return

        state = self.checkpoint_manager.load(
            checkpoint_path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=self.metrics,
        )
        logger.info("Resumed from checkpoint: %s (step %d)", ckpt_path, self.metrics.global_step)

    def train(self) -> Dict[str, Any]:
        self._prepare()

        logger.info(
            "Starting training: %d steps, %d epochs, %d batches/epoch",
            self._total_steps,
            self.config.num_train_epochs,
            len(self.train_loader),
        )

        state = {"should_stop": False}
        self.callback_manager.on_train_begin(state)

        self.model.train()
        self.metrics.epoch = 0

        for epoch in range(self.config.num_train_epochs):
            if state.get("should_stop"):
                logger.info("Training stopped early at epoch %d", epoch)
                break

            self.metrics.epoch = epoch
            self.metrics.reset_running()
            state["epoch"] = epoch
            self.callback_manager.on_epoch_begin(state)

            for batch in self.train_loader:
                if state.get("should_stop") or self.metrics.global_step >= self._total_steps:
                    break

                self.callback_manager.on_step_begin(state)
                loss = self._training_step(batch)
                self.callback_manager.on_step_end(state)

                state["loss"] = loss
                state["global_step"] = self.metrics.global_step
                state["learning_rate"] = self._get_lr()

                self._maybe_log(state)
                self._maybe_evaluate(state)
                self._maybe_save(state)

            self.callback_manager.on_epoch_end(state)

        self.callback_manager.on_train_end(state)
        self.logger_manager.close()

        if self.config.load_best_model_at_end:
            self._load_best_model()

        return {
            "global_step": self.metrics.global_step,
            "total_loss": self.metrics.total_loss,
            "best_eval_loss": self.metrics.best_eval_loss,
            "best_eval_global_step": self.metrics.best_eval_global_step,
            "elapsed_seconds": self.metrics.get_elapsed_time(),
        }

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        loss = outputs.loss / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.metrics.step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        loss_val = loss.item() * self.config.gradient_accumulation_steps
        batch_size = batch["input_ids"].size(0)
        num_tokens = batch["input_ids"].numel()

        self.metrics.update(
            loss=loss_val,
            batch_size=batch_size,
            num_tokens=num_tokens,
            lr=self._get_lr(),
        )

        return loss_val

    def _evaluate(self) -> Dict[str, float]:
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        self.model.train()

        metrics = {"eval_loss": avg_loss}

        if avg_loss < self.metrics.best_eval_loss:
            self.metrics.best_eval_loss = avg_loss
            self.metrics.best_eval_global_step = self.metrics.global_step

        return metrics

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _maybe_log(self, state: Dict[str, Any]) -> None:
        if self.config.logging_strategy == "no":
            return
        if self.config.logging_strategy == "steps" and (
            self.metrics.global_step % self.config.logging_steps != 0
        ):
            return

        log_metrics = {
            "loss": state.get("loss", 0),
            "learning_rate": state.get("learning_rate", 0),
            "epoch": self.metrics.epoch,
            "global_step": self.metrics.global_step,
            "throughput_tok_s": self.metrics.get_throughput(),
        }
        self.logger_manager.log_metrics(log_metrics, self.metrics.global_step)
        logger.info(
            "Step %d | Loss: %.4f | LR: %.2e | Tok/s: %.1f",
            self.metrics.global_step,
            state.get("loss", 0),
            state.get("learning_rate", 0),
            self.metrics.get_throughput(),
        )

    def _maybe_evaluate(self, state: Dict[str, Any]) -> None:
        if self.config.evaluation_strategy == "no":
            return
        if self.config.evaluation_strategy == "steps" and (
            self.metrics.global_step % self.config.eval_steps != 0
        ):
            return

        self.callback_manager.on_evaluate_begin(state)
        eval_metrics = self._evaluate()
        state.update(eval_metrics)
        self.callback_manager.on_evaluate_end(state)

        if eval_metrics:
            self.logger_manager.log_metrics(eval_metrics, self.metrics.global_step)
            logger.info(
                "Evaluation at step %d | Eval loss: %.4f (best: %.4f at step %d)",
                self.metrics.global_step,
                eval_metrics.get("eval_loss", 0),
                self.metrics.best_eval_loss,
                self.metrics.best_eval_global_step,
            )

    def _maybe_save(self, state: Dict[str, Any]) -> None:
        if self.config.save_strategy == "no":
            return
        if self.config.save_strategy == "steps" and (
            self.metrics.global_step % self.config.save_steps != 0
        ):
            return

        is_best = self.metrics.best_eval_global_step == self.metrics.global_step
        self.checkpoint_manager.save(
            global_step=self.metrics.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=self.metrics,
            config=self.config.__dict__,
            is_best=is_best,
        )
        self.callback_manager.on_save_checkpoint(state)

    def _load_best_model(self) -> None:
        best_path = os.path.join(self.config.output_dir, "checkpoints", "best_model")
        if os.path.isdir(best_path):
            from peft import PeftModel

            if isinstance(self.model, PeftModel):
                self.model = self.model.merge_and_unload()
            logger.info("Loaded best model from %s", best_path)
