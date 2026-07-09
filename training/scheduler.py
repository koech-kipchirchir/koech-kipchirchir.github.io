import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import TrainingConfig
from .utils import get_logger

logger = get_logger(__name__)


class SchedulerManager:
    def __init__(self, config: TrainingConfig, optimizer: Optimizer) -> None:
        self.config = config
        self.optimizer = optimizer

    def build(self, num_training_steps: int) -> LRScheduler:
        scheduler_type = self.config.lr_scheduler_type
        warmup_steps = int(self.config.warmup_ratio * num_training_steps)

        if scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        elif scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        elif scheduler_type == "cosine_with_restarts":
            from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        elif scheduler_type == "constant":
            from transformers import get_constant_schedule

            scheduler = get_constant_schedule(self.optimizer)

        elif scheduler_type == "constant_with_warmup":
            from transformers import get_constant_schedule_with_warmup

            scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps
            )

        elif scheduler_type == "polynomial":
            from transformers import get_polynomial_decay_schedule_with_warmup

            scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        else:
            raise ValueError(f"Unsupported LR scheduler: {scheduler_type}")

        logger.info(
            "Scheduler: %s (warmup_steps=%d, total_steps=%d)",
            scheduler_type,
            warmup_steps,
            num_training_steps,
        )
        return scheduler

    @staticmethod
    def build_num_training_steps(
        config: TrainingConfig, num_train_samples: int
    ) -> int:
        if config.max_steps > 0:
            return config.max_steps

        steps_per_epoch = (
            num_train_samples
            // (config.per_device_train_batch_size * config.world_size)
            // config.gradient_accumulation_steps
        )
        num_training_steps = steps_per_epoch * config.num_train_epochs
        return max(1, num_training_steps)
