from typing import Iterator

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel

from .config import TrainingConfig
from .utils import get_logger

logger = get_logger(__name__)


class OptimizerManager:
    def __init__(self, config: TrainingConfig, model: PreTrainedModel) -> None:
        self.config = config
        self.model = model

    def _get_param_groups(self) -> list:
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return groups

    def build(self) -> AdamW:
        groups = self._get_param_groups()
        optimizer = AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        logger.info(
            "Optimizer: AdamW (lr=%.2e, weight_decay=%.2e, groups=%d)",
            self.config.learning_rate,
            self.config.weight_decay,
            len(groups),
        )
        return optimizer
