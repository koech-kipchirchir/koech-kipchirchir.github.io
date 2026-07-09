import glob
import os
import re
import json
import shutil
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch

from training.utils import get_logger

logger = get_logger(__name__)


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


@dataclass
class CheckpointManager:
    output_dir: str
    save_total_limit: int = 3
    _best_metric: float = field(default=float("inf"), init=False)

    def save(
        self,
        global_step: int,
        model,
        optimizer=None,
        scheduler=None,
        metrics=None,
        config=None,
        is_best: bool = False,
    ) -> str:
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
        else:
            logger.warning("Model has no save_pretrained method; saving state_dict.")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

        training_state = {
            "global_step": global_step,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metrics": metrics.state_dict() if metrics else None,
            "config": config,
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

        if is_best:
            best_path = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_path, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(best_path)
            with open(os.path.join(best_path, "best_global_step.txt"), "w") as f:
                f.write(str(global_step))
            logger.info("Best model saved to %s (step %d)", best_path, global_step)

        self._prune_old_checkpoints()

        logger.info("Checkpoint saved at step %d: %s", global_step, checkpoint_dir)
        return checkpoint_dir

    def load(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        metrics=None,
    ) -> Dict[str, Any]:
        ckpt_path = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Training state not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
            logger.info("Optimizer state loaded.")
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
            logger.info("Scheduler state loaded.")
        if metrics and state.get("metrics"):
            metrics.load_state_dict(state["metrics"])
            logger.info("Metrics state loaded.")

        logger.info("Checkpoint loaded from %s (step %d)", checkpoint_path, state.get("global_step", 0))
        return state

    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = self._sorted_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def get_best_checkpoint(self) -> Optional[str]:
        best_file = os.path.join(self.output_dir, "best_model", "best_global_step.txt")
        if os.path.isfile(best_file):
            with open(best_file) as f:
                step = f.read().strip()
            ckpt = os.path.join(self.output_dir, f"checkpoint-{step}")
            if os.path.isdir(ckpt):
                return ckpt
        return None

    def list_checkpoints(self) -> List[str]:
        return self._sorted_checkpoints()

    def _sorted_checkpoints(self) -> List[str]:
        if not os.path.isdir(self.output_dir):
            return []
        checkpoints = []
        for entry in os.listdir(self.output_dir):
            m = CHECKPOINT_PATTERN.match(entry)
            if m and os.path.isdir(os.path.join(self.output_dir, entry)):
                checkpoints.append((int(m.group(1)), os.path.join(self.output_dir, entry)))
        checkpoints.sort(key=lambda x: x[0])
        return [c[1] for c in checkpoints]

    def _prune_old_checkpoints(self) -> None:
        if self.save_total_limit <= 0:
            return
        checkpoints = self._sorted_checkpoints()
        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
            logger.info("Pruned old checkpoint: %s", oldest)
