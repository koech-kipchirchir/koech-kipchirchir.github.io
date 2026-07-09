import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsTracker:
    start_time: float = field(default_factory=time.time)
    step: int = 0
    epoch: int = 0
    global_step: int = 0

    total_loss: float = 0.0
    running_loss: float = 0.0
    running_samples: int = 0
    running_tokens: int = 0

    _loss_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Eval
    best_eval_loss: float = float("inf")
    best_eval_global_step: int = 0

    def reset_running(self) -> None:
        self.running_loss = 0.0
        self.running_samples = 0
        self.running_tokens = 0

    def update(
        self,
        loss: float,
        batch_size: int,
        num_tokens: int,
        lr: float,
    ) -> dict:
        self.step += 1
        self.global_step += 1
        self.total_loss += loss
        self.running_loss += loss
        self.running_samples += batch_size
        self.running_tokens += num_tokens
        self._loss_history.append(loss)

        return {
            "loss": loss,
            "learning_rate": lr,
            "step": self.global_step,
            "epoch": self.epoch,
        }

    def get_average_loss(self) -> float:
        if self.running_samples == 0:
            return 0.0
        return self.running_loss / self.running_samples

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time

    def get_throughput(self) -> float:
        elapsed = self.get_elapsed_time()
        if elapsed < 1e-6:
            return 0.0
        return self.running_tokens / elapsed

    def state_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "step": self.step,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "total_loss": self.total_loss,
            "best_eval_loss": self.best_eval_loss,
            "best_eval_global_step": self.best_eval_global_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.start_time = state.get("start_time", time.time())
        self.step = state.get("step", 0)
        self.epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)
        self.total_loss = state.get("total_loss", 0.0)
        self.best_eval_loss = state.get("best_eval_loss", float("inf"))
        self.best_eval_global_step = state.get("best_eval_global_step", 0)
