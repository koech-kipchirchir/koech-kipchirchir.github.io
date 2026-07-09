from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Callback(ABC):
    @abstractmethod
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_train_end(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_epoch_begin(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_step_begin(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_step_end(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_evaluate_begin(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_evaluate_end(self, state: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def on_save_checkpoint(self, state: Dict[str, Any]) -> None:
        ...


class CallbackManager:
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        self.callbacks.remove(callback)

    def _invoke(self, method: str, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            getattr(cb, method)(state)

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        self._invoke("on_train_begin", state)

    def on_train_end(self, state: Dict[str, Any]) -> None:
        self._invoke("on_train_end", state)

    def on_epoch_begin(self, state: Dict[str, Any]) -> None:
        self._invoke("on_epoch_begin", state)

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        self._invoke("on_epoch_end", state)

    def on_step_begin(self, state: Dict[str, Any]) -> None:
        self._invoke("on_step_begin", state)

    def on_step_end(self, state: Dict[str, Any]) -> None:
        self._invoke("on_step_end", state)

    def on_evaluate_begin(self, state: Dict[str, Any]) -> None:
        self._invoke("on_evaluate_begin", state)

    def on_evaluate_end(self, state: Dict[str, Any]) -> None:
        self._invoke("on_evaluate_end", state)

    def on_save_checkpoint(self, state: Dict[str, Any]) -> None:
        self._invoke("on_save_checkpoint", state)


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._counter = 0

    def on_evaluate_end(self, state: Dict[str, Any]) -> None:
        loss = state.get("eval_loss")
        if loss is None:
            return
        if loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                state["should_stop"] = True

    def on_train_begin(self, state: Dict[str, Any]) -> None: ...
    def on_train_end(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_begin(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: Dict[str, Any]) -> None: ...
    def on_step_begin(self, state: Dict[str, Any]) -> None: ...
    def on_step_end(self, state: Dict[str, Any]) -> None: ...
    def on_evaluate_begin(self, state: Dict[str, Any]) -> None: ...
    def on_save_checkpoint(self, state: Dict[str, Any]) -> None: ...


class ProgressCallback(Callback):
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps

    def on_step_end(self, state: Dict[str, Any]) -> None:
        from training.utils import get_logger

        log = get_logger(__name__)
        step = state.get("global_step", 0)
        loss = state.get("loss")
        lr = state.get("learning_rate")
        if step % max(1, self.total_steps // 20) == 0:
            log.info(
                "Step %d/%d | Loss: %.4f | LR: %.2e",
                step, self.total_steps, loss, lr,
            )

    def on_train_begin(self, state: Dict[str, Any]) -> None: ...
    def on_train_end(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_begin(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: Dict[str, Any]) -> None: ...
    def on_step_begin(self, state: Dict[str, Any]) -> None: ...
    def on_evaluate_begin(self, state: Dict[str, Any]) -> None: ...
    def on_evaluate_end(self, state: Dict[str, Any]) -> None: ...
    def on_save_checkpoint(self, state: Dict[str, Any]) -> None: ...
