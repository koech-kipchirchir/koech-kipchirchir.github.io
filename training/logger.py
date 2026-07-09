import os
from typing import Optional, Dict, Any, List

from torch.utils.tensorboard import SummaryWriter

from training.utils import get_logger

logger = get_logger(__name__)


class LoggerManager:
    def __init__(self, log_dir: str, backends: Optional[List[str]] = None) -> None:
        self.log_dir = log_dir
        self.backends = backends or ["tensorboard"]
        self.tb_writer: Optional[SummaryWriter] = None
        self.wandb_run = None

        os.makedirs(log_dir, exist_ok=True)

        self._init_backends()

    def _init_backends(self) -> None:
        for backend in self.backends:
            if backend == "tensorboard":
                try:
                    self.tb_writer = SummaryWriter(log_dir=self.log_dir)
                    logger.info("TensorBoard writer initialized: %s", self.log_dir)
                except Exception as e:
                    logger.warning("Failed to init TensorBoard: %s", e)

            elif backend == "wandb":
                try:
                    import wandb as wb

                    self.wandb_run = wb.init(
                        project=os.environ.get("WANDB_PROJECT", "aios-trainer"),
                        dir=self.log_dir,
                        sync_tensorboard=True,
                    )
                    logger.info("WandB initialized: %s", self.wandb_run)
                except ImportError:
                    logger.warning("wandb not installed; skipping.")
                except Exception as e:
                    logger.warning("Failed to init WandB: %s", e)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
            self.tb_writer.flush()

        if self.wandb_run:
            try:
                import wandb as wb

                wb.log(metrics, step=step)
            except Exception:
                pass

    def log_hyperparams(self, hparams: Dict[str, Any]) -> None:
        if self.tb_writer:
            try:
                self.tb_writer.add_hparams(
                    {k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))},
                    {"dummy": 0},
                )
            except Exception as e:
                logger.debug("Could not log hparams to TensorBoard: %s", e)

        if self.wandb_run:
            try:
                import wandb as wb

                wb.config.update(hparams)
            except Exception:
                pass

    def log_figure(self, tag: str, figure, step: int) -> None:
        if self.tb_writer:
            self.tb_writer.add_figure(tag, figure, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)

    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()
            logger.info("TensorBoard writer closed.")

        if self.wandb_run:
            try:
                import wandb as wb

                wb.finish()
                logger.info("WandB run finished.")
            except Exception:
                pass
