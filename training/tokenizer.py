from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from .config import TrainingConfig
from .utils import get_logger

logger = get_logger(__name__)


class TokenizerManager:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def load(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        logger.info(
            "Loaded tokenizer '%s' (vocab_size=%d, pad_token='%s')",
            self.config.model_name,
            len(tokenizer),
            tokenizer.pad_token,
        )
        return tokenizer

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples[self.config.dataset_text_field],
            truncation=True,
            padding=False,
            max_length=self.config.max_seq_length,
        )
