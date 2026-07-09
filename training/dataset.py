import os
import random
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, interleave_datasets
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling

from .config import TrainingConfig
from .utils import get_logger

logger = get_logger(__name__)


class DatasetManager:
    def __init__(self, config: TrainingConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        dataset = self._load_raw()

        is_iterable = isinstance(dataset, IterableDataset)
        if not is_iterable:
            dataset = self._split_val(dataset)

        if self.config.packing:
            logger.warning("Packing enabled; tokenization happens inside the train loop.")
        else:
            dataset = dataset.map(
                self._tokenize,
                batched=True,
                remove_columns=self._get_remove_columns(dataset) if not is_iterable else None,
                num_proc=self.config.dataloader_num_workers or None,
                desc="Tokenizing",
            )

        if not is_iterable and self.config.dataset_val_split not in (None, 0):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("test")
        else:
            train_dataset = dataset if isinstance(dataset, Dataset) else dataset
            eval_dataset = None

        logger.info(
            "Train samples: %s  |  Eval samples: %s",
            len(train_dataset) if not is_iterable else "?",
            len(eval_dataset) if eval_dataset is not None and not is_iterable else "N/A",
        )
        return train_dataset, eval_dataset

    def _load_raw(self) -> Dataset:
        config = self.config

        if config.dataset_name and os.path.isdir(config.dataset_name):
            logger.info("Loading local dataset from '%s'", config.dataset_name)
            return load_dataset(
                config.dataset_name,
                split=config.dataset_split,
                trust_remote_code=config.trust_remote_code,
            )

        if config.dataset_name:
            logger.info("Loading HuggingFace dataset '%s'", config.dataset_name)
            return load_dataset(
                config.dataset_name,
                config.dataset_config,
                split=config.dataset_split,
                trust_remote_code=config.trust_remote_code,
            )

        raise ValueError(
            "No dataset specified. Set dataset_name in config to a HuggingFace "
            "dataset identifier or a local path."
        )

    def _split_val(self, dataset: Dataset) -> DatasetDict:
        val_split = self.config.dataset_val_split
        if val_split and 0 < val_split < 1:
            split = dataset.train_test_split(test_size=val_split, seed=self.config.seed)
            return DatasetDict({"train": split["train"], "test": split["test"]})
        return DatasetDict({"train": dataset})

    def _tokenize(self, examples):
        return self.tokenizer(
            examples[self.config.dataset_text_field],
            truncation=True,
            padding=False,
            max_length=self.config.max_seq_length,
        )

    def _get_remove_columns(self, dataset: Dataset) -> list:
        if isinstance(dataset, DatasetDict):
            return list(dataset["train"].column_names)
        return list(dataset.column_names)

    def get_data_collator(self) -> DataCollatorForLanguageModeling:
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def build_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and not isinstance(dataset, IterableDataset),
            collate_fn=self.get_data_collator(),
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )
