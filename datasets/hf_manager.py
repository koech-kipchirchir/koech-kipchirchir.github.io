import hashlib
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, Iterator, Union

import datasets
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    load_dataset,
    concatenate_datasets,
    interleave_datasets,
    get_dataset_config_names,
    get_dataset_split_names,
)

from training.config import TrainingConfig
from training.utils import get_logger

logger = get_logger(__name__)


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "description": "FineWeb educational web dataset",
        "configs": ["default", "sample-10BT"],
    },
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "description": "200k chat conversations",
        "configs": None,
    },
    "openhermes": {
        "path": "teknium/openhermes",
        "description": "OpenHermes 2.5 dataset",
        "configs": None,
    },
    "dolly": {
        "path": "databricks/databricks-dolly-15k",
        "description": "Databricks Dolly 15k instruct dataset",
        "configs": None,
    },
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "description": "Stanford Alpaca dataset",
        "configs": None,
    },
    "openwebtext": {
        "path": "openwebtext",
        "description": "OpenWebText corpus",
        "configs": None,
    },
    "redpajama": {
        "path": "togethercomputer/RedPajama-Data-1T",
        "description": "RedPajama 1T token dataset",
        "configs": None,
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "description": "GSM8K math reasoning",
        "configs": ["main", "socratic"],
    },
    "humaneval": {
        "path": "openai/humaneval",
        "description": "HumanEval code generation benchmark",
        "configs": None,
    },
    "metamathqa": {
        "path": "meta-math/MetaMathQA",
        "description": "MetaMathQA math dataset",
        "configs": None,
    },
}


class HFManager:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self._loaded: Dict[str, Dataset] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    #  Loading
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        streaming: Optional[bool] = None,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, IterableDataset]:
        info = DATASET_REGISTRY.get(name)
        if info is None:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
            )

        path = info["path"]
        s = streaming if streaming is not None else self.streaming

        logger.info(
            "Loading '%s' (path=%s, config=%s, split=%s, streaming=%s)",
            name, path, config, split, s,
        )

        dataset = load_dataset(
            path,
            config,
            split=split,
            streaming=s,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            **kwargs,
        )

        self._loaded[name] = dataset
        self._metadata[name] = {
            "name": name,
            "path": path,
            "config": config,
            "split": split,
            "streaming": s,
            "loaded_at": datetime.utcnow().isoformat(),
            "num_rows": len(dataset) if not s else None,
        }

        if not s and isinstance(dataset, Dataset):
            logger.info("Loaded %s rows from '%s'", len(dataset), name)

        return dataset

    def load_multiple(
        self,
        names: List[str],
        strategy: str = "concatenate",
        **kwargs,
    ) -> Dataset:
        datasets_list = []
        for name in names:
            ds = self.load(name, **kwargs)
            if isinstance(ds, DatasetDict):
                ds = ds.get("train", next(iter(ds.values())))
            if isinstance(ds, Dataset) or isinstance(ds, IterableDataset):
                datasets_list.append(ds)
            else:
                logger.warning("Skipping '%s': unexpected type %s", name, type(ds))

        if not datasets_list:
            raise ValueError("No datasets were loaded.")

        if strategy == "concatenate":
            if any(isinstance(d, IterableDataset) for d in datasets_list):
                logger.warning("concatenate not supported for streaming; using interleave.")
                return interleave_datasets(datasets_list)
            return concatenate_datasets(datasets_list)

        elif strategy == "interleave":
            return interleave_datasets(datasets_list)

        raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    #  Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        dataset: Dataset,
        condition: Callable[[Dict], bool],
        num_proc: Optional[int] = None,
    ) -> Dataset:
        original_len = len(dataset)
        filtered = dataset.filter(condition, num_proc=num_proc)
        new_len = len(filtered)
        logger.info(
            "Filtered %s -> %s rows (removed %s)",
            original_len, new_len, original_len - new_len,
        )
        return filtered

    def filter_by_length(
        self,
        dataset: Dataset,
        text_field: str = "text",
        min_length: int = 50,
        max_length: int = 8192,
    ) -> Dataset:
        def condition(example):
            text = example.get(text_field, "")
            return min_length <= len(text) <= max_length

        return self.filter(dataset, condition)

    def filter_by_language(
        self,
        dataset: Dataset,
        lang: str = "en",
        lang_field: Optional[str] = "language",
    ) -> Dataset:
        if lang_field is None:
            logger.warning("No language field available; skipping filter.")
            return dataset

        def condition(example):
            return example.get(lang_field) == lang

        return self.filter(dataset, condition)

    # ------------------------------------------------------------------
    #  Deduplication
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        dataset: Dataset,
        text_field: str = "text",
        hash_func: Optional[Callable] = None,
    ) -> Dataset:
        if hash_func is None:
            hash_func = lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()

        seen: set = set()
        indices_to_keep = []

        for i, example in enumerate(dataset):
            text = example.get(text_field, "")
            h = hash_func(text)
            if h not in seen:
                seen.add(h)
                indices_to_keep.append(i)

        deduped = dataset.select(indices_to_keep)
        logger.info(
            "Deduplicated %s -> %s rows (removed %s duplicates)",
            len(dataset), len(deduped), len(dataset) - len(deduped),
        )
        return deduped

    # ------------------------------------------------------------------
    #  Cleaning
    # ------------------------------------------------------------------

    def clean_text(
        self,
        dataset: Dataset,
        text_field: str = "text",
        num_proc: Optional[int] = None,
    ) -> Dataset:
        import re

        def clean(example):
            text = example.get(text_field, "")
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
            example[text_field] = text
            return example

        return dataset.map(clean, num_proc=num_proc)

    def remove_empty(
        self,
        dataset: Dataset,
        text_field: str = "text",
    ) -> Dataset:
        def condition(example):
            text = example.get(text_field, "")
            return bool(text and text.strip())

        return self.filter(dataset, condition)

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------

    def export_jsonl(
        self,
        dataset: Dataset,
        output_path: str,
        batch_size: int = 1000,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for batch in dataset.iter(batch_size=batch_size):
                for example in batch:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
        logger.info("Exported %s rows to %s", count, output_path)
        return output_path

    def export_parquet(
        self,
        dataset: Dataset,
        output_path: str,
        batch_size: int = 10000,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        dataset.to_parquet(output_path, batch_size=batch_size)
        logger.info("Exported to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    #  Statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        dataset: Dataset,
        text_field: str = "text",
        sample_size: Optional[int] = 10000,
    ) -> Dict[str, Any]:
        if sample_size and len(dataset) > sample_size:
            ds = dataset.shuffle(seed=42).select(range(sample_size))
        else:
            ds = dataset

        lengths = []
        total_chars = 0

        for example in ds:
            text = example.get(text_field, "")
            lengths.append(len(text))
            total_chars += len(text)

        if not lengths:
            return {"error": "empty dataset"}

        lengths.sort()
        n = len(lengths)
        stats = {
            "num_samples": n,
            "total_chars": total_chars,
            "mean_length": sum(lengths) / n,
            "median_length": lengths[n // 2],
            "min_length": lengths[0],
            "max_length": lengths[-1],
            "p10_length": lengths[int(n * 0.1)],
            "p90_length": lengths[int(n * 0.9)],
            "p99_length": lengths[int(n * 0.99)],
        }

        if isinstance(ds, Dataset):
            stats["num_columns"] = len(ds.column_names)
            stats["columns"] = ds.column_names

        logger.info("Dataset statistics: %s", json.dumps(stats, indent=2))
        return stats

    # ------------------------------------------------------------------
    #  Versioning
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        dataset: Dataset,
        name: str,
        version: str,
        base_dir: str = "./dataset_snapshots",
        format: str = "parquet",
    ) -> str:
        snapshot_dir = Path(base_dir) / name / version
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "name": name,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "num_rows": len(dataset),
            "columns": dataset.column_names,
            "format": format,
        }

        if format == "parquet":
            out_path = str(snapshot_dir / "data.parquet")
            self.export_parquet(dataset, out_path)
        elif format == "jsonl":
            out_path = str(snapshot_dir / "data.jsonl")
            self.export_jsonl(dataset, out_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(snapshot_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Snapshot saved: %s", snapshot_dir)
        return str(snapshot_dir)

    def load_snapshot(self, path: str) -> Dataset:
        path = Path(path)
        manifest_path = path / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        fmt = manifest.get("format", "parquet")
        data_path = path / f"data.{fmt}"

        if not data_path.exists():
            raise FileNotFoundError(f"No data file found at {data_path}")

        if fmt == "parquet":
            dataset = Dataset.from_parquet(str(data_path))
        elif fmt == "jsonl":
            dataset = Dataset.from_json(str(data_path))
        else:
            raise ValueError(f"Unknown format: {fmt}")

        logger.info("Loaded snapshot '%s' v%s (%s rows)", manifest["name"], manifest["version"], len(dataset))
        return dataset

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def list_available(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "path": info["path"],
                "description": info["description"],
                "configs": info.get("configs"),
            }
            for name, info in DATASET_REGISTRY.items()
        ]

    def get_loaded_info(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._metadata)

    def estimate_token_count(
        self,
        dataset: Dataset,
        text_field: str = "text",
        tokens_per_char: float = 0.25,
    ) -> int:
        total_chars = sum(len(example.get(text_field, "")) for example in dataset)
        estimated_tokens = int(total_chars * tokens_per_char)
        logger.info("Estimated token count: %s", f"{estimated_tokens:,}")
        return estimated_tokens

    def shard(
        self,
        dataset: Dataset,
        num_shards: int,
        output_dir: str,
        format: str = "parquet",
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for i in range(num_shards):
            shard = dataset.shard(num_shards, i)
            ext = "parquet" if format == "parquet" else "jsonl"
            out_path = os.path.join(output_dir, f"shard-{i:05d}.{ext}")
            if format == "parquet":
                shard.to_parquet(out_path)
            else:
                self.export_jsonl(shard, out_path)
            paths.append(out_path)

        logger.info("Sharded into %d files under %s", num_shards, output_dir)
        return paths
