"""
AIOS Fine-tuning Dataset Module

Loads chat-format datasets from JSONL files, validates every sample, removes
invalid samples, shuffles, splits into train and validation, tokenizes using
a HuggingFace ``AutoTokenizer``, and optionally packs sequences.

Supported formats:
    * **ShareGPT** — ``{"conversations": [{"from": "human"/"gpt", "value": …}]}``
    * **OpenAI messages** — ``{"messages": [{"role": "user"/"assistant", "content": …}]}``
    * **Alpaca** — ``{"instruction": …, "input": …, "output": …}``

Usage::

    from finetuning.dataset import load_dataset

    dataset = load_dataset(
        path="datasets/train.jsonl",
        tokenizer=tokenizer,
        max_seq_length=4096,
        validation_split=0.05,
        shuffle=True,
        seed=42,
        pack=True,
    )
    # dataset is a DatasetDict with "train" and "validation" splits.
    # Each split has "input_ids", "attention_mask", "labels" columns.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("aios.finetuning.dataset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, str] = {
    "qwen2.5": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n{assistant}<|im_end|>"
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{assistant}<|eot_id|>"
    ),
    "gemma": (
        "<bos><start_of_turn>user\n{user}<end_of_turn>\n"
        "<start_of_turn>model\n{assistant}<end_of_turn>"
    ),
    "mistral": (
        "<s>[INST] {user} [/INST]{assistant}</s>"
    ),
}

SYSTEM_PROMPTS: List[str] = [
    "You are a helpful, knowledgeable AI assistant.",
    "You are AIOS, an intelligent assistant for device control and general tasks.",
    "You are a precise and clear AI assistant that helps users with their questions.",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single user-assistant exchange."""

    user: str
    assistant: str


@dataclass
class ChatExample:
    """A complete chat example ready for formatting and tokenisation.

    Attributes:
        turns:   Ordered list of user-assistant exchanges.
        system:  System prompt for this example.
        source:  Format identifier (``"sharegpt"``, ``"messages"``, ``"alpaca"``).
    """

    turns: List[Turn] = field(default_factory=list)
    system: str = ""
    source: str = ""

    @property
    def num_turns(self) -> int:
        """Number of user-assistant exchanges."""
        return len(self.turns)

    @property
    def text(self) -> str:
        """Human-readable representation (for debugging)."""
        lines = [f"=== {self.source} ==="]
        if self.system:
            lines.append(f"SYSTEM: {self.system}")
        for i, t in enumerate(self.turns):
            lines.append(f"TURN {i}")
            lines.append(f"  USER:      {t.user[:80]}{'...' if len(t.user) > 80 else ''}")
            lines.append(f"  ASSISTANT: {t.assistant[:80]}{'...' if len(t.assistant) > 80 else ''}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_sharegpt(item: dict) -> Optional[ChatExample]:
    """Parse a ShareGPT-format conversation.

    Expected schemas::

        {"conversations": [{"from": "human"|"gpt", "value": …}]}
        {"conversation":  [{"role":  "user" |"assistant", "content": …}]}

    Args:
        item: A single JSON object from the dataset.

    Returns:
        A ``ChatExample``, or ``None`` if parsing failed.
    """
    conversations = item.get("conversations") or item.get("conversation")
    if not conversations or not isinstance(conversations, list):
        return None

    example = ChatExample(source="sharegpt")
    system = item.get("system", "")
    user_msg: Optional[str] = None

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or turn.get("role") or "").strip().lower()
        value = turn.get("value") or turn.get("content") or ""
        if not value:
            continue
        if role in ("human", "user"):
            if user_msg is not None:
                user_msg = None
                break
            user_msg = value
        elif role in ("gpt", "assistant", "model"):
            if user_msg is not None:
                example.turns.append(Turn(user=user_msg, assistant=value))
                user_msg = None

    example.system = system or ""
    if not example.turns:
        return None
    return example


def _parse_alpaca(item: dict) -> Optional[ChatExample]:
    """Parse an Alpaca-format instruction.

    Expected schema::

        {"instruction": …, "input": …, "output": …}

    Args:
        item: A single JSON object from the dataset.

    Returns:
        A ``ChatExample``, or ``None`` if parsing failed.
    """
    instruction = (item.get("instruction") or "").strip()
    output = (item.get("output") or "").strip()
    if not instruction or not output:
        return None

    inp = (item.get("input") or "").strip()
    user_text = f"{instruction}\n\n{inp}" if inp else instruction

    return ChatExample(
        turns=[Turn(user=user_text, assistant=output)],
        system=item.get("system", ""),
        source="alpaca",
    )


def _parse_messages(item: dict) -> Optional[ChatExample]:
    """Parse an OpenAI-style messages list.

    Expected schema::

        {"messages": [
            {"role": "system"|"user"|"assistant", "content": …}
        ]}

    Args:
        item: A single JSON object from the dataset.

    Returns:
        A ``ChatExample``, or ``None`` if parsing failed.
    """
    messages = item.get("messages")
    if not messages or not isinstance(messages, list):
        return None

    example = ChatExample(source="messages")
    system = ""
    user_msg: Optional[str] = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            system = content
        elif role == "user":
            if user_msg is not None:
                user_msg = None
                break
            user_msg = content
        elif role == "assistant":
            if user_msg is not None:
                example.turns.append(Turn(user=user_msg, assistant=content))
                user_msg = None

    example.system = system or ""
    if not example.turns:
        return None
    return example


PARSERS: Dict[str, Callable[[dict], Optional[ChatExample]]] = {
    "sharegpt": _parse_sharegpt,
    "alpaca": _parse_alpaca,
    "messages": _parse_messages,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_example(
    example: ChatExample,
    min_characters: int = 10,
    max_turns: int = 50,
) -> Tuple[bool, str]:
    """Validate a parsed ``ChatExample``.

    Checks:
        * At least one complete turn exists.
        * Turn count does not exceed *max_turns*.
        * Every user and assistant message meets the minimum length.
        * No message consists only of whitespace or common fillers.

    Args:
        example:       The example to validate.
        min_characters: Minimum character count per message.
        max_turns:      Maximum allowed turns.

    Returns:
        A tuple ``(is_valid, reason)``.
    """
    if not example.turns:
        return False, "no turns"

    if len(example.turns) > max_turns:
        return False, f"too many turns: {len(example.turns)} > {max_turns}"

    for i, turn in enumerate(example.turns):
        user_clean = turn.user.strip()
        asst_clean = turn.assistant.strip()

        if len(user_clean) < min_characters:
            return False, f"turn {i}: user message too short ({len(user_clean)} < {min_characters})"
        if len(asst_clean) < min_characters:
            return False, f"turn {i}: assistant message too short ({len(asst_clean)} < {min_characters})"

        _fillers = {"", " ", "ok", "okay", "yes", "no", "i don't know", "i don't know.",
                     "sorry", "sorry.", "i cannot answer that."}
        if user_clean.lower() in _fillers:
            return False, f"turn {i}: user message is a filler"
        if asst_clean.lower() in _fillers:
            return False, f"turn {i}: assistant message is a filler"

    return True, ""


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_example(
    example: ChatExample,
    template_key: str,
    system_prompt: str = "",
) -> Optional[str]:
    """Format a ``ChatExample`` into a single training string.

    Args:
        example:      The chat example to format.
        template_key: Chat template key (e.g. ``"qwen2.5"``, ``"llama3"``).
        system_prompt: Override system prompt. If empty, uses ``example.system``
                       or a random default.

    Returns:
        A formatted training string, or ``None`` if the template is unknown.
    """
    template = TEMPLATES.get(template_key)
    if template is None:
        return None

    system = system_prompt or example.system
    if not system:
        system = random.choice(SYSTEM_PROMPTS)

    parts: List[str] = []
    for i, turn in enumerate(example.turns):
        if i == 0:
            parts.append(template.format(system=system, user=turn.user, assistant=turn.assistant))
        else:
            part = template.format(system="", user=turn.user, assistant=turn.assistant)
            if "<|im_start|>system\n" in part:
                part = part.replace("<|im_start|>system\n", "<|im_start|>ignore\n", 1)
            parts.append(part)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_with_labels(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, Any]:
    """Tokenize a training string and create labels with assistant-only loss.

    Labels are set to ``-100`` for tokens that should *not* contribute to the
    loss (system prompt, user messages).  Only the assistant response tokens
    receive their original IDs as labels.

    The assistant portion is identified by tokenizing the prompt alone and
    the full text, then using the difference.  If that fails, a simple string
    heuristic is used as fallback.

    Args:
        text:       The formatted training string.
        tokenizer:  HuggingFace tokenizer.
        max_length: Maximum sequence length (truncation).

    Returns:
        A dict with ``input_ids``, ``attention_mask``, and ``labels``.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Approach: tokenize the prompt (everything up to the last assistant turn)
    # and use the difference to find assistant tokens.
    _assistant_markers = [
        "<|im_start|>assistant", "<|start_header_id|>assistant",
        "[/INST]", "<start_of_turn>model",
    ]

    prompt_end = 0
    for marker in _assistant_markers:
        idx = text.rfind(marker)
        if idx > prompt_end:
            prompt_end = idx

    if prompt_end > 0:
        prompt_text = text[:prompt_end]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
        labels = input_ids[:]
        for i in range(len(prompt_ids)):
            if i < len(labels):
                labels[i] = -100
    else:
        labels = input_ids[:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------


def pack_sequences(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    seed: int = 42,
) -> HFDataset:
    """Pack multiple short sequences into contiguous blocks of *max_length*.

    Sequences are concatenated with an EOS token separator, then split into
    chunks of exactly *max_length* tokens.  Labels are concatenated in the
    same way.

    Args:
        dataset:    Tokenized dataset with ``input_ids``, ``labels`` columns.
        tokenizer:  HuggingFace tokenizer (used for ``eos_token_id``).
        max_length: Target sequence length after packing.
        seed:       Random seed for shuffling before packing.

    Returns:
        A new dataset with packed sequences.
    """
    eos_token_id = tokenizer.eos_token_id or 0

    all_input_ids: List[int] = []
    all_labels: List[int] = []

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    for idx in indices:
        example = dataset[idx]
        all_input_ids.extend(example["input_ids"])
        all_labels.extend(example["labels"])
        all_input_ids.append(eos_token_id)
        all_labels.append(-100)

    packed_input_ids = []
    packed_labels = []
    for i in range(0, len(all_input_ids), max_length):
        chunk_ids = all_input_ids[i:i + max_length]
        chunk_labels = all_labels[i:i + max_length]
        if len(chunk_ids) < max_length:
            chunk_ids = chunk_ids + [tokenizer.pad_token_id or 0] * (max_length - len(chunk_ids))
            chunk_labels = chunk_labels + [-100] * (max_length - len(chunk_labels))
        packed_input_ids.append(chunk_ids)
        packed_labels.append(chunk_labels)

    packed_attention_mask = [[1 if tid != (tokenizer.pad_token_id or 0) else 0 for tid in ids]
                             for ids in packed_input_ids]

    return HFDataset.from_dict({
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
        "labels": packed_labels,
    })


# ---------------------------------------------------------------------------
# Main dataset loader
# ---------------------------------------------------------------------------


def load_dataset(
    path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 2048,
    template_key: str = "qwen2.5",
    validation_split: float = 0.0,
    shuffle: bool = True,
    seed: int = 42,
    pack: bool = False,
    max_samples: int = 0,
    system_prompt: str = "",
    min_characters: int = 10,
    max_turns: int = 50,
) -> DatasetDict:
    """Load a JSONL chat dataset from disk.

    The pipeline is:

    1. Read and parse every JSON line.
    2. Validate each parsed example (removing invalid ones).
    3. Format examples into training strings using a chat template.
    4. Tokenize with label masking (only assistant tokens contribute to loss).
    5. Optionally pack sequences.
    6. Shuffle and optionally split into train / validation.

    Args:
        path:              Path to a ``.jsonl`` file.
        tokenizer:         HuggingFace tokenizer.
        max_seq_length:    Maximum sequence length for tokenization.
        template_key:      Chat template key (e.g. ``"qwen2.5"``, ``"llama3"``).
        validation_split:  Fraction of data to hold out for validation
                           (``0.0`` = no split).
        shuffle:           Shuffle the dataset after loading.
        seed:              Random seed for shuffling and splitting.
        pack:              Enable sequence packing (increases throughput).
        max_samples:       Limit to this many examples (``0`` = all).
        system_prompt:     Override system prompt for all examples.
        min_characters:    Minimum character count per message for validation.
        max_turns:         Maximum turns per example for validation.

    Returns:
        A ``DatasetDict`` with ``"train"`` and optionally ``"validation"``
        splits.  Each split contains ``input_ids``, ``attention_mask``, and
        ``labels`` columns.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If no valid examples are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset: %s", path)
    logger.info("Template: %s | max_seq_length: %s | pack: %s | validation_split: %s",
                template_key, max_seq_length, pack, validation_split)

    # ------------------------------------------------------------------
    # 1. Parse
    # ------------------------------------------------------------------
    raw_examples: List[ChatExample] = []
    total_lines = 0
    parse_errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            example: Optional[ChatExample] = None
            if "conversations" in item or "conversation" in item:
                example = _parse_sharegpt(item)
            elif "instruction" in item:
                example = _parse_alpaca(item)
            elif "messages" in item:
                example = _parse_messages(item)

            if example is not None:
                raw_examples.append(example)

    if parse_errors:
        logger.warning("Parse errors: %s / %s lines", parse_errors, total_lines)
    logger.info("Parsed %s raw examples from %s lines", len(raw_examples), total_lines)

    if not raw_examples:
        raise ValueError(f"No parseable examples found in {path}")

    # ------------------------------------------------------------------
    # 2. Validate
    # ------------------------------------------------------------------
    valid_examples: List[ChatExample] = []
    validation_errors: Dict[str, int] = {}

    for ex in raw_examples:
        is_valid, reason = validate_example(ex, min_characters, max_turns)
        if is_valid:
            valid_examples.append(ex)
        else:
            validation_errors[reason] = validation_errors.get(reason, 0) + 1

    if validation_errors:
        logger.warning("Validation removed %s / %s examples:", len(raw_examples) - len(valid_examples), len(raw_examples))
        for reason, count in sorted(validation_errors.items(), key=lambda x: -x[1]):
            logger.warning("  %s: %s", reason, count)

    if not valid_examples:
        raise ValueError(f"No valid examples after validation in {path}")

    logger.info("Valid examples: %s", len(valid_examples))

    # ------------------------------------------------------------------
    # 3. Limit
    # ------------------------------------------------------------------
    if max_samples > 0 and max_samples < len(valid_examples):
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(valid_examples)
        valid_examples = valid_examples[:max_samples]
        logger.info("Limited to %s examples", max_samples)

    # ------------------------------------------------------------------
    # 4. Format
    # ------------------------------------------------------------------
    texts: List[str] = []
    format_errors = 0
    for ex in valid_examples:
        text = format_example(ex, template_key, system_prompt)
        if text is None:
            format_errors += 1
            continue
        texts.append(text)

    if format_errors:
        logger.warning("Format errors: %s", format_errors)
    logger.info("Formatted %s training strings", len(texts))

    if not texts:
        raise ValueError(f"No examples could be formatted in {path}")

    # ------------------------------------------------------------------
    # 5. Tokenize
    # ------------------------------------------------------------------
    logger.info("Tokenizing %s examples (max_seq_length=%s) ...", len(texts), max_seq_length)

    all_encodings: List[Dict[str, Any]] = []
    tokenize_errors = 0
    for text in texts:
        try:
            enc = tokenize_with_labels(text, tokenizer, max_seq_length)
            all_encodings.append(enc)
        except Exception:
            tokenize_errors += 1
            continue

    if tokenize_errors:
        logger.warning("Tokenization errors: %s / %s", tokenize_errors, len(texts))
    logger.info("Tokenized %s examples", len(all_encodings))

    if not all_encodings:
        raise ValueError(f"No examples could be tokenized from {path}")

    dataset = HFDataset.from_dict({
        "input_ids": [e["input_ids"] for e in all_encodings],
        "attention_mask": [e["attention_mask"] for e in all_encodings],
        "labels": [e["labels"] for e in all_encodings],
    })

    # ------------------------------------------------------------------
    # 6. Pack
    # ------------------------------------------------------------------
    if pack:
        logger.info("Packing sequences (max_length=%s) ...", max_seq_length)
        original_count = len(dataset)
        dataset = pack_sequences(dataset, tokenizer, max_seq_length, seed)
        logger.info("Packed %s examples into %s sequences", original_count, len(dataset))

    # ------------------------------------------------------------------
    # 7. Shuffle & split
    # ------------------------------------------------------------------
    if shuffle and not pack:
        dataset = dataset.shuffle(seed=seed)
        logger.info("Shuffled dataset (seed=%s)", seed)

    if validation_split > 0.0 and not pack:
        split_dataset = dataset.train_test_split(
            test_size=validation_split,
            seed=seed,
        )
        logger.info("Split: %s train, %s validation",
                     len(split_dataset["train"]), len(split_dataset["test"]))
        split_dataset["validation"] = split_dataset.pop("test")
        dataset = split_dataset
    elif pack:
        # When packed, split after shuffling the encoded examples
        n = len(dataset)
        n_val = max(1, int(n * validation_split)) if validation_split > 0 else 0
        if n_val > 0:
            indices = list(range(n))
            random.Random(seed).shuffle(indices)

            def _select(idx_list: List[int]) -> HFDataset:
                return HFDataset.from_dict({
                    "input_ids": [dataset[i]["input_ids"] for i in idx_list],
                    "attention_mask": [dataset[i]["attention_mask"] for i in idx_list],
                    "labels": [dataset[i]["labels"] for i in idx_list],
                })

            dataset = DatasetDict({
                "train": _select(indices[n_val:]),
                "validation": _select(indices[:n_val]),
            })
            logger.info("Split (packed): %s train, %s validation",
                         len(dataset["train"]), len(dataset["validation"]))
        else:
            dataset = DatasetDict({"train": dataset, "validation": HFDataset.from_list([])})
    else:
        dataset = DatasetDict({"train": dataset, "validation": HFDataset.from_list([])})

    return dataset


def load_datasets(
    config: "TrainConfig",  # noqa: F821
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """Load train and validation datasets from a ``TrainConfig``.

    This is a convenience wrapper around :func:`load_dataset` that reads
    ``train_file`` / ``valid_file`` from the config.  If a separate
    validation file exists it is loaded directly; otherwise the training
    set is split using ``validation_split``.

    Args:
        config:   A :class:`TrainConfig` instance.
        tokenizer: HuggingFace tokenizer.

    Returns:
        A ``DatasetDict`` with ``"train"`` and ``"validation"`` splits.
    """
    from finetuning.config import TrainConfig  # noqa: F811

    template = config.model_info.chat_template
    system_prompt = (
        "You are AIOS, an intelligent AI assistant. "
        "You are helpful, precise, and follow instructions accurately."
    )
    val_path = config.resolved_valid_file
    has_separate_val = val_path.exists() and val_path.stat().st_size > 0

    logger.info("Loading training dataset ...")
    train_val_split = 0.0 if has_separate_val else 0.05

    result = load_dataset(
        path=config.resolved_train_file,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        template_key=template,
        validation_split=train_val_split,
        shuffle=True,
        seed=config.seed,
        pack=config.packing,
        max_samples=config.dataset_size,
        system_prompt=system_prompt,
    )

    if has_separate_val:
        logger.info("Loading separate validation dataset from %s ...", val_path)
        val = load_dataset(
            path=val_path,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            template_key=template,
            validation_split=0.0,
            shuffle=False,
            pack=False,
            system_prompt=system_prompt,
        )
        result["validation"] = val["train"]

    return result
