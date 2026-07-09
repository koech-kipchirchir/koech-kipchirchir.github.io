"""
Simple byte-level tokenizer for AIOS v2 training.
Vocabulary: 4 special tokens + 256 byte tokens = 260 total.
Fast, no BPE training needed.
"""
import json
import os
from typing import List, Optional, Dict


class ByteTokenizer:
    def __init__(self, vocab_size: int = 260):
        # Fixed: 4 special + 256 bytes = 260
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.num_special = 4
        self.base_size = 256
        self._vocab_size = self.num_special + self.base_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        return [f"<byte_{i}>" for i in range(self.base_size)]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for byte in text.encode("utf-8"):
            ids.append(self.num_special + byte)
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_list = []
        for tid in ids:
            if tid < self.num_special:
                continue  # skip special tokens
            byte_val = tid - self.num_special
            if 0 <= byte_val < 256:
                bytes_list.append(byte_val)
        return bytes(bytes_list).decode("utf-8", errors="replace")


class ByteTokenizerWrapper:
    """Wrapper compatible with v2 trainer interface."""
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, bpe_path=None, vocab_size=260):
        self.tokenizer = ByteTokenizer(vocab_size=vocab_size)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        data = {"type": "byte", "vocab_size": self.vocab_size}
        with open(path, "w") as f:
            json.dump(data, f)
