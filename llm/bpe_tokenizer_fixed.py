"""
Simplified BPE Tokenizer for AIOS v2 training.
Treats all text (including special tokens like <user>, <assistant>) as regular byte sequences.
"""
import json
import os
from collections import defaultdict
from typing import List, Optional, Dict, Tuple


class BPETokenizerFixed:
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        # Special tokens - only used for encode/decode boundaries, not for special handling
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.num_special = 4  # pad, bos, eos, unk
        self.base_size = 256  # byte-level tokens (0-255)

        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[str, int] = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        for i in range(256):
            self.vocab[f"<byte_{i}>"] = self.num_special + i
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._initialized = False

    def _get_stats(self, words):
        stats = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                stats[(word[i], word[i + 1])] += 1
        return stats

    def _merge_pair(self, words, pair, new_id):
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, texts: List[str], verbose: bool = True):
        """Train BPE tokenizer on text corpus."""
        word_freqs = defaultdict(int)
        total_chars = 0
        for text in texts:
            byte_list = list(text.encode("utf-8"))
            word_freqs[tuple(byte_list)] += 1
            total_chars += len(byte_list)

        words = [list(w) for w in word_freqs.keys()]

        current_id = self.num_special + self.base_size
        num_merges = min(self.vocab_size - current_id, 16000)  # Cap merges for speed

        if num_merges <= 0:
            return

        if verbose:
            print(f"BPE training: {len(word_freqs)} unique sequences, {total_chars} chars, {num_merges} target merges")

        for i in range(num_merges):
            stats = self._get_stats(words)
            if not stats:
                break

            best_pair = max(stats, key=lambda p: stats[p] * sum(1 for w in words if list(p) in [w[j:j+2] for j in range(len(w)-1)]))

            new_id = current_id
            self.merges[best_pair] = new_id
            current_id += 1

            left = self.inverse_vocab.get(best_pair[0], f"<b{best_pair[0]}>")
            right = self.inverse_vocab.get(best_pair[1], f"<b{best_pair[1]}>")
            merged_str = left + right
            self.vocab[merged_str] = new_id
            self.inverse_vocab[new_id] = merged_str

            words = self._merge_pair(words, best_pair, new_id)

            if verbose and (i + 1) % 2000 == 0:
                print(f"  BPE merge {i + 1}/{num_merges}: id={new_id} freq={stats[best_pair]}")

        self._initialized = True
        curr_vocab = len(self.vocab) - self.num_special - self.base_size
        print(f"BPE done: {len(self.vocab)} tokens ({curr_vocab} merges)")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        byte_list = list(text.encode("utf-8"))

        if not self._initialized:
            for byte in byte_list:
                ids.append(self.num_special + byte)
            if add_special_tokens:
                ids.append(self.eos_id)
            return ids

        # Start with byte-level tokens
        tokens = [self.num_special + b for b in byte_list]

        # Apply BPE merges from highest priority (lowest ID) to lowest
        if self.merges:
            changed = True
            while changed:
                changed = False
                stats = self._get_stats([tokens])
                # Find the merge with the smallest ID (earliest merge)
                best_pair = None
                best_id = float('inf')
                for pair in stats:
                    if pair in self.merges:
                        mid = self.merges[pair]
                        if mid < best_id:
                            best_id = mid
                            best_pair = pair
                if best_pair is not None:
                    tokens = self._merge_pair([tokens], best_pair, best_id)[0]
                    changed = True

        ids.extend(tokens)

        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        parts = []
        for tid in ids:
            if tid == self.bos_id or tid == self.eos_id or tid == self.pad_id:
                continue
            if tid in self.inverse_vocab:
                token_str = self.inverse_vocab[tid]
                if token_str.startswith("<byte_"):
                    byte_val = int(token_str[6:-1])
                    parts.append(bytes([byte_val]).decode("utf-8", errors="replace"))
                elif token_str.startswith("<"):
                    parts.append(token_str)
                else:
                    parts.append(token_str)
            else:
                parts.append("<unk>")
        return "".join(parts)


class BPETokenizerWrapper:
    """Wrapper compatible with the v2 trainer."""
    def __init__(self, bpe_path: Optional[str] = None, vocab_size: int = 32768):
        if bpe_path and os.path.exists(bpe_path):
            self.bpe = self._load(bpe_path)
        else:
            self.bpe = BPETokenizerFixed(vocab_size=vocab_size)
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    @property
    def vocab(self):
        return list(self.bpe.vocab.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.bpe.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.bpe.encode(text, add_special_tokens)

    def decode(self, ids: List[int]) -> str:
        return self.bpe.decode(ids)

    def save(self, path: str):
        data = {
            "vocab_size": self.bpe.vocab_size,
            "vocab": self.bpe.vocab,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.bpe.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = BPETokenizerFixed(vocab_size=data.get("vocab_size", 32768))
        tok.vocab = data["vocab"]
        tok.merges = {tuple(int(x) for x in k.split(",")): v for k, v in data["merges"].items()}
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok._initialized = bool(tok.merges)
        return tok
