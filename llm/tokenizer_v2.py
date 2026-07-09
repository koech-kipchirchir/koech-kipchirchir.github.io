"""
AIOS BPE Tokenizer v2
Professional-grade Byte-Pair Encoding tokenizer with special tokens
Supports training on large corpora and proper encoding/decoding
"""
import json
import os
from collections import defaultdict
from typing import List, Optional, Dict, Tuple


class BPETokenizer:
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<system>": 4,
            "<user>": 5,
            "<assistant>": 6,
            "<tool_call>": 7,
            "<tool_result>": 8,
            "<image>": 9,
        }
        self.num_special = len(self.special_tokens)

        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        self.byte_encoder: Dict[int, str] = self._build_byte_encoder()
        self.byte_decoder: Dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

        self._initialized = False

    def _build_byte_encoder(self) -> Dict[int, str]:
        chars = []
        for i in range(33, 127):
            chars.append(chr(i))
        for i in range(161, 256):
            chars.append(chr(i))
        for i in range(256, 500):
            chars.append(chr(i))

        # Unicode blocks for comprehensive coverage
        blocks = [
            (0x00C0, 0x024F),  # Latin Extended
            (0x0370, 0x03FF),  # Greek and Coptic
            (0x0400, 0x04FF),  # Cyrillic
            (0x4E00, 0x9FFF),  # CJK Unified
            (0xAC00, 0xD7AF),  # Hangul
            (0x0600, 0x06FF),  # Arabic
            (0x0900, 0x097F),  # Devanagari
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
        ]

        for start, end in blocks:
            for i in range(start, end + 1):
                if chr(i) not in chars:
                    chars.append(chr(i))

        return {i: c for i, c in enumerate(chars)}

    def _get_stats(self, words: List[List[int]]) -> Dict[Tuple[int, int], int]:
        stats = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                stats[(word[i], word[i + 1])] += 1
        return stats

    def _merge_pair(self, words: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
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
        word_freqs = defaultdict(int)
        for text in texts:
            text_bytes = text.encode("utf-8")
            word = list(text_bytes)
            word_freqs[tuple(word)] += 1

        base_vocab_size = len(self.byte_encoder)
        current_id = self.num_special + base_vocab_size

        words = [list(w) for w in word_freqs.keys()]
        freqs = list(word_freqs.values())

        num_merges = self.vocab_size - current_id
        if num_merges <= 0:
            return

        for i in range(num_merges):
            stats = self._get_stats(words)
            if not stats:
                break

            most_frequent = max(stats, key=lambda p: stats[p] * freqs[words.index(list(p))] if list(p) in [list(w) for w in words] else 0)

            pair = most_frequent
            new_id = current_id
            self.merges[pair] = new_id
            current_id += 1

            words = self._merge_pair(words, pair, new_id)

            # Build vocab
            merged_token = self._decode_pair(pair)
            self.vocab[merged_token] = new_id
            self.inverse_vocab[new_id] = merged_token

            if verbose and (i + 1) % 1000 == 0:
                print(f"BPE merge {i + 1}/{num_merges}")
                print(f"  Pair: {pair} -> '{merged_token}' (freq: {stats[pair]})")

        self._initialized = True
        print(f"BPE training complete. Vocab size: {len(self.vocab)}")

    def _decode_pair(self, pair: Tuple[int, int]) -> str:
        try:
            left = self.inverse_vocab.get(pair[0])
            right = self.inverse_vocab.get(pair[1])
            if left and right:
                return left + right
            return f"<byte_{pair[0]}_{pair[1]}>"
        except:
            return f"<byte_{pair[0]}_{pair[1]}>"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not self._initialized:
            return self._basic_encode(text, add_special_tokens)

        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens["<bos>"])

        # Parse special tokens first
        remaining = text
        while remaining:
            found = False
            for st in sorted(self.special_tokens.keys(), key=len, reverse=True):
                if remaining.startswith(st):
                    ids.append(self.special_tokens[st])
                    remaining = remaining[len(st):]
                    found = True
                    break
            if not found:
                # Take one regular character
                remaining = remaining[1:]  # We'll handle it below

        # BPE encode the text (excluding special tokens)
        text_bytes = text.encode("utf-8")
        word = list(text_bytes)

        # Apply merges
        while len(word) > 1:
            stats = self._get_stats([word])
            min_pair = None
            min_rank = float('inf')

            for pair in stats:
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair

            if min_pair is None:
                break

            word = self._merge_pair([word], min_pair, min_rank)[0]

        # Convert to IDs
        for token in word:
            token_str = self.inverse_vocab.get(token)
            if token_str and token_str in self.vocab:
                ids.append(self.vocab[token_str])
            else:
                ids.append(self.special_tokens["<unk>"])

        if add_special_tokens:
            ids.append(self.special_tokens["<eos>"])

        return ids

    def _basic_encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens["<bos>"])

        i = 0
        n = len(text)
        while i < n:
            found_special = False
            for st in sorted(self.special_tokens.keys(), key=len, reverse=True):
                if text[i:i+len(st)] == st:
                    ids.append(self.special_tokens[st])
                    i += len(st)
                    found_special = True
                    break
            if not found_special:
                char = text[i]
                try:
                    byte_val = char.encode("utf-8")[0]
                    ids.append(self.num_special + byte_val)
                except:
                    ids.append(self.special_tokens["<unk>"])
                i += 1

        if add_special_tokens:
            ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        text_parts = []
        for idx in ids:
            if idx in self.special_tokens.values():
                for token, tid in self.special_tokens.items():
                    if tid == idx:
                        text_parts.append(token)
                        break
            elif idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                if len(token) == 1:
                    try:
                        text_parts.append(token.encode("latin-1").decode("utf-8"))
                    except:
                        text_parts.append(token)
                else:
                    text_parts.append(token)
            else:
                text_parts.append("<unk>")
        return "".join(text_parts)

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
            "byte_encoder": {str(k): v for k, v in self.byte_encoder.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data.get("vocab_size", 32768))
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = {tuple(int(x) for x in k.split(",")): v for k, v in data["merges"].items()}
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.byte_encoder = {int(k): v for k, v in data["byte_encoder"].items()}
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._initialized = bool(tokenizer.merges)

        return tokenizer

    @property
    def vocab_size_property(self) -> int:
        return len(self.vocab)


class BPETokenizerWrapper:
    """Wrapper that mimics the original CharacterTokenizer interface"""
    def __init__(self, bpe_path: Optional[str] = None, vocab_size: int = 32768):
        if bpe_path and os.path.exists(bpe_path):
            self.bpe = BPETokenizer.load(bpe_path)
        else:
            self.bpe = BPETokenizer(vocab_size=vocab_size)
        self.special_tokens = list(self.bpe.special_tokens.keys())
        self.token_to_id = self.bpe.vocab
        self.id_to_token = self.bpe.inverse_vocab

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


if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=5000)

    sample_texts = [
        "Hello, how are you today? I am AIOS, your intelligent assistant.",
        "The quick brown fox jumps over the lazy dog near the bank of the river.",
        "Artificial Intelligence is transforming the way we interact with technology.",
        "Set an alarm for 7:30 AM tomorrow morning please.",
        "Turn on the flashlight and vibrate the phone for 2 seconds.",
    ]

    print("Training BPE tokenizer...")
    tokenizer.train(sample_texts, verbose=True)

    test_phrase = "Hello! Turn on the flashlight and set alarm for 7:30 AM"
    encoded = tokenizer.encode(test_phrase)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest:")
    print(f"  Original: {test_phrase}")
    print(f"  Encoded:  {encoded[:30]}... (len={len(encoded)})")
    print(f"  Decoded:  {decoded}")
    print(f"  Vocab:    {len(tokenizer.vocab)} tokens")
