from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("aios.core.tokenizer")


class TokenCounter:
    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._encoding: Optional[object] = None
        self._logger = logging.getLogger("aios.core.tokenizer")
        self._init_encoding()

    def _init_encoding(self) -> None:
        try:
            import tiktoken

            self._encoding = tiktoken.encoding_for_model(self._model)
            self._logger.info("Loaded tiktoken encoding for %s", self._model)
        except ImportError:
            self._logger.warning("tiktoken not installed; using approximate counting")
        except KeyError:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding("cl100k_base")
                self._logger.warning("Model %s not found; using cl100k_base", self._model)
            except ImportError:
                pass

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            import tiktoken

            return len(self._encoding.encode(text, disallowed_special=()))
        return self._approximate_count(text)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
        total += len(messages) * 4
        total += 3
        return total

    @staticmethod
    def _approximate_count(text: str) -> int:
        return int(len(text) * 1.3) + 1
