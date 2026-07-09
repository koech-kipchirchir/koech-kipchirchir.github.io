from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from aios_core.exceptions import AIOSEngineError
from aios_core.models import UsageStats

logger = logging.getLogger("aios.core.token_manager")

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-4-32k": (60.00, 120.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-2.1": (8.00, 24.00),
    "claude-2.0": (8.00, 24.00),
    "claude-instant-1.2": (0.80, 2.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.04, 0.15),
    "gemini-2.0-flash": (0.10, 0.40),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-coder": (0.14, 0.28),
    "qwen-turbo": (0.30, 0.60),
    "qwen-plus": (0.80, 2.00),
    "qwen-max": (2.00, 6.00),
    "qwen3-8b": (0.00, 0.00),
    "qwen3-32b": (0.00, 0.00),
    "qwen3-72b": (0.00, 0.00),
    "llama-3-8b": (0.00, 0.00),
    "llama-3-70b": (0.00, 0.00),
    "llama-3.1-8b": (0.00, 0.00),
    "llama-3.1-70b": (0.00, 0.00),
    "llama-3.1-405b": (0.00, 0.00),
    "mistral-large": (2.00, 6.00),
    "mistral-medium": (2.00, 6.00),
    "mistral-small": (1.00, 3.00),
    "codestral": (1.00, 3.00),
    "mixtral-8x7b": (0.00, 0.00),
    "command-r": (0.15, 0.60),
    "command-r-plus": (2.50, 10.00),
}

_MODEL_ALIASES: dict[str, str] = {
    "gpt-4o-1": "gpt-4o",
    "gpt-4o-2": "gpt-4o",
    "gpt-4o-3": "gpt-4o",
    "gpt-4o-4": "gpt-4o",
    "gpt-4o-5": "gpt-4o",
    "claude-sonnet-4": "claude-3-5-sonnet-20241022",
    "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
    "claude-opus-4": "claude-3-opus-20240229",
    "claude-haiku-3.5": "claude-3-haiku-20240307",
}


@dataclass
class TokenUsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 6),
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
            "timestamp": self.timestamp,
        }


class TokenizationError(AIOSEngineError):
    pass


def resolve_model_name(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def lookup_pricing(model: str) -> tuple[float, float]:
    key = resolve_model_name(model)
    for pattern, prices in MODEL_PRICING.items():
        if key == pattern or key.startswith(pattern + "-") or key.startswith(pattern + ":"):
            return prices
    return (0.0, 0.0)


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    input_price, output_price = lookup_pricing(model)
    cost = (prompt_tokens / 1_000_000) * input_price
    cost += (completion_tokens / 1_000_000) * output_price
    return cost


class TokenizationStrategy(ABC):
    @abstractmethod
    def count(self, text: str) -> int: ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
            if msg.get("name"):
                total += self.count(str(msg["name"]))
        total += len(messages) * 4
        total += 3
        return total

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...

    def estimate(self, text: str) -> int:
        return self.count(text)

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Strategy", "").lower()


class ApproximateStrategy(TokenizationStrategy):
    def count(self, text: str) -> int:
        if not text:
            return 0
        return int(len(text) * 1.3) + 1

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    @property
    def name(self) -> str:
        return "approximate"


class TikTokenStrategy(TokenizationStrategy):
    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._encoding: Any = None
        self._logger = logging.getLogger("aios.token_manager.tiktoken")
        self._init_encoding()

    def _init_encoding(self) -> None:
        try:
            import tiktoken
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
                self._logger.info("Loaded tiktoken encoding for %s", self._model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
                self._logger.warning(
                    "Model %s not found in tiktoken; using cl100k_base", self._model
                )
        except ImportError:
            self._logger.warning("tiktoken not available; will raise on use")

    def _ensure(self) -> None:
        if self._encoding is None:
            raise TokenizationError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        self._ensure()
        return len(self._encoding.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        self._ensure()
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        self._ensure()
        return self._encoding.decode(tokens)

    def estimate(self, text: str) -> int:
        try:
            return self.count(text)
        except TokenizationError:
            return int(len(text) * 1.3) + 1


class HuggingFaceStrategy(TokenizationStrategy):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        self._model_name = model_name
        self._tokenizer: Any = None
        self._logger = logging.getLogger("aios.token_manager.hf")
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._logger.info("Loaded HuggingFace tokenizer: %s", self._model_name)
        except ImportError:
            self._logger.warning(
                "transformers not available; HuggingFace strategy will fall back"
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to load HF tokenizer %s: %s", self._model_name, exc
            )

    def _ensure(self) -> None:
        if self._tokenizer is None:
            raise TokenizationError(
                "HuggingFace tokenizer not available. "
                "Install with: pip install transformers"
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        self._ensure()
        return len(self._tokenizer.encode(text))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        self._ensure()
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        self._ensure()
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0
        self._ensure()
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return self.count(formatted)
        except Exception:
            return super().count_messages(messages)


class LlamaCPPStrategy(TokenizationStrategy):
    def __init__(self, model_path: str = "") -> None:
        self._model_path = model_path
        self._model: Any = None
        self._logger = logging.getLogger("aios.token_manager.llamacpp")
        if model_path:
            self._init_model()

    def _init_model(self) -> None:
        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=self._model_path, n_ctx=4096, verbose=False
            )
            self._logger.info("Loaded llama.cpp model: %s", self._model_path)
        except ImportError:
            self._logger.warning(
                "llama-cpp-python not available; LlamaCPP strategy will fall back"
            )
        except Exception as exc:
            self._logger.warning("Failed to load llama.cpp model: %s", exc)

    def _ensure(self) -> None:
        if self._model is None:
            raise TokenizationError(
                "llama.cpp model not loaded. "
                "Install with: pip install llama-cpp-python"
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        self._ensure()
        return len(self._model.tokenize(text.encode("utf-8")))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        self._ensure()
        return self._model.tokenize(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        self._ensure()
        return self._model.detokenize(tokens).decode("utf-8", errors="replace")


class OllamaStrategy(TokenizationStrategy):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: Any = None
        self._logger = logging.getLogger("aios.token_manager.ollama")
        self._init_client()

    def _init_client(self) -> None:
        try:
            import httpx
            self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
            self._logger.info("Ollama client initialized (%s)", self._base_url)
        except ImportError:
            self._logger.warning(
                "httpx not available; Ollama strategy will fall back"
            )

    def _ensure(self) -> None:
        if self._client is None:
            raise TokenizationError(
                "Ollama HTTP client not available. Install with: pip install httpx"
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        self._ensure()
        if not self._model:
            raise TokenizationError("Ollama model not specified for token counting")
        try:
            response = self._client.post(
                "/api/tokenize",
                json={"model": self._model, "content": text},
            )
            response.raise_for_status()
            data = response.json()
            return len(data.get("tokens", []))
        except Exception as exc:
            self._logger.warning("Ollama tokenize failed: %s", exc)
            raise TokenizationError(f"Ollama tokenization error: {exc}") from exc

    def encode(self, text: str) -> list[int]:
        return self.count(text)  # Ollama returns token count, not IDs

    def decode(self, tokens: list[int]) -> str:
        raise TokenizationError("Ollama does not support token decode via API")


class TokenUsageTracker:
    def __init__(self, model: str = "") -> None:
        self._model = model
        self._records: list[TokenUsageRecord] = []
        self._lock = threading.Lock()
        self._logger = logging.getLogger("aios.token_manager.usage_tracker")

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
        latency_ms: float = 0.0,
    ) -> TokenUsageRecord:
        model_name = model or self._model
        cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
        record = TokenUsageRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            model=model_name,
        )
        with self._lock:
            self._records.append(record)
        return record

    def record_usage_stats(
        self,
        usage: UsageStats,
        model: str | None = None,
        latency_ms: float = 0.0,
    ) -> TokenUsageRecord:
        return self.record(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=model,
            latency_ms=latency_ms,
        )

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return sum(r.completion_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.total_tokens for r in self._records)

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(r.cost for r in self._records)

    @property
    def total_latency_ms(self) -> float:
        with self._lock:
            return sum(r.latency_ms for r in self._records)

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def avg_latency_ms(self) -> float:
        count = self.request_count
        return self.total_latency_ms / count if count else 0.0

    @property
    def avg_cost_per_request(self) -> float:
        count = self.request_count
        return self.total_cost / count if count else 0.0

    @property
    def avg_tokens_per_request(self) -> float:
        count = self.request_count
        return self.total_tokens / count if count else 0.0

    def get_history(self, limit: int = 100) -> list[TokenUsageRecord]:
        with self._lock:
            return self._records[-limit:]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "request_count": self.request_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_cost_per_request": round(self.avg_cost_per_request, 8),
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 1),
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
        self._logger.info("Usage tracker reset")

    def to_usage_stats(self) -> UsageStats:
        return UsageStats(
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
            total_tokens=self.total_tokens,
        )


class TokenManager:
    def __init__(
        self,
        model: str = "gpt-4o",
        hf_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        llama_cpp_path: str = "",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "",
        preferred_strategy: str = "auto",
    ) -> None:
        self._model = model
        self._logger = logging.getLogger("aios.core.token_manager")
        self._usage_tracker = TokenUsageTracker(model=model)
        self._strategies: dict[str, TokenizationStrategy] = {}
        self._active_strategy: TokenizationStrategy = ApproximateStrategy()
        self._init_strategies(
            hf_model_name=hf_model_name,
            llama_cpp_path=llama_cpp_path,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )
        self._select_strategy(preferred_strategy)
        self._logger.info(
            "TokenManager initialized (model=%s, strategy=%s)",
            model,
            self._active_strategy.name,
        )

    def _init_strategies(
        self,
        hf_model_name: str,
        llama_cpp_path: str,
        ollama_base_url: str,
        ollama_model: str,
    ) -> None:
        self._strategies["tiktoken"] = TikTokenStrategy(model=self._model)
        if hf_model_name:
            self._strategies["huggingface"] = HuggingFaceStrategy(
                model_name=hf_model_name
            )
        if llama_cpp_path:
            self._strategies["llamacpp"] = LlamaCPPStrategy(
                model_path=llama_cpp_path
            )
        if ollama_model:
            self._strategies["ollama"] = OllamaStrategy(
                base_url=ollama_base_url, model=ollama_model
            )
        self._strategies["approximate"] = ApproximateStrategy()

    def _select_strategy(self, preferred: str) -> None:
        if preferred == "auto":
            for name in ["tiktoken", "huggingface", "llamacpp", "ollama", "approximate"]:
                strategy = self._strategies.get(name)
                if strategy is None:
                    continue
                try:
                    strategy.count("test")
                    self._active_strategy = strategy
                    self._logger.info("Auto-selected strategy: %s", strategy.name)
                    return
                except TokenizationError:
                    continue
            self._active_strategy = ApproximateStrategy()
            self._logger.info("Fallback to approximate strategy")
        elif preferred in self._strategies:
            self._active_strategy = self._strategies[preferred]
            self._logger.info("Using preferred strategy: %s", preferred)
        else:
            self._logger.warning(
                "Unknown strategy '%s'; falling back to approximate", preferred
            )
            self._active_strategy = ApproximateStrategy()

    @property
    def active_strategy(self) -> str:
        return self._active_strategy.name

    @property
    def available_strategies(self) -> list[str]:
        return list(self._strategies.keys())

    def set_strategy(self, name: str) -> None:
        if name not in self._strategies:
            raise TokenizationError(
                f"Unknown strategy '{name}'. Available: {list(self._strategies.keys())}"
            )
        self._active_strategy = self._strategies[name]
        self._logger.info("Switched to strategy: %s", name)

    def count(self, text: str) -> int:
        return self._active_strategy.count(text)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return self._active_strategy.count_messages(messages)

    def count_chat_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        return self.count_messages(messages)

    def estimate(self, text: str) -> int:
        return self._active_strategy.estimate(text)

    def encode(self, text: str) -> list[int]:
        return self._active_strategy.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._active_strategy.decode(tokens)

    def track_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
    ) -> TokenUsageRecord:
        return self._usage_tracker.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self._model,
            latency_ms=latency_ms,
        )

    def track_usage_stats(
        self,
        usage: UsageStats,
        latency_ms: float = 0.0,
    ) -> TokenUsageRecord:
        return self._usage_tracker.record_usage_stats(
            usage=usage, model=self._model, latency_ms=latency_ms
        )

    def track_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> TokenUsageRecord:
        record = self.track_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        self._logger.debug(
            "Request tracked: %d prompt + %d completion = %d tokens (cost=$%.6f, %.1fms)",
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            record.cost,
            latency_ms,
        )
        return record

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
    ) -> float:
        return calculate_cost(
            model or self._model, prompt_tokens, completion_tokens
        )

    @property
    def total_prompt_tokens(self) -> int:
        return self._usage_tracker.total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._usage_tracker.total_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._usage_tracker.total_tokens

    @property
    def total_cost(self) -> float:
        return self._usage_tracker.total_cost

    @property
    def total_latency_ms(self) -> float:
        return self._usage_tracker.total_latency_ms

    @property
    def request_count(self) -> int:
        return self._usage_tracker.request_count

    @property
    def avg_latency_ms(self) -> float:
        return self._usage_tracker.avg_latency_ms

    def get_usage_stats(self) -> dict[str, Any]:
        return self._usage_tracker.get_stats()

    def get_usage_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._usage_tracker.get_history(limit)]

    def to_usage_stats(self) -> UsageStats:
        return self._usage_tracker.to_usage_stats()

    def reset_usage(self) -> None:
        self._usage_tracker.reset()

    def resolve_model(self, model: str) -> str:
        return resolve_model_name(model)

    def lookup_pricing(self, model: str) -> tuple[float, float]:
        return lookup_pricing(model)

    def estimate_request_tokens(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, int]:
        prompt = self.count_messages(messages)
        return {
            "prompt_tokens": prompt,
            "max_completion_tokens": max_tokens,
            "estimated_total": prompt + max_tokens,
        }

    def estimate_cost(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, float]:
        estimate = self.estimate_request_tokens(messages, max_tokens)
        prompt_cost = calculate_cost(
            self._model, estimate["prompt_tokens"], 0
        )
        completion_cost = calculate_cost(
            self._model, 0, estimate["max_completion_tokens"]
        )
        return {
            "estimated_prompt_cost": round(prompt_cost, 8),
            "estimated_completion_cost": round(completion_cost, 8),
            "estimated_total_cost": round(prompt_cost + completion_cost, 8),
        }
