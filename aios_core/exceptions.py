from __future__ import annotations


class AIOSEngineError(Exception):
    """Base exception for all AIOS Engine errors."""


class ConfigurationError(AIOSEngineError):
    """Raised when engine configuration is invalid."""


class ProviderError(AIOSEngineError):
    """Raised when the LLM provider returns an error."""


class RateLimitError(ProviderError):
    """Raised when rate limited by the provider."""


class TimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ContextLengthExceededError(AIOSEngineError):
    """Raised when the context exceeds the model's maximum length."""
