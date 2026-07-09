"""
Model registry: name, provider, pricing, and metadata lookup.
"""

from __future__ import annotations

from benchmarks.config import MODEL_PRICING


# Provider groupings
PROVIDER_MODELS: dict[str, list[str]] = {
    "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet"],
    "meta":      ["llama-3-8b", "llama-3-70b"],
    "mistral":   ["mixtral-8x7b"],
    "google":    ["gemini-1.5-pro", "gemini-1.5-flash"],
    "deepseek":  ["deepseek-v2"],
    "cohere":    ["command-r-plus"],
}


def list_models() -> list[str]:
    """Return all known model names."""
    return list(MODEL_PRICING.keys())


def list_providers() -> list[str]:
    """Return all provider names."""
    return list(PROVIDER_MODELS.keys())


def get_models_by_provider(provider: str) -> list[str]:
    """Return model names for a given provider."""
    return PROVIDER_MODELS.get(provider, [])


def get_pricing(model_name: str) -> dict[str, float]:
    """Return {input, output} pricing dict for a model."""
    return MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})


def get_provider(model_name: str) -> str:
    """Return the provider for a model name."""
    for provider, models in PROVIDER_MODELS.items():
        if model_name in models:
            return provider
    return "unknown"


def get_model_info(model_name: str) -> dict:
    """Return full model info dict."""
    pricing = get_pricing(model_name)
    return {
        "name": model_name,
        "provider": get_provider(model_name),
        "cost_per_1k_input": pricing["input"],
        "cost_per_1k_output": pricing["output"],
    }


def register_model(name: str, provider: str, input_cost: float, output_cost: float) -> None:
    """Register a custom model."""
    MODEL_PRICING[name] = {"input": input_cost, "output": output_cost}
    if provider not in PROVIDER_MODELS:
        PROVIDER_MODELS[provider] = []
    if name not in PROVIDER_MODELS[provider]:
        PROVIDER_MODELS[provider].append(name)
