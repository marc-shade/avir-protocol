"""
AVIR Provider Implementations

AI provider integrations for AVIR verification.
"""

from .base import AVIRProvider
from .claude import ClaudeProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider

__all__ = [
    "AVIRProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "GeminiProvider",
]


def get_provider(name: str, **kwargs) -> AVIRProvider:
    """
    Get provider instance by name.

    Args:
        name: Provider name (claude, openai, gemini, ollama)
        **kwargs: Provider configuration

    Returns:
        AVIRProvider instance
    """
    providers = {
        "claude": ClaudeProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }

    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")

    return providers[name](**kwargs)
