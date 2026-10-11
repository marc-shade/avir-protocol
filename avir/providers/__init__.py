"""
AVIR Provider Implementations

AI provider integrations for AVIR verification.
"""

from typing import Any

from .base import AVIRProvider
from .claude import ClaudeProvider
from .cli import (
    ClaudeCLIProvider,
    CLIProvider,
    CodexCLIProvider,
    GeminiCLIProvider,
    HTTPModelProvider,
    ailab_model,
    fedora_ollama,
)
from .gemini import GeminiProvider
from .openai import OpenAIProvider

__all__ = [
    "AVIRProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "GeminiProvider",
    # Real-model providers (CLIs + cluster HTTP endpoints, no SDK)
    "CLIProvider",
    "ClaudeCLIProvider",
    "CodexCLIProvider",
    "GeminiCLIProvider",
    "HTTPModelProvider",
    "fedora_ollama",
    "ailab_model",
]


def get_provider(name: str, **kwargs: Any) -> AVIRProvider:
    """
    Get provider instance by name.

    Args:
        name: Provider name (claude, openai, gemini, ollama)
        **kwargs: Provider configuration

    Returns:
        AVIRProvider instance
    """
    providers: dict[str, type[AVIRProvider]] = {
        "claude": ClaudeProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        # Real-model, SDK-free adapters
        "claude-cli": ClaudeCLIProvider,
        "codex-cli": CodexCLIProvider,
        "gemini-cli": GeminiCLIProvider,
    }

    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")

    return providers[name](**kwargs)
