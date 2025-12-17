"""
AVIR Provider Base Class

Abstract base class defining the interface for AI verification providers.
Updated December 2025 with enhanced configuration and verdict handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ProviderConfig:
    """Base configuration for all providers."""
    model: str = ""
    temperature: float = 0.0  # Default to deterministic for verification
    max_tokens: int = 4096

    # Safety refusal handling
    treat_safety_refusal_as: str = "abstain"  # abstain, fail, retry

    # Version pinning
    pin_model_version: bool = True


@dataclass
class VerdictResponse:
    """
    Response from benchmark execution.

    Supports three verdict types:
    - PASS: Benchmark passed within tolerance
    - FAIL: Benchmark failed (outside tolerance)
    - ABSTAIN: Provider abstained (safety refusal, timeout, etc.)
    """
    value: Optional[float]
    verdict: str  # PASS, FAIL, ABSTAIN
    abstain_reason: Optional[str] = None
    provider_metadata: Optional[Dict] = None
    confidence: Optional[float] = None  # 0.0 to 1.0

    def is_abstain(self) -> bool:
        return self.verdict == "ABSTAIN"

    def is_pass(self) -> bool:
        return self.verdict == "PASS"

    def is_fail(self) -> bool:
        return self.verdict == "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "verdict": self.verdict,
            "abstain_reason": self.abstain_reason,
            "provider_metadata": self.provider_metadata,
            "confidence": self.confidence,
        }


class AVIRProvider(ABC):
    """
    Abstract base class for AVIR verification providers.

    All AI provider implementations must inherit from this class
    and implement the required methods.

    Key Design Principles (from expert panel review):
    1. Deterministic execution: temperature=0.0 by default
    2. Safety refusals as abstain: Not FAIL (preserves statistical validity)
    3. Version pinning: Model version tracked for attestation
    4. Capability reporting: Honest capability flags for routing
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model being used."""
        pass

    @property
    def config(self) -> ProviderConfig:
        """Provider configuration."""
        return ProviderConfig()

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize provider with configuration.

        Args:
            config: Provider configuration including API keys
        """
        pass

    @abstractmethod
    async def execute_benchmark(
        self,
        benchmark_spec: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Union[float, VerdictResponse]:
        """
        Execute a benchmark and return the result.

        Args:
            benchmark_spec: Benchmark specification
            context: Execution context

        Returns:
            Benchmark result value or VerdictResponse
        """
        pass

    @abstractmethod
    async def analyze_results(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        AI analysis of verification results.

        Args:
            results: List of benchmark results

        Returns:
            Analysis including insights and recommendations
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        pass

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get provider capabilities.

        Returns:
            Dict of capability flags
        """
        return {
            "code_execution": False,
            "multimodal": False,
            "function_calling": False,
            "streaming": False,
            "deterministic_seed": False,
            "large_context": False,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model information for attestation.

        Should include:
        - provider: Provider identifier
        - model: Model identifier
        - model_family: Model family (e.g., "GPT-5.2", "Gemini-3")
        - config: Relevant configuration
        """
        return {
            "provider": self.provider_id,
            "model": self.model,
            "config": {
                "temperature": self.config.temperature,
                "safety_refusal_handling": self.config.treat_safety_refusal_as,
            },
        }

    def get_organization(self) -> str:
        """
        Return the organization that created this provider.

        Used for ensuring cross-organization verification
        (different organizations = different training biases).
        """
        org_map = {
            "claude": "anthropic",
            "openai": "openai",
            "gemini": "google",
            "ollama": "local",
        }
        return org_map.get(self.provider_id, "unknown")

    def get_model_backbone(self) -> str:
        """
        Return the model backbone/architecture.

        Used for ensuring diverse model architectures
        (same backbone may share biases).
        """
        return f"{self.get_organization()}:{self.model}"
