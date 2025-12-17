"""
AVIR Provider Base Class

Abstract base class defining the interface for AI verification providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AVIRProvider(ABC):
    """
    Abstract base class for AVIR verification providers.

    All AI provider implementations must inherit from this class
    and implement the required methods.
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
    ) -> float:
        """
        Execute a benchmark and return the result.

        Args:
            benchmark_spec: Benchmark specification
            context: Execution context

        Returns:
            Benchmark result value
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
        }
