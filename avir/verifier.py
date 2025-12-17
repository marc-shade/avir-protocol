"""
AVIR Verifier

Main verification engine that orchestrates the complete AVIR
verification process: specification loading, benchmark execution,
result analysis, and attestation generation.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .attestation import (
    Attestation,
    EnvironmentInfo,
    ProviderInfo,
    VerificationLevel,
    create_attestation,
)
from .benchmark import Benchmark, BenchmarkSuite, BenchmarkVerdict
from .specification import Specification
from .verdict import Verdict, determine_overall_verdict


class AVIRVerifier:
    """
    Main AVIR verification engine.

    Orchestrates the complete verification process:
    1. Load and validate specification
    2. Initialize verification environment
    3. Execute benchmarks
    4. Analyze results
    5. Generate attestation

    Example:
        verifier = AVIRVerifier(provider="claude")
        results = await verifier.verify("./specs/system.yaml")
        print(f"Verdict: {results.attestation.verdict}")
    """

    def __init__(
        self,
        provider: str = "claude",
        model: Optional[str] = None,
        level: VerificationLevel = VerificationLevel.L2_STANDARD,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize verifier.

        Args:
            provider: AI provider (claude, openai, gemini, ollama)
            model: Specific model to use (auto-selected if not specified)
            level: Verification assurance level
            config: Additional configuration
        """
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.level = level
        self.config = config or {}

        self.instance_id = str(uuid.uuid4())[:8]
        self.specification: Optional[Specification] = None
        self.suite: Optional[BenchmarkSuite] = None
        self.environment: Optional[EnvironmentInfo] = None

    def _default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-4-turbo",
            "gemini": "gemini-pro",
            "ollama": "llama3.3",
        }
        return defaults.get(provider, "unknown")

    def _detect_environment(self) -> EnvironmentInfo:
        """Detect verification environment."""
        import os
        import platform
        import sys

        try:
            import psutil
            cpu_cores = psutil.cpu_count()
            memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            cpu_cores = os.cpu_count()
            memory_gb = None

        # Check if running in container
        container = None
        if Path("/.dockerenv").exists():
            container = "docker"
        elif Path("/run/.containerenv").exists():
            container = "podman"

        return EnvironmentInfo(
            container=container,
            os=platform.system().lower(),
            arch=platform.machine(),
            python_version=sys.version.split()[0],
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
        )

    def load_specification(self, spec_path: str | Path) -> Specification:
        """
        Load and validate specification.

        Args:
            spec_path: Path to specification file (YAML or JSON)

        Returns:
            Validated Specification object

        Raises:
            FileNotFoundError: If spec file doesn't exist
            ValueError: If spec is invalid
        """
        self.specification = Specification.from_file(spec_path)

        errors = self.specification.validate()
        if errors:
            raise ValueError(f"Invalid specification: {errors}")

        return self.specification

    def create_suite(
        self,
        executors: Dict[str, Callable] = None,
    ) -> BenchmarkSuite:
        """
        Create benchmark suite from specification.

        Args:
            executors: Optional custom executors for benchmarks

        Returns:
            BenchmarkSuite ready for execution
        """
        if not self.specification:
            raise RuntimeError("Load specification first")

        executors = executors or {}

        benchmarks = []
        for bench_spec in self.specification.benchmarks:
            executor = executors.get(bench_spec.id)
            benchmark = Benchmark(bench_spec, executor=executor)
            benchmarks.append(benchmark)

        self.suite = BenchmarkSuite(
            name=f"{self.specification.system.name} v{self.specification.system.version}",
            benchmarks=benchmarks,
        )

        return self.suite

    async def execute_benchmarks(
        self,
        context: Dict[str, Any] = None,
    ) -> List[BenchmarkVerdict]:
        """
        Execute all benchmarks in suite.

        Args:
            context: Execution context passed to benchmarks

        Returns:
            List of benchmark verdicts
        """
        if not self.suite:
            raise RuntimeError("Create benchmark suite first")

        context = context or {}
        context['provider'] = self.provider
        context['model'] = self.model
        context['level'] = self.level.value

        verdicts = await self.suite.execute_all(context)
        return verdicts

    def generate_attestation(
        self,
        started_at: datetime,
        completed_at: datetime,
        spec_source: Optional[str] = None,
    ) -> Attestation:
        """
        Generate attestation from verification results.

        Args:
            started_at: When verification started
            completed_at: When verification completed
            spec_source: Optional URL to specification

        Returns:
            Complete Attestation object
        """
        if not self.specification or not self.suite:
            raise RuntimeError("Run verification first")

        if not self.environment:
            self.environment = self._detect_environment()

        verifier_info = ProviderInfo(
            provider=self.provider,
            model=self.model,
            instance_id=self.instance_id,
        )

        # Get verdicts as dicts
        benchmark_verdicts = [v.to_dict() for v in self.suite.verdicts]

        return create_attestation(
            spec_hash=self.specification.hash,
            env=self.environment,
            verifier=verifier_info,
            benchmark_verdicts=benchmark_verdicts,
            system_name=self.specification.system.name,
            system_version=self.specification.system.version,
            started_at=started_at,
            completed_at=completed_at,
            results_hash=self.suite.get_results_hash(),
            level=self.level,
            spec_source=spec_source,
        )

    async def verify(
        self,
        spec_path: str | Path,
        executors: Dict[str, Callable] = None,
        context: Dict[str, Any] = None,
        spec_source: Optional[str] = None,
    ) -> "VerificationResult":
        """
        Complete verification workflow.

        This is the main entry point for AVIR verification.

        Args:
            spec_path: Path to specification file
            executors: Optional custom benchmark executors
            context: Execution context
            spec_source: Optional URL to specification

        Returns:
            VerificationResult with attestation and verdict
        """
        started_at = datetime.utcnow()

        # Initialize environment
        self.environment = self._detect_environment()

        # Load specification
        self.load_specification(spec_path)

        # Create benchmark suite
        self.create_suite(executors)

        # Execute benchmarks
        await self.execute_benchmarks(context)

        # Generate attestation
        completed_at = datetime.utcnow()
        attestation = self.generate_attestation(
            started_at=started_at,
            completed_at=completed_at,
            spec_source=spec_source,
        )

        # Determine overall verdict
        verdict = determine_overall_verdict(
            [v.to_dict() for v in self.suite.verdicts]
        )

        return VerificationResult(
            specification=self.specification,
            attestation=attestation,
            verdict=verdict,
            benchmarks=self.suite.verdicts,
        )


class VerificationResult:
    """
    Complete results from AVIR verification.

    Contains specification, attestation, verdict, and detailed
    benchmark results.
    """

    def __init__(
        self,
        specification: Specification,
        attestation: Attestation,
        verdict: Verdict,
        benchmarks: List[BenchmarkVerdict],
    ):
        self.specification = specification
        self.attestation = attestation
        self.verdict = verdict
        self.benchmarks = benchmarks

    @property
    def is_verified(self) -> bool:
        """Check if verification passed."""
        return self.verdict.is_success

    @property
    def hash(self) -> str:
        """Get attestation hash."""
        return self.attestation.hash

    def save(self, path: str | Path) -> None:
        """Save attestation to file."""
        self.attestation.save(str(path))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'system': {
                'name': self.specification.system.name,
                'version': self.specification.system.version,
            },
            'verdict': self.verdict.to_dict(),
            'attestation_hash': self.attestation.hash,
            'benchmarks': [b.to_dict() for b in self.benchmarks],
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "AVIR Verification Results",
            "=" * 60,
            f"System: {self.specification.system.name} v{self.specification.system.version}",
            f"Verdict: {self.verdict.level.value}",
            f"Pass Rate: {self.verdict.pass_rate:.1%}",
            f"Benchmarks: {self.verdict.passed}/{self.verdict.total} passed",
            "",
            "Attestation Hash:",
            f"  {self.attestation.hash}",
            "",
            "Benchmark Results:",
        ]

        for b in self.benchmarks:
            status = "PASS" if b.passed else "FAIL"
            lines.append(f"  [{status}] {b.benchmark_id}: {b.result:.2f} {b.unit}")

        if self.verdict.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.verdict.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 60)

        return "\n".join(lines)


async def quick_verify(
    spec_path: str,
    provider: str = "claude",
    level: VerificationLevel = VerificationLevel.L2_STANDARD,
) -> VerificationResult:
    """
    Quick verification with default settings.

    Args:
        spec_path: Path to specification file
        provider: AI provider to use
        level: Verification level

    Returns:
        VerificationResult
    """
    verifier = AVIRVerifier(provider=provider, level=level)
    return await verifier.verify(spec_path)
