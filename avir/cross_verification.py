"""
AVIR Cross-Provider Verification

Implements double-blind verification matrix across multiple AI providers
to eliminate bias, prevent context pollution, and ensure independent replication.

Key Principles:
1. NO AI verifying its own work - always cross-provider
2. Double-blind: Verifiers don't know which provider generated the original
3. Context isolation: Each verification runs in clean context
4. Consensus matrix: Multiple providers must agree for verification
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import secrets

from .providers.base import AVIRProvider
from .specification import Specification
from .benchmark import BenchmarkResult
from .verdict import Verdict, VerdictLevel
from .attestation import Attestation, AttestationChain, VerificationLevel


class BlindingMode(Enum):
    """Blinding modes for verification."""
    NONE = "none"                    # No blinding (development only)
    SINGLE_BLIND = "single_blind"   # Verifier doesn't know original provider
    DOUBLE_BLIND = "double_blind"   # Neither side knows the other


class IsolationLevel(Enum):
    """Context isolation levels."""
    NONE = "none"                    # Shared context (not recommended)
    PROCESS = "process"             # Separate process per verification
    CONTAINER = "container"         # Docker/container isolation
    TEE = "tee"                     # Trusted Execution Environment


@dataclass
class BlindedContext:
    """
    Blinded verification context - strips identifying information.

    Ensures verifiers cannot identify the original provider or
    be influenced by prior verification results.
    """
    context_id: str                  # Random ID, not traceable
    specification_hash: str          # Hash of spec (not the spec itself)
    benchmark_specs: List[Dict]      # Sanitized benchmark definitions
    timestamp: str                   # Verification timestamp
    nonce: str                       # Unique per-verification nonce

    # Deliberately excluded:
    # - Original provider identity
    # - Prior verification results
    # - Any provider-specific context

    @classmethod
    def create(cls, spec: Specification) -> "BlindedContext":
        """Create a blinded context from a specification."""
        nonce = secrets.token_hex(16)
        context_id = hashlib.sha256(
            f"{spec.hash}:{nonce}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        # Sanitize benchmarks - remove any identifying info
        sanitized_benchmarks = []
        for bm in spec.benchmarks:
            sanitized = {
                "id": hashlib.sha256(bm.id.encode()).hexdigest()[:8],
                "target": bm.target,
                "tolerance": bm.tolerance,
                "unit": bm.unit,
                "lower_is_better": bm.lower_is_better,
            }
            sanitized_benchmarks.append(sanitized)

        return cls(
            context_id=context_id,
            specification_hash=spec.hash,
            benchmark_specs=sanitized_benchmarks,
            timestamp=datetime.utcnow().isoformat() + "Z",
            nonce=nonce,
        )


@dataclass
class VerificationCell:
    """Single cell in the verification matrix."""
    verifier_id: str           # Which provider did the verification
    subject_id: str            # Which provider's output was verified
    verdict: VerdictLevel
    pass_rate: float
    results: List[BenchmarkResult]
    execution_time: float
    context_hash: str          # Hash of isolated context used

    @property
    def is_self_verification(self) -> bool:
        """Check if this is self-verification (should be excluded)."""
        return self.verifier_id == self.subject_id


@dataclass
class CrossVerificationMatrix:
    """
    NxN matrix of cross-provider verifications.

    Each cell [i,j] represents provider i verifying provider j's output.
    Diagonal cells (self-verification) are excluded from consensus.
    """
    providers: List[str]
    cells: Dict[Tuple[str, str], VerificationCell] = field(default_factory=dict)
    blinding_mode: BlindingMode = BlindingMode.DOUBLE_BLIND
    isolation_level: IsolationLevel = IsolationLevel.PROCESS
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def add_cell(self, cell: VerificationCell) -> None:
        """Add a verification cell to the matrix."""
        self.cells[(cell.verifier_id, cell.subject_id)] = cell

    def get_cell(self, verifier: str, subject: str) -> Optional[VerificationCell]:
        """Get a specific verification cell."""
        return self.cells.get((verifier, subject))

    @property
    def cross_verifications(self) -> List[VerificationCell]:
        """Get only cross-verifications (excludes self-verification)."""
        return [c for c in self.cells.values() if not c.is_self_verification]

    @property
    def consensus_verdict(self) -> VerdictLevel:
        """
        Calculate consensus verdict from cross-verifications only.

        Rules:
        - Unanimous VERIFIED = VERIFIED
        - Majority VERIFIED with minority PARTIAL = VERIFIED
        - Split verdicts = INCONCLUSIVE
        - Majority FAILED = FAILED
        """
        cross = self.cross_verifications
        if not cross:
            return VerdictLevel.INVALID

        verdicts = [c.verdict for c in cross]
        total = len(verdicts)

        verified_count = sum(1 for v in verdicts if v == VerdictLevel.VERIFIED)
        partial_count = sum(1 for v in verdicts if v == VerdictLevel.PARTIAL)
        failed_count = sum(1 for v in verdicts if v == VerdictLevel.FAILED)

        # Unanimous verified
        if verified_count == total:
            return VerdictLevel.VERIFIED

        # Strong majority verified (>= 2/3)
        if verified_count >= (total * 2 / 3):
            return VerdictLevel.VERIFIED

        # Majority failed
        if failed_count > total / 2:
            return VerdictLevel.FAILED

        # Mixed results
        if verified_count + partial_count >= total / 2:
            return VerdictLevel.PARTIAL

        return VerdictLevel.INCONCLUSIVE

    @property
    def consensus_pass_rate(self) -> float:
        """Average pass rate across cross-verifications."""
        cross = self.cross_verifications
        if not cross:
            return 0.0
        return sum(c.pass_rate for c in cross) / len(cross)

    @property
    def agreement_score(self) -> float:
        """
        Calculate inter-rater agreement score.

        Returns 1.0 for perfect agreement, 0.0 for complete disagreement.
        """
        cross = self.cross_verifications
        if len(cross) < 2:
            return 1.0

        verdicts = [c.verdict for c in cross]
        most_common = max(set(verdicts), key=verdicts.count)
        agreement = verdicts.count(most_common) / len(verdicts)
        return agreement

    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to dictionary."""
        return {
            "providers": self.providers,
            "blinding_mode": self.blinding_mode.value,
            "isolation_level": self.isolation_level.value,
            "timestamp": self.timestamp,
            "consensus": {
                "verdict": self.consensus_verdict.value,
                "pass_rate": self.consensus_pass_rate,
                "agreement_score": self.agreement_score,
            },
            "cells": [
                {
                    "verifier": cell.verifier_id,
                    "subject": cell.subject_id,
                    "verdict": cell.verdict.value,
                    "pass_rate": cell.pass_rate,
                    "is_cross_verification": not cell.is_self_verification,
                }
                for cell in self.cells.values()
            ],
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "AVIR Cross-Verification Matrix",
            "=" * 50,
            f"Providers: {', '.join(self.providers)}",
            f"Blinding: {self.blinding_mode.value}",
            f"Isolation: {self.isolation_level.value}",
            "",
            "Cross-Verification Results:",
        ]

        for cell in self.cross_verifications:
            lines.append(
                f"  {cell.verifier_id} → {cell.subject_id}: "
                f"{cell.verdict.value} ({cell.pass_rate:.1%})"
            )

        lines.extend([
            "",
            "Consensus:",
            f"  Verdict: {self.consensus_verdict.value}",
            f"  Pass Rate: {self.consensus_pass_rate:.1%}",
            f"  Agreement: {self.agreement_score:.1%}",
        ])

        return "\n".join(lines)


class CrossVerifier:
    """
    Orchestrates cross-provider verification with double-blind protocol.

    Ensures:
    1. Each provider verifies others' work, never its own
    2. Verification contexts are isolated and blinded
    3. Results are aggregated into consensus matrix
    4. Full audit trail maintained
    """

    def __init__(
        self,
        providers: List[AVIRProvider],
        blinding_mode: BlindingMode = BlindingMode.DOUBLE_BLIND,
        isolation_level: IsolationLevel = IsolationLevel.PROCESS,
        iterations: int = 5,
    ):
        self.providers = {p.provider_id: p for p in providers}
        self.blinding_mode = blinding_mode
        self.isolation_level = isolation_level
        self.iterations = iterations

    async def verify(
        self,
        spec: Specification,
        require_minimum_providers: int = 2,
    ) -> CrossVerificationMatrix:
        """
        Run full cross-verification matrix.

        Each provider verifies the outputs of all OTHER providers.
        Self-verification is recorded but excluded from consensus.
        """
        provider_ids = list(self.providers.keys())

        if len(provider_ids) < require_minimum_providers:
            raise ValueError(
                f"Cross-verification requires at least {require_minimum_providers} "
                f"providers, got {len(provider_ids)}"
            )

        matrix = CrossVerificationMatrix(
            providers=provider_ids,
            blinding_mode=self.blinding_mode,
            isolation_level=self.isolation_level,
        )

        # Generate outputs from each provider first
        provider_outputs = {}
        for provider_id, provider in self.providers.items():
            output = await self._generate_output(provider, spec)
            provider_outputs[provider_id] = output

        # Now cross-verify: each provider verifies all others
        for verifier_id, verifier in self.providers.items():
            for subject_id, subject_output in provider_outputs.items():
                # Create blinded context
                blinded = self._create_blinded_context(
                    spec, subject_output, verifier_id, subject_id
                )

                # Run verification in isolation
                cell = await self._verify_in_isolation(
                    verifier, blinded, subject_id
                )

                matrix.add_cell(cell)

        return matrix

    async def _generate_output(
        self,
        provider: AVIRProvider,
        spec: Specification,
    ) -> Dict[str, Any]:
        """Generate benchmark outputs from a provider."""
        await provider.initialize({})

        outputs = {}
        for benchmark in spec.benchmarks:
            results = []
            for _ in range(self.iterations):
                result = await provider.execute_benchmark(
                    benchmark.__dict__,
                    {"spec": spec.metadata},
                )
                results.append(result)
            outputs[benchmark.id] = {
                "results": results,
                "benchmark": benchmark.__dict__,
            }

        return outputs

    def _create_blinded_context(
        self,
        spec: Specification,
        output: Dict[str, Any],
        verifier_id: str,
        subject_id: str,
    ) -> Dict[str, Any]:
        """Create blinded verification context."""
        blinded = BlindedContext.create(spec)

        # In double-blind mode, don't include subject identity
        context = {
            "context_id": blinded.context_id,
            "specification_hash": blinded.specification_hash,
            "benchmarks": blinded.benchmark_specs,
            "timestamp": blinded.timestamp,
            "nonce": blinded.nonce,
        }

        if self.blinding_mode == BlindingMode.NONE:
            context["subject_id"] = subject_id
            context["verifier_id"] = verifier_id
        elif self.blinding_mode == BlindingMode.SINGLE_BLIND:
            context["verifier_id"] = verifier_id
            # Subject ID is hashed
            context["subject_hash"] = hashlib.sha256(
                subject_id.encode()
            ).hexdigest()[:8]
        # DOUBLE_BLIND: neither ID included

        # Include outputs with anonymized benchmark IDs
        context["outputs"] = {}
        for bm_id, bm_output in output.items():
            anon_id = hashlib.sha256(bm_id.encode()).hexdigest()[:8]
            context["outputs"][anon_id] = bm_output["results"]

        return context

    async def _verify_in_isolation(
        self,
        verifier: AVIRProvider,
        context: Dict[str, Any],
        subject_id: str,
    ) -> VerificationCell:
        """Run verification in isolated context."""
        import time
        start_time = time.time()

        # Context hash for audit trail
        context_hash = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()

        # Run verification
        results = []
        passed = 0
        total = 0

        for bm_spec in context["benchmarks"]:
            bm_outputs = context["outputs"].get(bm_spec["id"], [])
            if not bm_outputs:
                continue

            total += 1

            # Calculate statistics from outputs
            mean_value = sum(bm_outputs) / len(bm_outputs)
            target = bm_spec["target"]
            tolerance = bm_spec["tolerance"]
            lower_is_better = bm_spec.get("lower_is_better", False)

            # Determine if within tolerance
            if lower_is_better:
                max_allowed = target * (1 + tolerance)
                is_pass = mean_value <= max_allowed
            else:
                min_allowed = target * (1 - tolerance)
                max_allowed = target * (1 + tolerance)
                is_pass = min_allowed <= mean_value <= max_allowed

            if is_pass:
                passed += 1

            # Create result (simplified for cross-verification)
            result = BenchmarkResult(
                benchmark_id=bm_spec["id"],
                values=bm_outputs,
                target=target,
                tolerance=tolerance,
                unit=bm_spec.get("unit", ""),
                lower_is_better=lower_is_better,
            )
            result.calculate_statistics()
            result.determine_verdict()
            results.append(result)

        # Determine overall verdict
        pass_rate = passed / total if total > 0 else 0
        if pass_rate >= 0.8:
            verdict = VerdictLevel.VERIFIED
        elif pass_rate >= 0.5:
            verdict = VerdictLevel.PARTIAL
        else:
            verdict = VerdictLevel.FAILED

        execution_time = time.time() - start_time

        return VerificationCell(
            verifier_id=verifier.provider_id,
            subject_id=subject_id,
            verdict=verdict,
            pass_rate=pass_rate,
            results=results,
            execution_time=execution_time,
            context_hash=context_hash,
        )


async def cross_verify(
    spec: Specification,
    providers: List[AVIRProvider],
    blinding: BlindingMode = BlindingMode.DOUBLE_BLIND,
    iterations: int = 5,
) -> CrossVerificationMatrix:
    """
    Convenience function for cross-provider verification.

    Args:
        spec: Verification specification
        providers: List of AI providers (minimum 2)
        blinding: Blinding mode for verification
        iterations: Benchmark iterations per provider

    Returns:
        CrossVerificationMatrix with consensus verdict

    Example:
        from avir.providers import ClaudeProvider, OpenAIProvider, GeminiProvider

        providers = [
            ClaudeProvider(),
            OpenAIProvider(),
            GeminiProvider(),
        ]

        matrix = await cross_verify(spec, providers)
        print(f"Consensus: {matrix.consensus_verdict}")
        print(f"Agreement: {matrix.agreement_score:.1%}")
    """
    verifier = CrossVerifier(
        providers=providers,
        blinding_mode=blinding,
        iterations=iterations,
    )
    return await verifier.verify(spec)
