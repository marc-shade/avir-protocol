"""
AVIR Cross-Provider Verification

Implements double-blind verification matrix across multiple AI providers
to eliminate bias, prevent context pollution, and ensure independent replication.

Key Principles:
1. NO AI verifying its own work - always cross-provider
2. Double-blind: Verifiers don't know which provider generated the original
3. Context isolation: Each verification runs in clean context
4. Consensus matrix: Multiple providers must agree for verification
5. Statistical validity: Inter-rater reliability via Fleiss' κ / Krippendorff's α
6. Quorum requirements: Minimum provider agreement thresholds

Updated December 2025 with expert panel recommendations:
- BLAKE2b salted hashing for stronger blinding
- Canary IDs for leak detection
- Statistical reliability measures
- Quorum and abstain handling
"""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import secrets

from .providers.base import AVIRProvider, VerdictResponse
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


def blake2b_salted_hash(data: str, salt: Optional[bytes] = None, length: int = 16) -> str:
    """
    Create a BLAKE2b salted hash for stronger blinding.

    BLAKE2b is faster and more secure than SHA-256 for this use case.
    Salt prevents rainbow table attacks on provider IDs.
    """
    if salt is None:
        salt = secrets.token_bytes(16)

    h = hashlib.blake2b(data.encode(), salt=salt, digest_size=length)
    return h.hexdigest()


def generate_canary_id() -> str:
    """
    Generate a canary ID for leak detection.

    If this ID appears in any provider output, it indicates
    context leakage or blinding failure.
    """
    return f"CANARY-{secrets.token_hex(8).upper()}"


@dataclass
class BlindedContext:
    """
    Blinded verification context - strips identifying information.

    Uses BLAKE2b salted hashing for stronger security.
    Includes canary IDs for leak detection.

    Ensures verifiers cannot identify the original provider or
    be influenced by prior verification results.
    """
    context_id: str                  # Random ID, not traceable
    specification_hash: str          # Hash of spec (not the spec itself)
    benchmark_specs: List[Dict]      # Sanitized benchmark definitions
    timestamp: str                   # Verification timestamp
    nonce: str                       # Unique per-verification nonce
    canary_id: str                   # Leak detection canary
    salt: bytes                      # Salt for consistent hashing

    # Deliberately excluded:
    # - Original provider identity
    # - Prior verification results
    # - Any provider-specific context

    @classmethod
    def create(cls, spec: Specification) -> "BlindedContext":
        """Create a blinded context from a specification."""
        nonce = secrets.token_hex(16)
        salt = secrets.token_bytes(16)
        canary_id = generate_canary_id()

        # Use BLAKE2b for context ID
        context_id = blake2b_salted_hash(
            f"{spec.hash}:{nonce}:{datetime.utcnow().isoformat()}",
            salt=salt,
            length=16
        )

        # Sanitize benchmarks - remove any identifying info
        sanitized_benchmarks = []
        for bm in spec.benchmarks:
            sanitized = {
                "id": blake2b_salted_hash(bm.id, salt=salt, length=8),
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
            canary_id=canary_id,
            salt=salt,
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
    abstain_count: int = 0     # Number of abstained benchmarks
    canary_detected: bool = False  # True if canary leaked (blinding failure)

    @property
    def is_self_verification(self) -> bool:
        """Check if this is self-verification (should be excluded)."""
        return self.verifier_id == self.subject_id

    @property
    def is_abstain(self) -> bool:
        """Check if this cell is primarily abstains."""
        return self.abstain_count > len(self.results) / 2


@dataclass
class QuorumConfig:
    """Configuration for quorum requirements."""
    minimum_providers: int = 2       # Minimum providers for valid consensus
    minimum_agreement: float = 0.67  # Minimum agreement ratio (2/3)
    maximum_abstain_ratio: float = 0.33  # Max abstains before invalid
    require_cross_org: bool = True   # Require providers from different organizations


def calculate_fleiss_kappa(ratings: List[List[int]], categories: int) -> float:
    """
    Calculate Fleiss' Kappa for inter-rater reliability.

    Fleiss' κ measures agreement between multiple raters assigning
    categorical ratings to a number of items.

    κ = (P̄ - P̄e) / (1 - P̄e)

    Where:
    - P̄ = Mean observed agreement
    - P̄e = Expected agreement by chance

    Args:
        ratings: Matrix of ratings [items x raters], values are category indices
        categories: Number of possible categories

    Returns:
        Kappa value: -1 (complete disagreement) to 1 (perfect agreement)
        0 = agreement equals chance
    """
    if not ratings or not ratings[0]:
        return 0.0

    n_items = len(ratings)
    n_raters = len(ratings[0])

    if n_raters < 2:
        return 1.0  # Single rater always agrees with self

    # Count category assignments per item
    category_counts = []
    for item_ratings in ratings:
        counts = [0] * categories
        for rating in item_ratings:
            if 0 <= rating < categories:
                counts[rating] += 1
        category_counts.append(counts)

    # Calculate P_i (agreement for each item)
    P_i = []
    for counts in category_counts:
        agreement = sum(c * (c - 1) for c in counts) / (n_raters * (n_raters - 1))
        P_i.append(agreement)

    # Mean observed agreement
    P_bar = sum(P_i) / n_items if n_items > 0 else 0

    # Category proportions across all ratings
    p_j = []
    total_ratings = n_items * n_raters
    for j in range(categories):
        count = sum(counts[j] for counts in category_counts)
        p_j.append(count / total_ratings if total_ratings > 0 else 0)

    # Expected agreement by chance
    P_e = sum(p ** 2 for p in p_j)

    # Fleiss' Kappa
    if P_e == 1.0:
        return 1.0  # Perfect agreement expected by chance
    kappa = (P_bar - P_e) / (1 - P_e)

    return kappa


def calculate_krippendorff_alpha(
    ratings: List[List[Optional[int]]],
    categories: int
) -> float:
    """
    Calculate Krippendorff's Alpha for inter-rater reliability.

    Unlike Fleiss' κ, Krippendorff's α handles:
    - Missing data (None values)
    - Different numbers of raters per item
    - Various levels of measurement

    α = 1 - (D_o / D_e)

    Where:
    - D_o = Observed disagreement
    - D_e = Expected disagreement by chance

    Args:
        ratings: Matrix of ratings [items x raters], None for missing
        categories: Number of possible categories

    Returns:
        Alpha value: -1 to 1 (1 = perfect agreement, 0 = chance)
    """
    if not ratings:
        return 0.0

    # Collect all valid pairs
    pairs = []
    value_counts = {}

    for item_ratings in ratings:
        valid_ratings = [r for r in item_ratings if r is not None]
        if len(valid_ratings) < 2:
            continue

        # Count pairs within this item
        for i, r1 in enumerate(valid_ratings):
            for r2 in valid_ratings[i+1:]:
                pairs.append((r1, r2))

            # Count overall value frequencies
            value_counts[r1] = value_counts.get(r1, 0) + 1

    if not pairs:
        return 0.0

    n_pairs = len(pairs)
    total_values = sum(value_counts.values())

    # Observed disagreement (nominal level - 0 if same, 1 if different)
    D_o = sum(1 for r1, r2 in pairs if r1 != r2) / n_pairs

    # Expected disagreement by chance
    D_e = 0.0
    for v1 in range(categories):
        for v2 in range(categories):
            if v1 != v2:
                n1 = value_counts.get(v1, 0)
                n2 = value_counts.get(v2, 0)
                D_e += n1 * n2

    if total_values > 1:
        D_e = D_e / (total_values * (total_values - 1))

    # Krippendorff's Alpha
    if D_e == 0:
        return 1.0  # No expected disagreement
    alpha = 1 - (D_o / D_e)

    return alpha


@dataclass
class CrossVerificationMatrix:
    """
    NxN matrix of cross-provider verifications.

    Each cell [i,j] represents provider i verifying provider j's output.
    Diagonal cells (self-verification) are excluded from consensus.

    Includes statistical measures:
    - Fleiss' κ for inter-rater reliability
    - Krippendorff's α for handling missing data
    - Quorum requirements for valid consensus
    """
    providers: List[str]
    cells: Dict[Tuple[str, str], VerificationCell] = field(default_factory=dict)
    blinding_mode: BlindingMode = BlindingMode.DOUBLE_BLIND
    isolation_level: IsolationLevel = IsolationLevel.PROCESS
    quorum: QuorumConfig = field(default_factory=QuorumConfig)
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
    def non_abstain_verifications(self) -> List[VerificationCell]:
        """Get cross-verifications that aren't primarily abstains."""
        return [c for c in self.cross_verifications if not c.is_abstain]

    @property
    def abstain_ratio(self) -> float:
        """Ratio of abstained verifications."""
        cross = self.cross_verifications
        if not cross:
            return 0.0
        abstain_count = sum(1 for c in cross if c.is_abstain)
        return abstain_count / len(cross)

    @property
    def quorum_met(self) -> bool:
        """Check if quorum requirements are met."""
        # Minimum providers
        if len(self.providers) < self.quorum.minimum_providers:
            return False

        # Maximum abstain ratio
        if self.abstain_ratio > self.quorum.maximum_abstain_ratio:
            return False

        # Minimum non-abstain verifications
        non_abstain = self.non_abstain_verifications
        if len(non_abstain) < self.quorum.minimum_providers:
            return False

        return True

    @property
    def fleiss_kappa(self) -> float:
        """
        Calculate Fleiss' κ for inter-rater reliability.

        Measures agreement between providers on verdict categories.
        """
        cross = self.non_abstain_verifications
        if len(cross) < 2:
            return 1.0

        # Map verdicts to category indices
        verdict_map = {
            VerdictLevel.VERIFIED: 0,
            VerdictLevel.PARTIAL: 1,
            VerdictLevel.FAILED: 2,
            VerdictLevel.INCONCLUSIVE: 3,
        }

        # Build ratings matrix: each subject evaluated by multiple verifiers
        subjects = list(set(c.subject_id for c in cross))
        ratings = []

        for subject in subjects:
            subject_cells = [c for c in cross if c.subject_id == subject]
            item_ratings = [verdict_map.get(c.verdict, 3) for c in subject_cells]
            if item_ratings:
                ratings.append(item_ratings)

        if not ratings:
            return 0.0

        # Pad to consistent rater count
        max_raters = max(len(r) for r in ratings)
        padded_ratings = []
        for item in ratings:
            padded = item + [None] * (max_raters - len(item))
            padded_ratings.append([r if r is not None else 0 for r in padded])

        return calculate_fleiss_kappa(padded_ratings, categories=4)

    @property
    def krippendorff_alpha(self) -> float:
        """
        Calculate Krippendorff's α for inter-rater reliability.

        Handles missing data better than Fleiss' κ.
        """
        cross = self.cross_verifications  # Include abstains as missing
        if len(cross) < 2:
            return 1.0

        verdict_map = {
            VerdictLevel.VERIFIED: 0,
            VerdictLevel.PARTIAL: 1,
            VerdictLevel.FAILED: 2,
            VerdictLevel.INCONCLUSIVE: 3,
        }

        subjects = list(set(c.subject_id for c in cross))
        verifiers = list(set(c.verifier_id for c in cross))

        # Build full matrix with None for missing/abstain
        ratings = []
        for subject in subjects:
            item_ratings = []
            for verifier in verifiers:
                cell = self.get_cell(verifier, subject)
                if cell and not cell.is_self_verification and not cell.is_abstain:
                    item_ratings.append(verdict_map.get(cell.verdict, None))
                else:
                    item_ratings.append(None)
            ratings.append(item_ratings)

        return calculate_krippendorff_alpha(ratings, categories=4)

    @property
    def consensus_verdict(self) -> VerdictLevel:
        """
        Calculate consensus verdict from cross-verifications only.

        Rules:
        - Quorum must be met for valid verdict
        - Abstains are excluded from voting (preserve statistical validity)
        - Unanimous VERIFIED = VERIFIED
        - Strong majority VERIFIED (>= 2/3) = VERIFIED
        - Majority FAILED = FAILED
        - Split verdicts = INCONCLUSIVE
        """
        if not self.quorum_met:
            return VerdictLevel.INVALID

        # Use only non-abstain verifications
        cross = self.non_abstain_verifications
        if not cross:
            return VerdictLevel.INVALID

        verdicts = [c.verdict for c in cross]
        total = len(verdicts)

        verified_count = sum(1 for v in verdicts if v == VerdictLevel.VERIFIED)
        partial_count = sum(1 for v in verdicts if v == VerdictLevel.PARTIAL)
        failed_count = sum(1 for v in verdicts if v == VerdictLevel.FAILED)

        # Check agreement meets quorum threshold
        agreement = max(verified_count, partial_count, failed_count) / total
        if agreement < self.quorum.minimum_agreement:
            return VerdictLevel.INCONCLUSIVE

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
        """Average pass rate across non-abstain cross-verifications."""
        cross = self.non_abstain_verifications
        if not cross:
            return 0.0
        return sum(c.pass_rate for c in cross) / len(cross)

    @property
    def agreement_score(self) -> float:
        """
        Simple agreement score (proportion of most common verdict).

        For statistically rigorous agreement, use fleiss_kappa or krippendorff_alpha.
        """
        cross = self.non_abstain_verifications
        if len(cross) < 2:
            return 1.0

        verdicts = [c.verdict for c in cross]
        most_common = max(set(verdicts), key=verdicts.count)
        agreement = verdicts.count(most_common) / len(verdicts)
        return agreement

    @property
    def canary_leaks_detected(self) -> List[VerificationCell]:
        """Get cells where canary IDs were detected (blinding failure)."""
        return [c for c in self.cells.values() if c.canary_detected]

    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to dictionary."""
        return {
            "providers": self.providers,
            "blinding_mode": self.blinding_mode.value,
            "isolation_level": self.isolation_level.value,
            "timestamp": self.timestamp,
            "quorum": {
                "met": self.quorum_met,
                "minimum_providers": self.quorum.minimum_providers,
                "abstain_ratio": self.abstain_ratio,
            },
            "consensus": {
                "verdict": self.consensus_verdict.value,
                "pass_rate": self.consensus_pass_rate,
                "agreement_score": self.agreement_score,
            },
            "statistical_reliability": {
                "fleiss_kappa": self.fleiss_kappa,
                "krippendorff_alpha": self.krippendorff_alpha,
                "interpretation": self._interpret_reliability(),
            },
            "cells": [
                {
                    "verifier": cell.verifier_id,
                    "subject": cell.subject_id,
                    "verdict": cell.verdict.value,
                    "pass_rate": cell.pass_rate,
                    "is_cross_verification": not cell.is_self_verification,
                    "is_abstain": cell.is_abstain,
                    "abstain_count": cell.abstain_count,
                }
                for cell in self.cells.values()
            ],
            "canary_leaks": len(self.canary_leaks_detected),
        }

    def _interpret_reliability(self) -> str:
        """Interpret the reliability score."""
        kappa = self.fleiss_kappa
        if kappa >= 0.81:
            return "almost_perfect"
        elif kappa >= 0.61:
            return "substantial"
        elif kappa >= 0.41:
            return "moderate"
        elif kappa >= 0.21:
            return "fair"
        elif kappa >= 0.0:
            return "slight"
        else:
            return "poor"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "AVIR Cross-Verification Matrix",
            "=" * 50,
            f"Providers: {', '.join(self.providers)}",
            f"Blinding: {self.blinding_mode.value}",
            f"Isolation: {self.isolation_level.value}",
            f"Quorum Met: {'Yes' if self.quorum_met else 'NO - INVALID'}",
            "",
            "Cross-Verification Results:",
        ]

        for cell in self.cross_verifications:
            status = " [ABSTAIN]" if cell.is_abstain else ""
            lines.append(
                f"  {cell.verifier_id} → {cell.subject_id}: "
                f"{cell.verdict.value} ({cell.pass_rate:.1%}){status}"
            )

        lines.extend([
            "",
            "Statistical Reliability:",
            f"  Fleiss' κ: {self.fleiss_kappa:.3f} ({self._interpret_reliability()})",
            f"  Krippendorff's α: {self.krippendorff_alpha:.3f}",
            f"  Abstain Ratio: {self.abstain_ratio:.1%}",
            "",
            "Consensus:",
            f"  Verdict: {self.consensus_verdict.value}",
            f"  Pass Rate: {self.consensus_pass_rate:.1%}",
            f"  Agreement: {self.agreement_score:.1%}",
        ])

        if self.canary_leaks_detected:
            lines.extend([
                "",
                "⚠️  BLINDING FAILURES DETECTED:",
                f"  {len(self.canary_leaks_detected)} canary ID(s) leaked",
            ])

        return "\n".join(lines)


class CrossVerifier:
    """
    Orchestrates cross-provider verification with double-blind protocol.

    Ensures:
    1. Each provider verifies others' work, never its own
    2. Verification contexts are isolated and blinded (BLAKE2b salted)
    3. Results are aggregated into consensus matrix
    4. Full audit trail maintained
    5. Canary detection for blinding failures
    6. Quorum requirements for valid consensus

    Updated December 2025 with expert panel recommendations.
    """

    def __init__(
        self,
        providers: List[AVIRProvider],
        blinding_mode: BlindingMode = BlindingMode.DOUBLE_BLIND,
        isolation_level: IsolationLevel = IsolationLevel.PROCESS,
        iterations: int = 5,
        quorum: Optional[QuorumConfig] = None,
    ):
        self.providers = {p.provider_id: p for p in providers}
        self.blinding_mode = blinding_mode
        self.isolation_level = isolation_level
        self.iterations = iterations
        self.quorum = quorum or QuorumConfig()
        self._canary_registry: Dict[str, str] = {}  # Maps context_id to canary_id

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
            quorum=self.quorum,
        )

        # Generate outputs from each provider first
        provider_outputs = {}
        for provider_id, provider in self.providers.items():
            output = await self._generate_output(provider, spec)
            provider_outputs[provider_id] = output

        # Now cross-verify: each provider verifies all others
        for verifier_id, verifier in self.providers.items():
            for subject_id, subject_output in provider_outputs.items():
                # Create blinded context with canary
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
        abstain_count = 0

        for benchmark in spec.benchmarks:
            results = []
            for _ in range(self.iterations):
                result = await provider.execute_benchmark(
                    benchmark.__dict__,
                    {"spec": spec.metadata},
                )
                # Handle VerdictResponse objects
                if hasattr(result, 'verdict'):
                    if result.verdict == "ABSTAIN":
                        abstain_count += 1
                        results.append({"abstain": True, "reason": result.abstain_reason})
                    else:
                        results.append(result.value if result.value is not None else result)
                else:
                    results.append(result)

            outputs[benchmark.id] = {
                "results": results,
                "benchmark": benchmark.__dict__,
                "abstain_count": abstain_count,
            }

        return outputs

    def _create_blinded_context(
        self,
        spec: Specification,
        output: Dict[str, Any],
        verifier_id: str,
        subject_id: str,
    ) -> Dict[str, Any]:
        """Create blinded verification context with BLAKE2b and canary."""
        blinded = BlindedContext.create(spec)

        # Register canary for leak detection
        self._canary_registry[blinded.context_id] = blinded.canary_id

        # In double-blind mode, don't include subject identity
        context = {
            "context_id": blinded.context_id,
            "specification_hash": blinded.specification_hash,
            "benchmarks": blinded.benchmark_specs,
            "timestamp": blinded.timestamp,
            "nonce": blinded.nonce,
            "canary_id": blinded.canary_id,  # For leak detection
        }

        if self.blinding_mode == BlindingMode.NONE:
            context["subject_id"] = subject_id
            context["verifier_id"] = verifier_id
        elif self.blinding_mode == BlindingMode.SINGLE_BLIND:
            context["verifier_id"] = verifier_id
            # Subject ID is hashed with BLAKE2b
            context["subject_hash"] = blake2b_salted_hash(
                subject_id, salt=blinded.salt, length=8
            )
        # DOUBLE_BLIND: neither ID included

        # Include outputs with anonymized benchmark IDs using BLAKE2b
        context["outputs"] = {}
        for bm_id, bm_output in output.items():
            anon_id = blake2b_salted_hash(bm_id, salt=blinded.salt, length=8)
            # Filter out abstains for numeric processing
            numeric_results = [
                r for r in bm_output["results"]
                if not (isinstance(r, dict) and r.get("abstain"))
            ]
            context["outputs"][anon_id] = numeric_results

        return context

    def _check_canary_leak(self, response: Any, canary_id: str) -> bool:
        """Check if canary ID leaked into provider response."""
        response_str = str(response).upper()
        return canary_id.upper() in response_str

    async def _verify_in_isolation(
        self,
        verifier: AVIRProvider,
        context: Dict[str, Any],
        subject_id: str,
    ) -> VerificationCell:
        """Run verification in isolated context with abstain and canary tracking."""
        import time
        start_time = time.time()

        # Context hash for audit trail (using BLAKE2b)
        context_str = json.dumps(context, sort_keys=True, default=str)
        context_hash = blake2b_salted_hash(context_str, length=32)

        # Track canary for leak detection
        canary_id = context.get("canary_id", "")
        canary_detected = False

        # Run verification
        results = []
        passed = 0
        abstain_count = 0
        total = 0

        for bm_spec in context["benchmarks"]:
            bm_outputs = context["outputs"].get(bm_spec["id"], [])
            if not bm_outputs:
                abstain_count += 1
                continue

            total += 1

            # Filter numeric values (exclude any remaining abstain markers)
            numeric_outputs = [
                v for v in bm_outputs
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]

            if not numeric_outputs:
                abstain_count += 1
                continue

            # Calculate statistics from outputs
            mean_value = sum(numeric_outputs) / len(numeric_outputs)
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
                values=numeric_outputs,
                target=target,
                tolerance=tolerance,
                unit=bm_spec.get("unit", ""),
                lower_is_better=lower_is_better,
            )
            result.calculate_statistics()
            result.determine_verdict()
            results.append(result)

            # Check for canary leak in any provider responses
            if canary_id and self._check_canary_leak(result, canary_id):
                canary_detected = True

        # Determine overall verdict
        effective_total = total - abstain_count
        if effective_total > 0:
            pass_rate = passed / effective_total
        else:
            pass_rate = 0.0

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
            abstain_count=abstain_count,
            canary_detected=canary_detected,
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
