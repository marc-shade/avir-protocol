#!/usr/bin/env python3
"""
AVIR Advanced Verification Example

Demonstrates the enhanced cross-provider verification with:
- Statistical reliability measures (Fleiss' kappa, Krippendorff's alpha)
- Quorum configuration and validation
- Abstain/safety refusal handling
- Canary-based leak detection
- Provider-specific safety settings

This is production-ready verification code following expert panel recommendations.
"""

import asyncio
import json
from pathlib import Path
from dataclasses import asdict

from avir import (
    Specification,
    CrossVerifier,
    BlindingMode,
    IsolationLevel,
)
from avir.cross_verification import (
    QuorumConfig,
    calculate_fleiss_kappa,
    calculate_krippendorff_alpha,
)
from avir.providers import ClaudeProvider, OpenAIProvider, GeminiProvider
from avir.providers.claude import ClaudeConfig
from avir.providers.openai import OpenAIConfig
from avir.providers.gemini import GeminiConfig


def print_section(title: str):
    """Print section header."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


async def main():
    """Run advanced cross-provider verification with statistical analysis."""
    print_section("AVIR Advanced Verification Protocol")
    print("Using December 2025 model versions and expert panel recommendations")

    # Load specification
    spec_path = Path(__file__).parent / "basic_spec.json"
    spec = Specification.from_file(spec_path)

    print(f"\nSpecification: {spec.metadata.get('name', 'unnamed')}")
    print(f"Total benchmarks: {len(spec.benchmarks)}")

    # Configure providers with explicit safety settings
    print_section("Provider Configuration")

    # Claude with extended thinking disabled (faster verification)
    claude_config = ClaudeConfig(
        model="claude-sonnet-4-20250514",
        temperature=0.0,  # Deterministic
        treat_safety_refusal_as="abstain",  # Safety = ABSTAIN, not FAIL
        use_extended_thinking=False,
        pin_model_version=True,
    )

    # OpenAI with seed for reproducibility
    openai_config = OpenAIConfig(
        model="gpt-5.2-thinking",
        temperature=0.0,
        seed=42,  # Reproducible results
        treat_safety_refusal_as="abstain",
        pin_model_version=True,
    )

    # Gemini with top_k=1 for most deterministic output
    gemini_config = GeminiConfig(
        model="gemini-3-flash",
        temperature=0.0,
        top_k=1,  # Most deterministic
        treat_safety_refusal_as="abstain",
        safety_threshold="BLOCK_ONLY_HIGH",
    )

    providers = [
        ClaudeProvider(config=claude_config),
        OpenAIProvider(config=openai_config),
        GeminiProvider(config=gemini_config),
    ]

    print("\nProviders configured:")
    for p in providers:
        caps = p.get_capabilities()
        model_info = p.get_model_info()
        print(f"  - {p.provider_id}")
        print(f"    Model: {p.model}")
        print(f"    Organization: {p.get_organization()}")
        print(f"    Family: {model_info.get('model_family', 'unknown')}")
        print(f"    Safety handling: {p.config.treat_safety_refusal_as}")

    # Configure quorum requirements
    print_section("Quorum Configuration")

    quorum = QuorumConfig(
        minimum_providers=2,      # At least 2 providers must participate
        minimum_agreement=0.67,   # 2/3 majority required
        maximum_abstain_ratio=0.33,  # No more than 1/3 can abstain
        require_cross_org=True,   # Must have different organizations
    )

    print(f"Minimum providers: {quorum.minimum_providers}")
    print(f"Minimum agreement: {quorum.minimum_agreement:.0%}")
    print(f"Maximum abstain ratio: {quorum.maximum_abstain_ratio:.0%}")
    print(f"Require cross-organization: {quorum.require_cross_org}")

    # Create verifier with double-blind protocol
    print_section("Verification Execution")
    print("Running double-blind cross-verification...")
    print("  - BLAKE2b salted hashing for context blinding")
    print("  - Canary IDs for leak detection")
    print("  - Safety refusals tracked as ABSTAIN")

    verifier = CrossVerifier(
        providers=providers,
        blinding_mode=BlindingMode.DOUBLE_BLIND,
        isolation_level=IsolationLevel.PROCESS,
        iterations=5,
    )

    # Execute verification
    matrix = await verifier.verify(spec, require_minimum_providers=2)

    # Display matrix results
    print_section("Verification Results")
    print(matrix.summary())

    # Statistical analysis
    print_section("Statistical Reliability Analysis")

    # Extract ratings for statistical analysis
    # Convert verdicts to numeric: PASS=0, PARTIAL=1, FAIL=2, ABSTAIN=None
    verdict_map = {"VERIFIED": 0, "PARTIAL": 1, "FAILED": 2}

    # Build ratings matrix for Fleiss' kappa
    # Each row is an item (benchmark), each column count is category votes
    ratings_for_kappa = []
    ratings_for_alpha = []

    # Group cells by subject
    subjects = list(set(c.subject_id for c in matrix.cells))
    for subject in subjects:
        subject_cells = [c for c in matrix.cells if c.subject_id == subject and not c.is_self_verification]

        # Count verdicts for Fleiss' kappa (requires non-abstain)
        counts = [0, 0, 0]  # VERIFIED, PARTIAL, FAILED
        for cell in subject_cells:
            if cell.verdict.value in verdict_map:
                counts[verdict_map[cell.verdict.value]] += 1
        if sum(counts) > 0:
            ratings_for_kappa.append(counts)

        # Ratings for Krippendorff's alpha (handles abstains as None)
        alpha_row = []
        for cell in subject_cells:
            if cell.verdict.value == "ABSTAIN":
                alpha_row.append(None)
            elif cell.verdict.value in verdict_map:
                alpha_row.append(verdict_map[cell.verdict.value])
        if alpha_row:
            ratings_for_alpha.append(alpha_row)

    # Calculate reliability metrics
    if ratings_for_kappa:
        kappa = calculate_fleiss_kappa(ratings_for_kappa, categories=3)
        print(f"Fleiss' kappa (inter-rater reliability): {kappa:.3f}")

        # Interpret kappa
        if kappa > 0.8:
            print("  Interpretation: Almost perfect agreement")
        elif kappa > 0.6:
            print("  Interpretation: Substantial agreement")
        elif kappa > 0.4:
            print("  Interpretation: Moderate agreement")
        elif kappa > 0.2:
            print("  Interpretation: Fair agreement")
        else:
            print("  Interpretation: Slight or poor agreement")

    if ratings_for_alpha:
        alpha = calculate_krippendorff_alpha(ratings_for_alpha, categories=3)
        print(f"\nKrippendorff's alpha (with abstain handling): {alpha:.3f}")

        # Interpret alpha
        if alpha >= 0.8:
            print("  Interpretation: Reliable data")
        elif alpha >= 0.667:
            print("  Interpretation: Tentative conclusions possible")
        else:
            print("  Interpretation: Unreliable, needs investigation")

    # Abstain analysis
    print_section("Safety/Abstain Analysis")

    total_cells = len([c for c in matrix.cells if not c.is_self_verification])
    abstain_cells = len([c for c in matrix.cells if c.verdict.value == "ABSTAIN" and not c.is_self_verification])

    if total_cells > 0:
        abstain_ratio = abstain_cells / total_cells
        print(f"Total cross-verification cells: {total_cells}")
        print(f"Abstained (safety refusals): {abstain_cells}")
        print(f"Abstain ratio: {abstain_ratio:.1%}")

        if abstain_ratio > quorum.maximum_abstain_ratio:
            print(f"  WARNING: Exceeds maximum abstain ratio ({quorum.maximum_abstain_ratio:.0%})")
            print("  Quorum NOT met - verification inconclusive")
        else:
            print(f"  Within acceptable limits (< {quorum.maximum_abstain_ratio:.0%})")

    # Canary leak detection
    print_section("Canary Leak Detection")

    if hasattr(matrix, 'canary_leaks_detected'):
        if matrix.canary_leaks_detected:
            print("WARNING: Canary leaks detected!")
            print("  This indicates information leakage between providers")
            print("  Results may be compromised")
        else:
            print("No canary leaks detected")
            print("  Context isolation verified")
    else:
        print("Canary detection: Not available in this run")

    # Final verdict
    print_section("Final Verdict")

    consensus = matrix.consensus_verdict.value
    agreement = matrix.agreement_score

    print(f"Consensus: {consensus}")
    print(f"Agreement: {agreement:.1%}")

    # Quorum check
    quorum_met = (
        len(providers) >= quorum.minimum_providers and
        agreement >= quorum.minimum_agreement and
        (abstain_cells / total_cells if total_cells > 0 else 0) <= quorum.maximum_abstain_ratio
    )

    if quorum_met:
        print("\nQUORUM MET - Verification is VALID")
        if consensus == "VERIFIED":
            print("RESULT: System VERIFIED by cross-provider consensus")
        elif consensus == "PARTIAL":
            print("RESULT: System PARTIALLY VERIFIED - some benchmarks failed")
        else:
            print("RESULT: System FAILED verification")
    else:
        print("\nQUORUM NOT MET - Verification is INCONCLUSIVE")
        print("Possible reasons:")
        if len(providers) < quorum.minimum_providers:
            print(f"  - Insufficient providers ({len(providers)} < {quorum.minimum_providers})")
        if agreement < quorum.minimum_agreement:
            print(f"  - Low agreement ({agreement:.1%} < {quorum.minimum_agreement:.0%})")
        if total_cells > 0 and abstain_cells / total_cells > quorum.maximum_abstain_ratio:
            print(f"  - Too many abstains ({abstain_cells}/{total_cells})")

    # Save detailed results
    results_path = Path(__file__).parent / "advanced_verification_results.json"
    results = {
        "matrix": matrix.to_dict(),
        "statistical_analysis": {
            "fleiss_kappa": kappa if ratings_for_kappa else None,
            "krippendorff_alpha": alpha if ratings_for_alpha else None,
        },
        "quorum": {
            "config": asdict(quorum),
            "met": quorum_met,
        },
        "abstain_analysis": {
            "total_cells": total_cells,
            "abstain_cells": abstain_cells,
            "abstain_ratio": abstain_ratio if total_cells > 0 else 0,
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
