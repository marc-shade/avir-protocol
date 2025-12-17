#!/usr/bin/env python3
"""
AVIR Cross-Verification Example

Demonstrates the double-blind cross-provider verification protocol.

This is the RECOMMENDED way to use AVIR - multiple AI providers
verify each other's work, eliminating the "AI grading its own test" problem.
"""

import asyncio
from pathlib import Path

# Import AVIR components
from avir import (
    Specification,
    CrossVerifier,
    BlindingMode,
    IsolationLevel,
    cross_verify,
)
from avir.providers import ClaudeProvider, OpenAIProvider, GeminiProvider


async def main():
    """Run cross-provider verification example."""
    print("AVIR Cross-Provider Verification Example")
    print("=" * 50)
    print()

    # Load specification
    spec_path = Path(__file__).parent / "basic_spec.json"
    spec = Specification.from_file(spec_path)

    print(f"Specification: {spec.metadata.get('name', 'unnamed')}")
    print(f"Benchmarks: {len(spec.benchmarks)}")
    print()

    # Create providers (at least 2 required for cross-verification)
    # Note: In production, you'd have API keys configured
    providers = [
        ClaudeProvider(),      # Claude (Anthropic)
        OpenAIProvider(),      # GPT (OpenAI)
        GeminiProvider(),      # Gemini (Google)
    ]

    print("Providers configured:")
    for p in providers:
        print(f"  - {p.provider_id} ({p.model})")
    print()

    # Run cross-verification with double-blind protocol
    print("Running double-blind cross-verification...")
    print("  - Each provider verifies others' outputs")
    print("  - Providers don't know which provider generated the output")
    print("  - Self-verification excluded from consensus")
    print()

    # Option 1: Simple convenience function
    matrix = await cross_verify(
        spec=spec,
        providers=providers,
        blinding=BlindingMode.DOUBLE_BLIND,
        iterations=5,
    )

    # Option 2: More control with CrossVerifier class
    # verifier = CrossVerifier(
    #     providers=providers,
    #     blinding_mode=BlindingMode.DOUBLE_BLIND,
    #     isolation_level=IsolationLevel.PROCESS,
    #     iterations=5,
    # )
    # matrix = await verifier.verify(spec)

    # Display results
    print(matrix.summary())
    print()

    # Check consensus
    print("Verification Result:")
    if matrix.consensus_verdict.value == "VERIFIED":
        print("  ✓ VERIFIED - Cross-provider consensus achieved")
    elif matrix.consensus_verdict.value == "PARTIAL":
        print("  ~ PARTIAL - Some benchmarks passed, others failed")
    elif matrix.consensus_verdict.value == "FAILED":
        print("  ✗ FAILED - Benchmarks did not meet requirements")
    else:
        print(f"  ? {matrix.consensus_verdict.value} - Inconclusive result")

    print()
    print(f"Agreement Score: {matrix.agreement_score:.1%}")
    if matrix.agreement_score < 0.8:
        print("  Warning: Low agreement between providers")
        print("  This may indicate provider-specific biases or implementation differences")

    # Save results
    import json
    results_path = Path(__file__).parent / "cross_verification_results.json"
    with open(results_path, "w") as f:
        json.dump(matrix.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
