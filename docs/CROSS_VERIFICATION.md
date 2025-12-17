# Cross-Provider Verification Guide

## Overview

Cross-provider verification is the **core innovation** of AVIR. It solves the fundamental problem of AI systems verifying their own capabilities by requiring multiple independent AI providers to verify each other using a double-blind protocol.

## The Problem

When a single AI provider runs benchmarks on itself, several issues arise:

1. **Self-Interest Bias**: The provider may unconsciously favor its own outputs
2. **Shared Training Biases**: Same training data produces same blind spots
3. **Context Leakage**: Prior context can influence verification results
4. **Gaming Risk**: Systems can be optimized to pass specific tests
5. **Lack of Independence**: No true independent verification

**Analogy**: An AI verifying its own benchmarks is like a student grading their own test.

## The Solution: Cross-Provider Verification

AVIR requires **different AI providers** to verify each other's outputs:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROSS-VERIFICATION MATRIX                     │
├─────────────────┬───────────────────────────────────────────────┤
│                 │           SUBJECT (Being Verified)            │
│                 ├─────────────┬─────────────┬─────────────┬─────┤
│                 │   Claude    │   OpenAI    │   Gemini    │     │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────┤
│ V  │   Claude   │  (excluded) │   ✓ PASS    │   ✓ PASS    │     │
│ E  ├─────────────┼─────────────┼─────────────┼─────────────┤     │
│ R  │   OpenAI   │   ✓ PASS    │  (excluded) │   ✓ PASS    │     │
│ I  ├─────────────┼─────────────┼─────────────┼─────────────┤     │
│ F  │   Gemini   │   ✓ PASS    │   ~ PARTIAL │  (excluded) │     │
│ I  ├─────────────┼─────────────┼─────────────┼─────────────┤     │
│ E  │            │             │             │             │     │
│ R  │ CONSENSUS  │  VERIFIED   │   PARTIAL   │  VERIFIED   │     │
└────┴────────────┴─────────────┴─────────────┴─────────────┴─────┘
```

Key principles:
- **Diagonal cells (self-verification) are excluded** from consensus
- **Only cross-verification cells** determine the final verdict
- **Multiple providers must agree** for verification

## Blinding Modes

### NONE (Development Only)

```python
from avir import BlindingMode

# No blinding - both parties know each other
# Use only for debugging
matrix = await cross_verify(spec, providers, blinding=BlindingMode.NONE)
```

- Verifier knows the subject provider
- Subject knows the verifier
- **Not recommended for production**

### SINGLE_BLIND

```python
# Verifier doesn't know which provider generated the output
matrix = await cross_verify(spec, providers, blinding=BlindingMode.SINGLE_BLIND)
```

- Verifier receives anonymized outputs
- Subject identity is hashed
- Good for basic verification

### DOUBLE_BLIND (Recommended)

```python
# Neither party knows the other
matrix = await cross_verify(spec, providers, blinding=BlindingMode.DOUBLE_BLIND)
```

- Verifier doesn't know the subject
- Subject doesn't know the verifier
- All identifying information stripped
- **Recommended for production**

## Context Isolation

### Why Isolation Matters

"Context pollution" occurs when:
- Prior conversation context influences verification
- Shared state between verifications affects results
- Cached responses are reused incorrectly

### Isolation Levels

```python
from avir import IsolationLevel

# Process isolation (default)
verifier = CrossVerifier(
    providers=providers,
    isolation_level=IsolationLevel.PROCESS
)

# Container isolation (enhanced)
verifier = CrossVerifier(
    providers=providers,
    isolation_level=IsolationLevel.CONTAINER
)

# TEE isolation (maximum)
verifier = CrossVerifier(
    providers=providers,
    isolation_level=IsolationLevel.TEE
)
```

| Level | Isolation | Use Case |
|-------|-----------|----------|
| `NONE` | Shared context | Development only |
| `PROCESS` | Separate process | Default, good for most cases |
| `CONTAINER` | Docker container | Enhanced isolation |
| `TEE` | Trusted Execution Environment | Maximum security |

## Implementation

### Basic Cross-Verification

```python
from avir import Specification, cross_verify, BlindingMode
from avir.providers import ClaudeProvider, OpenAIProvider, GeminiProvider

# Load specification
spec = Specification.from_file("spec.json")

# Create providers (minimum 2 required)
providers = [
    ClaudeProvider(),
    OpenAIProvider(),
    GeminiProvider(),
]

# Run verification
matrix = await cross_verify(
    spec=spec,
    providers=providers,
    blinding=BlindingMode.DOUBLE_BLIND,
    iterations=5,
)

# Check results
print(f"Consensus: {matrix.consensus_verdict}")
print(f"Agreement: {matrix.agreement_score:.1%}")
```

### Advanced Configuration

```python
from avir import CrossVerifier, BlindingMode, IsolationLevel

# Create verifier with custom settings
verifier = CrossVerifier(
    providers=providers,
    blinding_mode=BlindingMode.DOUBLE_BLIND,
    isolation_level=IsolationLevel.CONTAINER,
    iterations=10,
)

# Run verification with requirements
matrix = await verifier.verify(
    spec,
    require_minimum_providers=3,  # Require at least 3 providers
)
```

### Analyzing Results

```python
# Get summary
print(matrix.summary())

# Check individual cells
for cell in matrix.cross_verifications:
    print(f"{cell.verifier_id} → {cell.subject_id}: {cell.verdict}")

# Check agreement
if matrix.agreement_score < 0.8:
    print("Warning: Low agreement between providers")
    print("This may indicate provider-specific biases")

# Export results
results = matrix.to_dict()
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Consensus Calculation

### Rules

1. **Unanimous VERIFIED** = VERIFIED
2. **Strong majority VERIFIED** (≥2/3) = VERIFIED
3. **Majority FAILED** (>50%) = FAILED
4. **Mixed results** with majority passing = PARTIAL
5. **No clear majority** = INCONCLUSIVE

### Agreement Score

The agreement score measures inter-rater reliability:

```
agreement_score = count(most_common_verdict) / total_verdicts
```

- **1.0**: All providers agree (unanimous)
- **0.67**: 2/3 agree (strong consensus)
- **0.5**: Split decision (weak consensus)
- **<0.5**: No consensus (unreliable)

## Best Practices

### Provider Selection

1. **Use providers from different organizations**
   - Avoids shared training biases
   - Different architectures = different perspectives

2. **Minimum 2 providers, recommend 3+**
   - More providers = more robust consensus
   - Diminishing returns above 5 providers

3. **Include diverse model types**
   - Mix general and specialized models
   - Include different model sizes

### Specification Design

1. **Clear, unambiguous benchmarks**
   - Minimize interpretation differences between providers

2. **Appropriate tolerances**
   - Too tight = false failures
   - Too loose = meaningless verification

3. **Sufficient iterations**
   - Minimum 5 for statistical validity
   - 10+ for production verification

### Handling Disagreements

When providers disagree (low agreement score):

1. **Investigate the cause**
   - Review individual results
   - Check for systematic differences

2. **Consider specification issues**
   - Ambiguous benchmarks
   - Provider-specific interpretations

3. **Add more providers**
   - More data points for consensus
   - May clarify the disagreement

4. **Manual review**
   - Human expert analysis
   - Root cause investigation

## Security Considerations

### Preventing Collusion

- Providers don't know which other providers are involved
- Results submitted independently
- No inter-provider communication during verification

### Preventing Gaming

- Double-blind prevents optimization for specific verifiers
- Multiple verifications from different providers
- Context isolation prevents cached responses

### Audit Trail

Every verification produces:
- Cryptographic hash chain
- Blinded context records
- Provider attribution (after verification)
- Timestamp evidence

## Example: Production Verification

```python
import asyncio
from avir import (
    Specification,
    CrossVerifier,
    BlindingMode,
    IsolationLevel,
)
from avir.providers import ClaudeProvider, OpenAIProvider, GeminiProvider

async def production_verification(spec_path: str) -> dict:
    """Run production-grade cross-verification."""

    # Load specification
    spec = Specification.from_file(spec_path)

    # Create diverse provider set
    providers = [
        ClaudeProvider(model="claude-sonnet-4-20250514"),
        OpenAIProvider(model="gpt-4-turbo"),
        GeminiProvider(model="gemini-pro"),
    ]

    # Configure verifier for production
    verifier = CrossVerifier(
        providers=providers,
        blinding_mode=BlindingMode.DOUBLE_BLIND,
        isolation_level=IsolationLevel.CONTAINER,
        iterations=10,
    )

    # Run verification
    matrix = await verifier.verify(
        spec,
        require_minimum_providers=3,
    )

    # Validate results
    if matrix.agreement_score < 0.67:
        raise ValueError(f"Insufficient agreement: {matrix.agreement_score:.1%}")

    return {
        "verdict": matrix.consensus_verdict.value,
        "pass_rate": matrix.consensus_pass_rate,
        "agreement": matrix.agreement_score,
        "providers": matrix.providers,
        "matrix": matrix.to_dict(),
    }

if __name__ == "__main__":
    result = asyncio.run(production_verification("spec.json"))
    print(f"Verdict: {result['verdict']}")
    print(f"Agreement: {result['agreement']:.1%}")
```

## Troubleshooting

### Low Agreement Score

**Symptoms**: Agreement score < 0.67

**Possible causes**:
1. Ambiguous benchmark specifications
2. Provider-specific implementation differences
3. Insufficient iterations (high variance)
4. Incompatible provider capabilities

**Solutions**:
1. Review and clarify benchmark specs
2. Increase iterations for statistical stability
3. Add more providers to break ties
4. Check provider capability compatibility

### Inconsistent Results

**Symptoms**: Same verification produces different results

**Possible causes**:
1. Non-deterministic benchmarks
2. Context leakage
3. Provider model updates

**Solutions**:
1. Ensure benchmarks are deterministic
2. Use container isolation
3. Pin provider model versions

### Performance Issues

**Symptoms**: Verification takes too long

**Possible causes**:
1. Too many iterations
2. Too many providers
3. Complex benchmarks

**Solutions**:
1. Optimize iteration count
2. Run providers in parallel
3. Simplify benchmark definitions
