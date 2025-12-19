# AVIR Protocol

<img align="right" width="300" height="300" alt="AVIR-protocol" src="https://github.com/user-attachments/assets/b0733676-f36e-4f6d-b2e1-7b5bdb8b6528" />

[![Protocol Version](https://img.shields.io/badge/protocol-v1.0.0-blue)](PROTOCOL.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FAIR Principles](https://img.shields.io/badge/FAIR-Compliant-green)](docs/FAIR_COMPLIANCE.md)

**AI-Verified Independent Replication** - A protocol for AI systems to independently verify each other through double-blind cross-provider verification, eliminating the "AI grading its own test" problem.

**The Problem AVIR Solves**

> *An AI verifying its own benchmarks is like a student grading their own test.*

When a single AI provider claims certain capabilities, there's no independent verification. AVIR solves this by requiring **multiple different AI providers** to verify each other's outputs using a **double-blind protocol** where:

1. **No AI verifies its own work** - Claude verifies GPT's outputs, GPT verifies Gemini's outputs, etc.
2. **Verifiers are blinded** - They don't know which provider generated the output
3. **Context is isolated** - Each verification runs in a clean environment to prevent "context pollution"
4. **Consensus required** - Multiple independent verifications must agree

## Key Features

- **Cross-Provider Verification** - Multiple AI providers verify each other, never themselves
- **Double-Blind Protocol** - Eliminates bias and prevents providers from gaming the system
- **Context Isolation** - Verifiable isolation of intellectual functions
- **Consensus Matrix** - NxN verification matrix with statistical agreement scoring
- **Cryptographic Attestation** - Tamper-proof hash chains for audit trails
- **In-Context Data** - All verification data preserved for full transparency

## Quick Start

### Installation

```bash
pip install avir-protocol

# Install with all AI providers
pip install avir-protocol[all-providers]
```

### Cross-Provider Verification (Recommended)

```python
from avir import Specification, cross_verify, BlindingMode
from avir.providers import ClaudeProvider, OpenAIProvider, GeminiProvider

# Load verification specification
spec = Specification.from_file("spec.json")

# Configure multiple providers (minimum 2 required)
providers = [
    ClaudeProvider(),      # Claude (Anthropic)
    OpenAIProvider(),      # GPT (OpenAI)
    GeminiProvider(),      # Gemini (Google)
]

# Run double-blind cross-verification
matrix = await cross_verify(
    spec=spec,
    providers=providers,
    blinding=BlindingMode.DOUBLE_BLIND,
    iterations=5,
)

# Check results
print(f"Consensus Verdict: {matrix.consensus_verdict}")
print(f"Agreement Score: {matrix.agreement_score:.1%}")
print(f"Pass Rate: {matrix.consensus_pass_rate:.1%}")
```

### CLI Usage

```bash
# Run cross-provider verification
avir verify spec.json --provider claude --provider openai --provider gemini

# Generate signed attestation
avir verify spec.json --attestation results.json --sign --key-file private.pem

# Validate a specification
avir validate spec.json

# Check existing attestation
avir check attestation.json --public-key public.pem
```

## How Cross-Verification Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROSS-VERIFICATION MATRIX                    │
├─────────────────┬───────────────────────────────────────────────┤
│                 │           SUBJECT (Being Verified)            │
│                 ├─────────────┬─────────────┬─────────────┬─────┤
│                 │   Claude    │   OpenAI    │   Gemini    │     │
├─────────────────┼─────────────┼─────────────┼─────────────┼─────┤
│ V  │   Claude   │  (excluded) │   ✓ PASS    │   ✓ PASS    │     │
│ E  ├────────────┼─────────────┼─────────────┼─────────────┤     │
│ R  │   OpenAI   │   ✓ PASS    │  (excluded) │   ✓ PASS    │     │
│ I  ├────────────┼─────────────┼─────────────┼─────────────┤     │
│ F  │   Gemini   │   ✓ PASS    │   ~ PARTIAL │  (excluded) │     │
│ I  ├────────────┼─────────────┼─────────────┼─────────────┤     │
│ E  │            │             │             │             │     │
│ R  │ CONSENSUS  │  VERIFIED   │   PARTIAL   │  VERIFIED   │     │
└────┴────────────┴─────────────┴─────────────┴─────────────┴─────┘

Self-verification (diagonal) is EXCLUDED from consensus calculation
Cross-verification cells determine the final verdict
```

### Blinding Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **NONE** | No blinding (development only) | Debugging |
| **SINGLE_BLIND** | Verifier doesn't know the subject | Basic verification |
| **DOUBLE_BLIND** | Neither side knows the other | Production (recommended) |

### Isolation Levels

| Level | Isolation | Description |
|-------|-----------|-------------|
| **NONE** | Shared context | Not recommended |
| **PROCESS** | Separate process | Default isolation |
| **CONTAINER** | Docker isolation | Enhanced isolation |
| **TEE** | Trusted Execution Environment | Maximum isolation |

## Verification Levels

| Level | Requirements | Cross-Verification |
|-------|--------------|-------------------|
| **L1** Basic | 1 provider, 1 run | No |
| **L2** Standard | 1 provider, 5 runs, container | No |
| **L3** Enhanced | 2+ providers, 5 runs each | **Yes, single-blind** |
| **L4** Comprehensive | 3+ providers, 10 runs, TEE | **Yes, double-blind** |

## Specification Format

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "My AI System Verification",
    "description": "AVIR specification for verifying AI capabilities"
  },
  "system": {
    "name": "my-ai-system",
    "version": "1.0.0"
  },
  "benchmarks": [
    {
      "id": "throughput",
      "name": "Operation Throughput",
      "target": 100,
      "unit": "ops/s",
      "tolerance": 0.2,
      "iterations": 5
    },
    {
      "id": "latency",
      "name": "Response Latency",
      "target": 50,
      "unit": "ms",
      "tolerance": 0.25,
      "lower_is_better": true
    }
  ],
  "cross_verification": {
    "enabled": true,
    "blinding_mode": "double_blind",
    "exclude_self_verification": true
  }
}
```

## Attestation Chain

Every verification produces a cryptographic attestation chain:

```json
{
  "version": "1.0.0",
  "timestamp": "2025-01-01T00:00:00Z",
  "specification_hash": "a1b2c3d4...",
  "verdict": "VERIFIED",
  "cross_verification": {
    "enabled": true,
    "blinding_mode": "double_blind",
    "providers": ["claude", "openai", "gemini"],
    "consensus_verdict": "VERIFIED",
    "agreement_score": 0.833
  },
  "chain": {
    "chain_hash": "SHA256(spec||env||results||timestamp)",
    "entries": [...]
  },
  "signature": {
    "algorithm": "Ed25519",
    "signature": "base64_encoded_signature"
  }
}
```

## Verdict Criteria

| Verdict | Criteria |
|---------|----------|
| **VERIFIED** | ≥80% benchmarks pass, consensus achieved |
| **PARTIAL** | 50-79% benchmarks pass |
| **FAILED** | <50% benchmarks pass |
| **INCONCLUSIVE** | Providers disagree (low agreement score) |
| **INVALID** | Specification errors or verification failures |

## Supported Providers

| Provider | Status | Best For |
|----------|--------|----------|
| **Claude** (Anthropic) | Stable | Complex reasoning, analysis |
| **GPT-4/Codex** (OpenAI) | Stable | Code-focused benchmarks |
| **Gemini** (Google) | Stable | Multimodal verification |
| **Ollama** (Local) | Beta | Privacy-preserving verification |

## Why Multiple Providers Matter

Using a single AI provider to verify AI capabilities has fundamental problems:

1. **Self-Interest**: A provider may unconsciously (or consciously) favor its own outputs
2. **Shared Biases**: Same training data can produce same blind spots
3. **Context Leakage**: Prior context can influence verification
4. **Gaming Risk**: Single-provider systems can be optimized to pass tests

AVIR's cross-provider approach creates a **trustless verification network** where:
- No single provider can unilaterally claim capability
- Biases from different training regimes cancel out
- Context isolation prevents "cheating"
- Consensus requirements ensure reliability

## Repository Structure

```
avir-protocol/
├── avir/                    # Core Python library
│   ├── __init__.py
│   ├── verifier.py          # Main verification engine
│   ├── benchmark.py         # Benchmark execution
│   ├── attestation.py       # Cryptographic attestation
│   ├── cross_verification.py # Double-blind cross-provider
│   ├── verdict.py           # Verdict determination
│   ├── specification.py     # Spec parsing/validation
│   ├── cli.py               # Command-line interface
│   └── providers/           # AI provider integrations
│       ├── base.py          # Abstract base class
│       ├── claude.py        # Anthropic Claude
│       ├── openai.py        # OpenAI GPT/Codex
│       └── gemini.py        # Google Gemini
├── specs/                   # JSON schemas
│   ├── specification.schema.json
│   └── attestation.schema.json
├── examples/                # Usage examples
│   ├── basic_spec.json
│   └── cross_verify_example.py
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## Documentation

- [Protocol Specification](PROTOCOL.md) - Full technical specification
- [Cross-Verification Guide](docs/CROSS_VERIFICATION.md) - Double-blind protocol details
- [Benchmark Design](docs/BENCHMARK_DESIGN.md) - How to create benchmarks
- [Attestation System](docs/ATTESTATION.md) - Cryptographic attestation
- [API Reference](docs/API.md) - Python API documentation

## Contributing

Contributions welcome! Key areas:

- **New Providers**: Add support for additional AI systems
- **Benchmark Types**: Expand verification capabilities
- **Isolation Methods**: Enhance context isolation
- **Documentation**: Improve guides and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{avir_protocol_2025,
  title = {AVIR Protocol: AI-Verified Independent Replication},
  author = {Shade, Marc},
  year = {2025},
  url = {https://github.com/marcshade/avir-protocol},
  note = {Cross-provider double-blind verification for AI systems}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Model Cards](https://arxiv.org/abs/1810.03993) - Transparent ML documentation
- [FAIR Principles](https://www.go-fair.org/fair-principles/) - Data accessibility standards
- [TEE/SGX Attestation](https://sgx101.gitbook.io/sgx101/sgx-bootstrap/attestation) - Remote attestation design
- [MLCommons](https://mlcommons.org/benchmarks/) - Benchmark standards

---

**Part of the Agentic System ecosystem**
https://github.com/marc-shade/agentic-system-oss
