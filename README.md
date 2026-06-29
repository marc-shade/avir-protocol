# AVIR Protocol

<img align="right" width="300" height="300" alt="AVIR-protocol" src="https://github.com/user-attachments/assets/b0733676-f36e-4f6d-b2e1-7b5bdb8b6528" />

[![Protocol Version](https://img.shields.io/badge/protocol-v1.0.0-blue)](PROTOCOL.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FAIR Principles](https://img.shields.io/badge/FAIR-aligned-green)](https://www.go-fair.org/fair-principles/)

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

# real-model providers via SDK (need API keys)
pip install avir-protocol[all-providers]

# cryptographic attestation + identity (Ed25519)
pip install avir-protocol[crypto]

# host a verification-network coordinator (FastAPI daemon + cert gen)
pip install avir-protocol[network,crypto]

# expose the network to an agent over MCP
pip install avir-protocol[mcp]
```

The headless-CLI providers (`claude`/`codex`/`gemini`) and the verifier/requester
network clients are stdlib-only and need no extra install beyond the CLIs
themselves.

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

### Real-Model Verification (no SDK, v1.1+)

For verification that actually invokes models, use the CLI providers (headless
`claude` / `codex` / `gemini`, each a separate OS process) and/or HTTP/cluster
providers (a model on another machine). Rubric benchmarks have the verifier
model judge a blinded answer -- a real judgment that can disagree.

```python
from avir import Specification, BlindingMode, IsolationLevel
from avir.cross_verification import CrossVerifier
from avir.providers import ClaudeCLIProvider, GeminiCLIProvider, fedora_ollama, ailab_model

spec = Specification.from_file("examples/cluster_demonstrator_spec.json")  # rubric
cv = CrossVerifier(
    providers=[
        ClaudeCLIProvider(),                                              # Anthropic (process)
        GeminiCLIProvider(),                                              # Google (process)
        fedora_ollama("deepseek-v4-pro:cloud", organization="deepseek"), # node
        ailab_model("ornith-9b", organization="deepreinforce"),          # node
    ],
    blinding_mode=BlindingMode.DOUBLE_BLIND,
    isolation_level=IsolationLevel.DISTRIBUTED,
    iterations=1,
)
matrix = await cv.verify(spec)
print(matrix.consensus_verdict, matrix.agreement_score)
```

Each cell records a `verifier_locus` (e.g. `node:fedora.local/...`) so the
attested isolation is auditable. See `examples/cluster_cross_verify.py` for a
full run that emits a signed attestation. Numeric benchmarks remain deterministic
arithmetic; the SDK providers (`ClaudeProvider`, etc.) require API keys and are
not exercised in CI -- the CLI/cluster providers are the tested real-model path.

### Verification Network (external verifiers, v1.2+)

The network lets **independent operators** contribute their own compute and AI
providers as external verification sources, so the graders are people you do not
control. A coordinator daemon queues blinded jobs; verifier nodes pull work,
judge with their own models, and submit **Ed25519-signed** verdicts; the
requester re-verifies every signature locally. A persistent participant registry
lets nodes find and verify each other.

```bash
# host a coordinator (job board daemon)
avir-coordinator --host 127.0.0.1 --port 8770

# anyone contributes a verifier node with their own model
avir-verifier --coordinator http://HOST:8770 --provider claude-cli --org alice
avir-verifier --coordinator http://HOST:8770 --provider http \
              --endpoint http://localhost:11434 --model llama3 --org bob

# request external verification (re-verifies every signature)
avir-verify-remote --coordinator http://HOST:8770 --spec spec.json \
                   --responses responses.json --min-verifiers 3 --require-distinct-orgs

# or drive it from an agent over MCP
AVIR_COORDINATOR=http://HOST:8770 avir-mcp
```

Trust model: the coordinator is untrusted for integrity (it cannot forge signed
verdicts); consensus counts distinct verifier keys (optionally distinct orgs)
and excludes the requester's own key. This is operator-level decentralization,
not formal independent-lab replication; Sybil/collusion is only partially
mitigated. Full detail and honest limits: [docs/NETWORK.md](docs/NETWORK.md).

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

| Level | Isolation | Status |
|-------|-----------|--------|
| **NONE** | Shared context | Not recommended |
| **PROCESS** | Separate OS process (CLI providers) | Implemented |
| **DISTRIBUTED** | Separate physical node (cluster/network) | Implemented |
| **CONTAINER** | Docker isolation | Reserved (not yet enforced) |
| **TEE** | Trusted Execution Environment | Reserved (not yet enforced) |

Each cell records a `verifier_locus`, so the attested isolation is auditable.

## Verification Levels

| Level | Requirements | Cross-Verification |
|-------|--------------|-------------------|
| **L1** Basic | 1 provider, 1 run | No |
| **L2** Standard | 1 provider, 5 runs, container | No |
| **L3** Enhanced | 2+ providers, 5 runs each | **Yes, single-blind** |
| **L4** Comprehensive | 3+ providers, 10 runs, TEE | **Yes, double-blind** |

## Specification Format

A specification has `avir_version`, `system`, `capabilities`, and `benchmarks`.
Benchmarks are either `numeric` (measured value vs `target` within `tolerance`,
verified by deterministic arithmetic) or `rubric` (a model answers `prompt` and
verifier models from other providers judge the blinded answer against `rubric`).

```json
{
  "avir_version": "1.0.0",
  "system": { "name": "my-ai-system", "version": "1.0.0" },
  "capabilities": [
    {
      "name": "throughput",
      "description": "Operation throughput under load",
      "category": "performance",
      "benchmarks": ["throughput"]
    },
    {
      "name": "reasoning",
      "description": "Trap-aware arithmetic",
      "category": "reasoning",
      "benchmarks": ["bat_and_ball"]
    }
  ],
  "benchmarks": [
    {
      "id": "throughput",
      "kind": "numeric",
      "description": "Operation Throughput",
      "methodology": "measured by the system under test",
      "target": 100,
      "unit": "ops/s",
      "tolerance": 0.2,
      "runs": 5
    },
    {
      "id": "bat_and_ball",
      "kind": "rubric",
      "description": "Cognitive-reflection arithmetic",
      "prompt": "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball, in cents?",
      "rubric": "PASS iff the final answer is 5 cents; 10 cents or anything else is FAIL.",
      "expected": "5 cents (bat $1.05, ball $0.05)."
    }
  ]
}
```

See [Real-Model Verification](docs/REAL_MODEL_VERIFICATION.md) for rubric
benchmarks and running across the CLI/cluster providers.

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

Two families of provider, both implementing the same interface
(`generate_response` + `judge_output`):

| Provider | Class | Transport | Notes |
|----------|-------|-----------|-------|
| Claude CLI | `ClaudeCLIProvider` | `claude --print` subprocess | Anthropic; no SDK/key, real PROCESS isolation |
| Codex CLI | `CodexCLIProvider` | `codex exec` subprocess | OpenAI; no SDK/key |
| Gemini CLI | `GeminiCLIProvider` | `gemini -p` subprocess | Google; no SDK/key |
| Any OpenAI-compatible endpoint | `HTTPModelProvider` (`fedora_ollama`, `ailab_model`) | HTTP `/v1/chat/completions` | ollama / vLLM / cluster GPU nodes; real DISTRIBUTED isolation |
| Claude / OpenAI / Gemini SDK | `ClaudeProvider` / `OpenAIProvider` / `GeminiProvider` | vendor SDK | Require API keys; not exercised in CI |

The CLI and HTTP providers are the tested real-model path and need no API keys
(auth lives in the CLI or the endpoint). The SDK providers abstain on numeric
benchmarks (an LLM cannot measure throughput) and judge rubric benchmarks when a
key is present.

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
│   ├── attestation.py       # Cryptographic attestation (Ed25519 chains)
│   ├── cross_verification.py # Double-blind cross-provider matrix
│   ├── verdict.py           # Verdict determination
│   ├── specification.py     # Spec parsing (numeric + rubric benchmarks)
│   ├── cli.py               # Command-line interface
│   ├── mcp_server.py        # MCP server (drive the network from an agent)
│   ├── providers/           # AI provider integrations
│   │   ├── base.py          # Abstract base (generate_response, judge_output)
│   │   ├── cli.py           # Real-model: Claude/Codex/Gemini CLIs + HTTP/cluster
│   │   ├── claude.py        # Anthropic Claude (SDK)
│   │   ├── openai.py        # OpenAI GPT/Codex (SDK)
│   │   └── gemini.py        # Google Gemini (SDK)
│   └── network/             # Distributed verification network
│       ├── keys.py          # Ed25519 identity (sign/verify/fingerprint)
│       ├── protocol.py      # Verdict signing + consensus aggregation
│       ├── registry.py      # Persistent participant/job/verdict store (sqlite)
│       ├── coordinator.py   # FastAPI job-board daemon (avir-coordinator)
│       ├── verifier.py      # Verifier node client (avir-verifier)
│       ├── requester.py     # Requester client (avir-verify-remote)
│       └── tls.py           # TLS certs + client SSL context (avir-gen-cert)
├── specs/                   # JSON schemas
├── examples/                # Usage examples (incl. cluster + network demos)
├── deploy/                  # launchd setup for persistent coordinator/verifier
├── tests/                   # Test suite (95 tests)
└── docs/                    # Documentation
```

CLI entry points: `avir`, `avir-coordinator`, `avir-verifier`,
`avir-verify-remote`, `avir-gen-cert`, `avir-mcp`.

## Documentation

- [Protocol Specification](PROTOCOL.md) - Full technical specification
- [Cross-Verification Guide](docs/CROSS_VERIFICATION.md) - Double-blind protocol details
- [Real-Model Verification](docs/REAL_MODEL_VERIFICATION.md) - Rubric benchmarks, CLI/cluster providers, auditable isolation
- [Verification Network](docs/NETWORK.md) - Distributed external verifiers, participant registry, signed verdicts, MCP server
- [Changelog](CHANGELOG.md) - Version history
- [Contributing](CONTRIBUTING.md) - Development and contribution guide

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
  url = {https://github.com/marc-shade/avir-protocol},
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
