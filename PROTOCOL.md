# AVIR Protocol Specification v1.0.0

**AI-Verified Independent Replication Protocol**

## Abstract

This document specifies the AVIR (AI-Verified Independent Replication) Protocol, a standardized method for AI systems to independently verify the capabilities of other AI systems through reproducible benchmarks and cryptographic attestation.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Terminology](#2-terminology)
3. [Protocol Architecture](#3-protocol-architecture)
4. [Specification Document](#4-specification-document)
5. [Benchmark System](#5-benchmark-system)
6. [Verification Process](#6-verification-process)
7. [Attestation Chain](#7-attestation-chain)
8. [Isolation Requirements](#8-isolation-requirements)
9. [Provider Interface](#9-provider-interface)
10. [Verdict Determination](#10-verdict-determination)
11. [Security Considerations](#11-security-considerations)
12. [Extensibility](#12-extensibility)

---

## 1. Introduction

### 1.1 Purpose

The AVIR Protocol enables trustworthy verification of AI system capabilities without requiring access to source code or internal implementations. This is achieved through:

- Standardized benchmark specifications
- Container-isolated verification environments
- Multi-provider AI verification
- Cryptographic attestation chains

### 1.2 Scope

This protocol covers:
- Specification document format and validation
- Benchmark definition and execution methodology
- Verification environment requirements
- Attestation generation and verification
- Verdict criteria and reporting

### 1.3 Design Principles

1. **Reproducibility**: Same inputs must produce statistically equivalent outputs
2. **Isolation**: Verification must occur in controlled environments
3. **Transparency**: All specifications and methodologies must be public
4. **Provider Agnostic**: Support for multiple AI verification providers
5. **Tamper Evidence**: Cryptographic proof of verification integrity

---

## 2. Terminology

| Term | Definition |
|------|------------|
| **SUT** | System Under Test - the AI system being verified |
| **Verifier** | Independent AI system performing verification |
| **Specification** | Document describing SUT capabilities and benchmarks |
| **Benchmark** | Standardized test with defined methodology |
| **Attestation** | Cryptographic proof of verification results |
| **Tolerance** | Acceptable deviation from target performance |
| **Run** | Single execution of a benchmark |
| **Suite** | Collection of related benchmarks |

---

## 3. Protocol Architecture

### 3.1 Components

```
┌─────────────────────────────────────────────────────────────┐
│                     AVIR PROTOCOL                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ SPEC LAYER  │    │  EXEC LAYER │    │ ATTEST LAYER│     │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤     │
│  │ - Schema    │    │ - Container │    │ - Hash Chain│     │
│  │ - Validator │    │ - Sandbox   │    │ - Signatures│     │
│  │ - Parser    │    │ - Providers │    │ - Timestamp │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                    ┌───────▼───────┐                        │
│                    │  COORDINATOR  │                        │
│                    └───────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
INPUT                    PROCESS                    OUTPUT
─────────────────────────────────────────────────────────────
Specification   ──►  Validation    ──►  Valid Spec
                          │
                          ▼
Environment     ──►  Isolation     ──►  Sandbox Ready
                          │
                          ▼
Benchmarks      ──►  Execution     ──►  Raw Results
                          │
                          ▼
Results         ──►  Aggregation   ──►  Statistics
                          │
                          ▼
Statistics      ──►  Verdict       ──►  Pass/Fail
                          │
                          ▼
All Data        ──►  Attestation   ──►  Signed Proof
```

---

## 4. Specification Document

### 4.1 Format

Specifications MUST be in YAML or JSON format and MUST conform to the AVIR specification schema.

### 4.2 Required Fields

```yaml
# avir-spec.yaml
avir_version: "1.0.0"

system:
  name: string          # Human-readable system name
  version: string       # Semantic version (X.Y.Z)
  description: string   # Brief system description
  repository: url       # Optional: public repository URL

capabilities:
  - name: string        # Capability identifier
    description: string # What this capability provides
    category: string    # memory|reasoning|coordination|custom
    benchmarks:         # Associated benchmarks
      - benchmark_id

benchmarks:
  - id: string          # Unique benchmark identifier
    description: string # What is being tested
    methodology: string # How the test is conducted
    target: number      # Expected performance value
    unit: string        # Unit of measurement
    tolerance: number   # Acceptable deviation (0.0-1.0)
    lower_is_better: boolean  # Optional, default false
    runs: integer       # Minimum runs (default: 5)
    requirements:       # Environmental requirements
      - requirement

metadata:
  author: string
  created: date
  updated: date
  tags: [string]
```

### 4.3 Schema Validation

All specifications MUST pass JSON Schema validation before verification:

```json
{
  "$schema": "https://avir-protocol.dev/schemas/v1/specification.json",
  "type": "object",
  "required": ["avir_version", "system", "capabilities", "benchmarks"]
}
```

### 4.4 Example Specification

```yaml
avir_version: "1.0.0"

system:
  name: "Agentic System"
  version: "2.0.0"
  description: "24/7 autonomous AI with persistent memory"
  repository: "https://github.com/marc-shade/agentic-system-oss"

capabilities:
  - name: "persistent_memory"
    description: "4-tier memory with automatic curation"
    category: "memory"
    benchmarks:
      - memory_entity_creation
      - semantic_search
      - memory_promotion

benchmarks:
  - id: memory_entity_creation
    description: "Create memory entities with versioning"
    methodology: "Create 1000 entities with varying sizes (100B-10KB), measure throughput"
    target: 435
    unit: "ops/s"
    tolerance: 0.20
    runs: 5
    requirements:
      - "Qdrant vector database"
      - "enhanced-memory-mcp server"
```

---

## 5. Benchmark System

### 5.1 Benchmark Categories

| Category | Description | Example Benchmarks |
|----------|-------------|-------------------|
| **memory** | Storage and retrieval operations | entity_creation, search, promotion |
| **reasoning** | Inference and decision making | task_decomposition, planning |
| **coordination** | Multi-agent operations | baton_handoff, message_passing |
| **performance** | System-level metrics | latency, throughput, concurrency |
| **custom** | Domain-specific tests | User-defined benchmarks |

### 5.2 Benchmark Definition

```yaml
benchmark:
  id: unique_identifier

  # What is being tested
  description: "Human-readable description"

  # Detailed test procedure
  methodology: |
    Step-by-step methodology that ANY verifier can execute:
    1. Initialize test environment
    2. Perform N iterations of operation X
    3. Measure metric Y
    4. Calculate aggregate statistics

  # Performance target
  target: 100
  unit: "ops/s"

  # Acceptable variance from target
  tolerance: 0.20  # ±20%

  # Direction of improvement
  lower_is_better: false

  # Statistical requirements
  runs: 5              # Minimum independent runs
  warmup_runs: 2       # Runs to discard for warmup
  outlier_policy: "iqr" # iqr|zscore|none

  # Environmental prerequisites
  requirements:
    - "Service X running on port Y"
    - "Minimum 4GB RAM available"

  # Setup commands (optional)
  setup:
    - "docker run -d service:latest"

  # Teardown commands (optional)
  teardown:
    - "docker stop service"
```

### 5.3 Statistical Requirements

All benchmarks MUST:

1. **Minimum Runs**: Execute at least 5 independent runs
2. **Warmup**: Discard first 1-2 runs if warmup_runs specified
3. **Outlier Handling**: Apply IQR method by default
4. **Reporting**: Report mean, std dev, min, max, p95

### 5.4 Tolerance Calculation

```
PASS if:
  - For lower_is_better=false: result >= target * (1 - tolerance)
  - For lower_is_better=true:  result <= target * (1 + tolerance)
```

Example:
- Target: 435 ops/s, Tolerance: 0.20
- Pass threshold: 435 * 0.80 = 348 ops/s
- Result 400 ops/s → PASS
- Result 300 ops/s → FAIL

---

## 6. Verification Process

### 6.1 Process Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   VERIFICATION PROCESS                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. VALIDATE      2. PROVISION     3. EXECUTE     4. ATTEST │
│  ─────────────    ────────────     ─────────      ───────── │
│                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐    ┌─────────┐│
│  │  Parse  │ ──► │ Create  │ ──► │  Run    │ ──►│ Generate││
│  │  Spec   │     │ Sandbox │     │ Tests   │    │  Proof  ││
│  └─────────┘     └─────────┘     └─────────┘    └─────────┘│
│       │               │               │              │      │
│       ▼               ▼               ▼              ▼      │
│  [Valid Spec]   [Container]    [Results]     [Attestation] │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Phase 1: Validation

1. Parse specification document
2. Validate against AVIR schema
3. Check benchmark references
4. Verify requirement declarations
5. Generate specification hash: `SPEC_HASH = SHA256(canonical_spec)`

### 6.3 Phase 2: Provisioning

1. Create isolated container environment
2. Install required dependencies
3. Copy specification (NO source code)
4. Configure AI provider
5. Generate environment hash: `ENV_HASH = SHA256(container_state)`

### 6.4 Phase 3: Execution

For each benchmark in the suite:

```python
for benchmark in suite.benchmarks:
    results = []

    # Warmup runs (discarded)
    for _ in range(benchmark.warmup_runs):
        execute_benchmark(benchmark)

    # Measurement runs
    for run in range(benchmark.runs):
        result = execute_benchmark(benchmark)
        results.append(result)

    # Statistical analysis
    filtered = remove_outliers(results, benchmark.outlier_policy)
    statistics = calculate_statistics(filtered)

    # Verdict for this benchmark
    benchmark.verdict = determine_verdict(statistics, benchmark)
```

### 6.5 Phase 4: Attestation

1. Collect all results
2. Calculate results hash: `RESULTS_HASH = SHA256(canonical_results)`
3. Generate attestation chain
4. Sign attestation (optional)
5. Output final verdict

---

## 7. Attestation Chain

### 7.1 Chain Structure

The attestation chain provides cryptographic proof of verification integrity:

```
ATTESTATION_CHAIN = {
    spec_hash:     SHA256(specification),
    env_hash:      SHA256(environment_state),
    results_hash:  SHA256(verification_results),
    timestamp:     ISO8601_UTC,
    chain_hash:    SHA256(spec_hash || env_hash || results_hash || timestamp)
}
```

### 7.2 Full Attestation Document

```json
{
  "avir_protocol_version": "1.0.0",
  "verification_level": "L3",

  "system": {
    "name": "Agentic System",
    "version": "2.0.0",
    "spec_source": "https://github.com/.../avir-spec.yaml"
  },

  "verifier": {
    "provider": "claude",
    "model": "claude-sonnet-4-20250514",
    "instance_id": "uuid"
  },

  "environment": {
    "container": "avir-verification:1.0.0",
    "os": "linux",
    "arch": "amd64",
    "resources": {
      "cpu_cores": 4,
      "memory_gb": 16
    }
  },

  "execution": {
    "started_at": "2025-01-15T10:00:00Z",
    "completed_at": "2025-01-15T10:05:23Z",
    "duration_seconds": 323
  },

  "results": {
    "benchmarks": [
      {
        "id": "memory_entity_creation",
        "runs": 5,
        "statistics": {
          "mean": 421.3,
          "std_dev": 15.2,
          "min": 398.1,
          "max": 445.7,
          "p95": 443.2
        },
        "target": 435,
        "tolerance": 0.20,
        "verdict": "PASS"
      }
    ],
    "summary": {
      "total": 5,
      "passed": 5,
      "failed": 0,
      "pass_rate": 1.0
    }
  },

  "attestation_chain": {
    "spec_hash": "a1b2c3d4e5f6...",
    "env_hash": "f7g8h9i0j1k2...",
    "results_hash": "l3m4n5o6p7q8...",
    "timestamp": "2025-01-15T10:05:23Z",
    "chain_hash": "r9s0t1u2v3w4..."
  },

  "signature": {
    "algorithm": "Ed25519",
    "public_key": "base64_encoded_pubkey",
    "signature": "base64_encoded_signature"
  },

  "verdict": "VERIFIED",
  "verdict_details": "All 5 benchmarks passed within tolerance"
}
```

### 7.3 Hash Calculation

All hashes use SHA-256 with canonical JSON encoding:

```python
import hashlib
import json

def canonical_hash(data: dict) -> str:
    """Generate deterministic hash of data."""
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()
```

### 7.4 Signature (Optional)

For L3+ verification levels, attestations SHOULD be signed:

```python
from cryptography.hazmat.primitives.asymmetric import ed25519

def sign_attestation(attestation: dict, private_key: bytes) -> str:
    """Sign attestation chain hash."""
    chain_hash = attestation['attestation_chain']['chain_hash']
    signing_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
    signature = signing_key.sign(chain_hash.encode())
    return base64.b64encode(signature).decode()
```

---

## 8. Isolation Requirements

### 8.1 Container Specification

```dockerfile
# AVIR Verification Container
FROM python:3.11-slim

# Minimal base - no system source code
RUN apt-get update && apt-get install -y \
    curl git && rm -rf /var/lib/apt/lists/*

WORKDIR /verification

# Install AVIR runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY specifications
COPY specs/ /verification/specs/

# Verification entry point
COPY verify.py .
ENTRYPOINT ["python3", "verify.py"]
```

### 8.2 Resource Limits

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| Memory | 4 GB | 8 GB |
| Disk | 10 GB | 20 GB |
| Network | Required | Required |

### 8.3 Security Requirements

1. **No Source Access**: Container must NOT contain SUT source code
2. **Network Isolation**: Only outbound to required APIs
3. **Read-Only Specs**: Specifications mounted read-only
4. **No Persistent State**: Fresh container per verification

---

## 9. Provider Interface

### 9.1 Provider Contract

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class AVIRProvider(ABC):
    """Interface for AI verification providers."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize provider with configuration."""
        pass

    @abstractmethod
    async def execute_benchmark(
        self,
        benchmark: 'Benchmark',
        context: Dict[str, Any]
    ) -> 'BenchmarkResult':
        """Execute a single benchmark run."""
        pass

    @abstractmethod
    async def analyze_results(
        self,
        results: List['BenchmarkResult']
    ) -> 'Analysis':
        """AI analysis of verification results."""
        pass

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        pass
```

### 9.2 Supported Providers

| Provider | ID | API Endpoint | Auth |
|----------|-----|--------------|------|
| Anthropic Claude | `claude` | api.anthropic.com | API Key |
| OpenAI GPT/Codex | `openai` | api.openai.com | API Key |
| Google Gemini | `gemini` | generativelanguage.googleapis.com | API Key |
| Local (Ollama) | `ollama` | localhost:11434 | None |

### 9.3 Provider Configuration

```yaml
# providers.yaml
providers:
  claude:
    api_key: ${ANTHROPIC_API_KEY}
    model: "claude-sonnet-4-20250514"
    max_tokens: 4096

  openai:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-4-turbo"

  gemini:
    api_key: ${GOOGLE_API_KEY}
    model: "gemini-pro"

  ollama:
    base_url: "http://localhost:11434"
    model: "llama3.3"
```

---

## 10. Verdict Determination

### 10.1 Benchmark-Level Verdict

Each benchmark receives individual verdict:

```python
def determine_benchmark_verdict(result: Statistics, benchmark: Benchmark) -> str:
    if benchmark.lower_is_better:
        threshold = benchmark.target * (1 + benchmark.tolerance)
        passed = result.mean <= threshold
    else:
        threshold = benchmark.target * (1 - benchmark.tolerance)
        passed = result.mean >= threshold

    return "PASS" if passed else "FAIL"
```

### 10.2 Overall Verdict

| Verdict | Criteria |
|---------|----------|
| **VERIFIED** | 100% benchmarks pass |
| **PARTIAL** | 60-99% benchmarks pass |
| **FAILED** | <60% benchmarks pass |
| **INVALID** | Specification or execution errors |
| **INCONCLUSIVE** | Insufficient data or high variance |

### 10.3 Verification Levels

| Level | Provider Requirements | Statistical Requirements |
|-------|----------------------|-------------------------|
| L1 Basic | 1 provider | 1 run |
| L2 Standard | 1 provider | 5 runs, outlier removal |
| L3 Enhanced | 2+ providers | 5 runs each, cross-validation |
| L4 Comprehensive | 3+ providers | 10 runs each, TEE attestation |

---

## 11. Security Considerations

### 11.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Specification tampering | Hash chain verification |
| Result manipulation | Cryptographic signatures |
| Environment compromise | Container isolation |
| Provider collusion | Multi-provider verification |
| Replay attacks | Timestamp inclusion |

### 11.2 Hash Chain Verification

```python
def verify_attestation(attestation: dict) -> bool:
    """Verify attestation chain integrity."""
    chain = attestation['attestation_chain']

    # Reconstruct chain hash
    data = f"{chain['spec_hash']}{chain['env_hash']}{chain['results_hash']}{chain['timestamp']}"
    expected_hash = hashlib.sha256(data.encode()).hexdigest()

    return chain['chain_hash'] == expected_hash
```

### 11.3 Signature Verification

```python
from cryptography.hazmat.primitives.asymmetric import ed25519

def verify_signature(attestation: dict) -> bool:
    """Verify attestation signature."""
    sig_data = attestation['signature']
    chain_hash = attestation['attestation_chain']['chain_hash']

    public_key = ed25519.Ed25519PublicKey.from_public_bytes(
        base64.b64decode(sig_data['public_key'])
    )
    signature = base64.b64decode(sig_data['signature'])

    try:
        public_key.verify(signature, chain_hash.encode())
        return True
    except Exception:
        return False
```

---

## 12. Extensibility

### 12.1 Custom Benchmark Types

Register custom benchmark types:

```python
from avir import BenchmarkRegistry

@BenchmarkRegistry.register("custom_ml_inference")
class MLInferenceBenchmark:
    async def execute(self, config: dict) -> float:
        # Custom implementation
        return inference_time_ms
```

### 12.2 Custom Providers

Add new AI providers:

```python
from avir.providers import AVIRProvider

class CustomProvider(AVIRProvider):
    @property
    def provider_id(self) -> str:
        return "custom"

    async def execute_benchmark(self, benchmark, context):
        # Custom implementation
        pass
```

### 12.3 Plugin Architecture

```
avir/
├── plugins/
│   ├── benchmarks/
│   │   └── custom_benchmark.py
│   └── providers/
│       └── custom_provider.py
```

---

## Appendix A: JSON Schemas

### A.1 Specification Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://avir-protocol.dev/schemas/v1/specification.json",
  "type": "object",
  "required": ["avir_version", "system", "capabilities", "benchmarks"],
  "properties": {
    "avir_version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$"
    },
    "system": {
      "type": "object",
      "required": ["name", "version"],
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "description": {"type": "string"},
        "repository": {"type": "string", "format": "uri"}
      }
    },
    "capabilities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "description", "category", "benchmarks"],
        "properties": {
          "name": {"type": "string"},
          "description": {"type": "string"},
          "category": {"enum": ["memory", "reasoning", "coordination", "performance", "custom"]},
          "benchmarks": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "benchmarks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "description", "methodology", "target", "unit", "tolerance"],
        "properties": {
          "id": {"type": "string"},
          "description": {"type": "string"},
          "methodology": {"type": "string"},
          "target": {"type": "number"},
          "unit": {"type": "string"},
          "tolerance": {"type": "number", "minimum": 0, "maximum": 1},
          "lower_is_better": {"type": "boolean", "default": false},
          "runs": {"type": "integer", "minimum": 1, "default": 5},
          "requirements": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }
}
```

### A.2 Attestation Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://avir-protocol.dev/schemas/v1/attestation.json",
  "type": "object",
  "required": ["avir_protocol_version", "system", "verifier", "results", "attestation_chain", "verdict"],
  "properties": {
    "avir_protocol_version": {"type": "string"},
    "verification_level": {"enum": ["L1", "L2", "L3", "L4"]},
    "attestation_chain": {
      "type": "object",
      "required": ["spec_hash", "env_hash", "results_hash", "timestamp", "chain_hash"],
      "properties": {
        "spec_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "env_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "results_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "chain_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"}
      }
    },
    "verdict": {"enum": ["VERIFIED", "PARTIAL", "FAILED", "INVALID", "INCONCLUSIVE"]}
  }
}
```

---

## Appendix B: Reference Implementation

See the `avir/` directory for the reference Python implementation of this protocol.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-15 | Initial release |

---

**AVIR Protocol** - Enabling trust through independent verification.
