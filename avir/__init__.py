"""
AVIR - AI-Verified Independent Replication Protocol

A protocol enabling AI systems to independently verify each other's
claims, benchmarks, and attestations through cryptographically-signed
verification chains.

Key Principles:
- Multi-provider verification: AI systems verify OTHER systems, never themselves
- Double-blind protocol: Eliminates bias and context pollution
- Context isolation: Each verification runs in clean, isolated environment
- Consensus matrix: Multiple independent verifications required for trust
- Cryptographic attestation: Tamper-evident verification chains

Why Cross-Provider Verification Matters:
  An AI system verifying its own benchmarks is like a student grading their own test.
  AVIR requires DIFFERENT AI providers to verify each other, creating a trustless
  verification network where no single provider can unilaterally claim capability.
"""

__version__ = "1.0.0"
__protocol_version__ = "1.0.0"

from .verifier import AVIRVerifier
from .benchmark import Benchmark, BenchmarkSuite, BenchmarkResult
from .attestation import Attestation, AttestationChain
from .specification import Specification
from .verdict import Verdict, VerdictLevel
from .cross_verification import (
    CrossVerifier,
    CrossVerificationMatrix,
    BlindingMode,
    IsolationLevel,
    cross_verify,
)

__all__ = [
    # Core verification
    "AVIRVerifier",
    "Benchmark",
    "BenchmarkSuite",
    "BenchmarkResult",
    "Attestation",
    "AttestationChain",
    "Specification",
    "Verdict",
    "VerdictLevel",
    # Cross-provider verification (RECOMMENDED for production)
    "CrossVerifier",
    "CrossVerificationMatrix",
    "BlindingMode",
    "IsolationLevel",
    "cross_verify",
    # Version info
    "__version__",
    "__protocol_version__",
]
