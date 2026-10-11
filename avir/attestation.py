"""
AVIR Attestation System

Provides cryptographic attestation of verification results,
including hash chains and digital signatures.
"""

import base64
import binascii
import copy
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class VerificationLevel(Enum):
    """Verification assurance levels."""
    L1_BASIC = "L1"           # Single provider, 1 run
    L2_STANDARD = "L2"        # Single provider, 5 runs, isolation
    L3_ENHANCED = "L3"        # Multi-provider (2+), 5 runs each
    L4_COMPREHENSIVE = "L4"   # Multi-provider (3+), 10 runs, TEE


@dataclass
class EnvironmentInfo:
    """Verification environment metadata."""
    container: Optional[str] = None
    os: str = field(default_factory=lambda: platform.system().lower())
    arch: str = field(default_factory=platform.machine)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'container': self.container,
            'os': self.os,
            'arch': self.arch,
            'python_version': self.python_version,
            'resources': {
                'cpu_cores': self.cpu_cores,
                'memory_gb': self.memory_gb,
            }
        }

    def hash(self) -> str:
        """Calculate environment hash."""
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ProviderInfo:
    """AI verification provider metadata."""
    provider: str  # claude, openai, gemini, ollama
    model: str
    instance_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'provider': self.provider,
            'model': self.model,
            'instance_id': self.instance_id,
        }


@dataclass
class AttestationChain:
    """
    Cryptographic chain proving verification integrity.

    The chain links:
    1. Specification hash - what was being verified
    2. Environment hash - where verification ran
    3. Results hash - what the outcomes were
    4. Timestamp - when verification occurred

    The chain_hash combines all above, making tampering evident.
    """
    spec_hash: str
    env_hash: str
    results_hash: str
    timestamp: str
    chain_hash: str = ""

    def __post_init__(self) -> None:
        if not self.chain_hash:
            self.chain_hash = self.calculate_chain_hash()

    def calculate_chain_hash(self) -> str:
        """Calculate the chain hash from components."""
        data = f"{self.spec_hash}{self.env_hash}{self.results_hash}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify chain integrity."""
        expected = self.calculate_chain_hash()
        return self.chain_hash == expected

    def to_dict(self) -> dict[str, Any]:
        return {
            'spec_hash': self.spec_hash,
            'env_hash': self.env_hash,
            'results_hash': self.results_hash,
            'timestamp': self.timestamp,
            'chain_hash': self.chain_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttestationChain":
        return cls(
            spec_hash=data['spec_hash'],
            env_hash=data['env_hash'],
            results_hash=data['results_hash'],
            timestamp=data['timestamp'],
            chain_hash=data.get('chain_hash', ''),
        )


# Signature scope: what the signed message covers.
#   "document" -- the canonical serialization of the whole attestation
#                 (``Attestation.signing_payload()``): every field except the
#                 ``signature`` object itself.
#   absent     -- legacy scope: only ``attestation_chain.chain_hash`` was
#                 signed, which does not bind the verdict, results, system,
#                 verifier, environment, execution or level fields. Legacy
#                 signatures never verify (fail closed).
SIGNATURE_SCOPE_DOCUMENT = "document"


@dataclass
class Signature:
    """Digital signature for attestation."""
    algorithm: str  # Ed25519
    public_key: str  # Base64 encoded
    signature: str   # Base64 encoded
    scope: Optional[str] = None  # None = legacy chain-hash-only signature

    @property
    def is_legacy(self) -> bool:
        """True for a signature without a recorded scope (chain hash only)."""
        return self.scope is None

    def to_dict(self) -> dict[str, Any]:
        data = {
            'algorithm': self.algorithm,
            'public_key': self.public_key,
            'signature': self.signature,
        }
        if self.scope is not None:
            data['scope'] = self.scope
        return data


@dataclass
class Attestation:
    """
    Complete AVIR attestation document.

    Provides cryptographic proof of verification including:
    - System under test identification
    - Verification environment
    - Benchmark results with statistics
    - Cryptographic chain
    - Optional digital signature
    - Final verdict
    """
    protocol_version: str
    verification_level: VerificationLevel

    # System info
    system_name: str
    system_version: str
    spec_source: Optional[str]

    # Verifier info
    verifier: ProviderInfo

    # Environment
    environment: EnvironmentInfo

    # Execution metadata
    started_at: datetime
    completed_at: datetime

    # Results
    benchmark_verdicts: list[dict[str, Any]]
    summary: dict[str, Any]

    # Attestation chain
    chain: AttestationChain

    # Signature (optional)
    signature: Optional[Signature] = None

    # Final verdict
    verdict: str = ""
    verdict_details: str = ""

    # The document this attestation was loaded from, kept so verification
    # also covers fields that from_dict() does not parse back or recomputes
    # (e.g. execution.duration_seconds, unknown keys).
    _source: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.verdict:
            self.verdict = self._determine_verdict()
            self.verdict_details = self._generate_verdict_details()

    def _determine_verdict(self) -> str:
        """Determine overall verdict from benchmark results."""
        total = self.summary.get('total', 0)
        passed = self.summary.get('passed', 0)

        if total == 0:
            return "INVALID"

        pass_rate = passed / total

        if pass_rate == 1.0:
            return "VERIFIED"
        elif pass_rate >= 0.6:
            return "PARTIAL"
        else:
            return "FAILED"

    def _generate_verdict_details(self) -> str:
        """Generate human-readable verdict explanation."""
        total = self.summary.get('total', 0)
        passed = self.summary.get('passed', 0)
        failed = self.summary.get('failed', 0)

        if self.verdict == "VERIFIED":
            return f"All {total} benchmarks passed within tolerance"
        elif self.verdict == "PARTIAL":
            return f"{passed}/{total} benchmarks passed ({failed} failed)"
        elif self.verdict == "FAILED":
            return f"Only {passed}/{total} benchmarks passed ({failed} failed)"
        else:
            return "Verification could not be completed"

    @property
    def duration_seconds(self) -> float:
        """Calculate verification duration."""
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def hash(self) -> str:
        """Get attestation chain hash."""
        return self.chain.chain_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert attestation to dictionary."""
        return {
            'avir_protocol_version': self.protocol_version,
            'verification_level': self.verification_level.value,

            'system': {
                'name': self.system_name,
                'version': self.system_version,
                'spec_source': self.spec_source,
            },

            'verifier': self.verifier.to_dict(),
            'environment': self.environment.to_dict(),

            'execution': {
                'started_at': self.started_at.isoformat() + 'Z',
                'completed_at': self.completed_at.isoformat() + 'Z',
                'duration_seconds': round(self.duration_seconds, 2),
            },

            'results': {
                'benchmarks': self.benchmark_verdicts,
                'summary': self.summary,
            },

            'attestation_chain': self.chain.to_dict(),
            'signature': self.signature.to_dict() if self.signature else None,

            'verdict': self.verdict,
            'verdict_details': self.verdict_details,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert attestation to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save attestation to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attestation":
        """Create attestation from dictionary."""
        attestation = cls(
            protocol_version=data['avir_protocol_version'],
            verification_level=VerificationLevel(data['verification_level']),
            system_name=data['system']['name'],
            system_version=data['system']['version'],
            spec_source=data['system'].get('spec_source'),
            verifier=ProviderInfo(
                provider=data['verifier']['provider'],
                model=data['verifier']['model'],
                instance_id=data['verifier']['instance_id'],
            ),
            environment=EnvironmentInfo(
                container=data['environment'].get('container'),
                os=data['environment']['os'],
                arch=data['environment']['arch'],
                python_version=data['environment'].get('python_version', ''),
                cpu_cores=data['environment']['resources'].get('cpu_cores'),
                memory_gb=data['environment']['resources'].get('memory_gb'),
            ),
            started_at=datetime.fromisoformat(data['execution']['started_at'].rstrip('Z')),
            completed_at=datetime.fromisoformat(data['execution']['completed_at'].rstrip('Z')),
            benchmark_verdicts=data['results']['benchmarks'],
            summary=data['results']['summary'],
            chain=AttestationChain.from_dict(data['attestation_chain']),
            signature=Signature(**data['signature']) if data.get('signature') else None,
            verdict=data['verdict'],
            verdict_details=data['verdict_details'],
        )
        attestation._source = copy.deepcopy(data)
        return attestation

    @classmethod
    def from_json(cls, json_str: str) -> "Attestation":
        """Create attestation from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str) -> "Attestation":
        """Load attestation from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())

    def verify_integrity(self) -> bool:
        """Verify attestation chain integrity."""
        return self.chain.verify()

    @staticmethod
    def _canonical(document: dict[str, Any]) -> bytes:
        """Deterministic JSON bytes: sorted keys, compact separators.

        The document is passed through one JSON round trip first, so the
        bytes are the same before save and after load (tuples become lists,
        non-string keys become strings and sort as strings).
        """
        normalized = json.loads(json.dumps(document))
        return json.dumps(
            normalized, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
        ).encode()

    def signing_payload(self) -> bytes:
        """Bytes covered by a ``document``-scope signature.

        The canonical serialization of ``to_dict()`` with the ``signature``
        field removed.
        """
        document = self.to_dict()
        del document['signature']
        return self._canonical(document)

    def sign(self, private_key: bytes, algorithm: str = "Ed25519") -> None:
        """
        Sign the whole attestation document with a private key.

        The signature covers ``signing_payload()`` and is recorded with
        ``scope="document"``.

        Args:
            private_key: 32-byte raw Ed25519 private key
            algorithm: Signing algorithm (only Ed25519 is supported)
        """
        if algorithm != "Ed25519":
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as e:
            raise RuntimeError("cryptography package required for signing") from e

        signing_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
        # The signed content is this object as it is now, not the document it
        # may have been loaded from.
        self._source = None
        signature_bytes = signing_key.sign(self.signing_payload())
        public_key_bytes = signing_key.public_key().public_bytes_raw()

        self.signature = Signature(
            algorithm=algorithm,
            public_key=base64.b64encode(public_key_bytes).decode(),
            signature=base64.b64encode(signature_bytes).decode(),
            scope=SIGNATURE_SCOPE_DOCUMENT,
        )

    def verify_signature(self) -> bool:
        """
        Verify the attestation signature against its embedded public key.

        Returns True only for an Ed25519 signature with ``scope="document"``
        over ``signing_payload()``, and, for an attestation loaded from a
        document, only if that document is exactly what this object
        serializes to (no edited or extra fields that parsing drops or
        recomputes). Legacy chain-hash-only signatures return False: they do
        not authenticate the attestation content.

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.signature:
            return False
        if self.signature.scope != SIGNATURE_SCOPE_DOCUMENT:
            return False
        if self.signature.algorithm != "Ed25519":
            return False

        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as e:
            raise RuntimeError("cryptography package required for signature verification") from e

        payload = self.signing_payload()
        if self._source is not None:
            source = copy.deepcopy(self._source)
            source.pop('signature', None)
            if self._canonical(source) != payload:
                return False

        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                base64.b64decode(self.signature.public_key, validate=True)
            )
            signature = base64.b64decode(self.signature.signature, validate=True)
        except (binascii.Error, ValueError):
            return False

        try:
            public_key.verify(signature, payload)
        except InvalidSignature:
            return False
        return True


def create_attestation(
    spec_hash: str,
    env: EnvironmentInfo,
    verifier: ProviderInfo,
    benchmark_verdicts: list[dict[str, Any]],
    system_name: str,
    system_version: str,
    started_at: datetime,
    completed_at: datetime,
    results_hash: str,
    level: VerificationLevel = VerificationLevel.L2_STANDARD,
    spec_source: Optional[str] = None,
) -> Attestation:
    """
    Factory function to create attestation.

    Args:
        spec_hash: Hash of specification
        env: Environment info
        verifier: Provider info
        benchmark_verdicts: List of verdict dictionaries
        system_name: Name of system being verified
        system_version: Version of system
        started_at: When verification started
        completed_at: When verification completed
        results_hash: Hash of results
        level: Verification level
        spec_source: Optional URL to specification

    Returns:
        Complete Attestation object
    """
    timestamp = completed_at.isoformat() + 'Z'

    # Create attestation chain
    chain = AttestationChain(
        spec_hash=spec_hash,
        env_hash=env.hash(),
        results_hash=results_hash,
        timestamp=timestamp,
    )

    # Calculate summary
    passed = sum(1 for v in benchmark_verdicts if v.get('verdict') == 'PASS')
    total = len(benchmark_verdicts)

    summary = {
        'total': total,
        'passed': passed,
        'failed': total - passed,
        'pass_rate': passed / total if total > 0 else 0.0,
    }

    return Attestation(
        protocol_version="1.0.0",
        verification_level=level,
        system_name=system_name,
        system_version=system_version,
        spec_source=spec_source,
        verifier=verifier,
        environment=env,
        started_at=started_at,
        completed_at=completed_at,
        benchmark_verdicts=benchmark_verdicts,
        summary=summary,
        chain=chain,
    )
