#!/usr/bin/env python3
"""
AVIR CLI - Command Line Interface for AI-Verified Independent Replication

Commands:

* ``verify``   run a double-blind cross-provider verification of a
  specification (``avir.cross_verification.CrossVerifier``) and optionally
  write an Ed25519-signed attestation.
* ``validate`` parse and validate a specification file.
* ``check``    verify an attestation's chain integrity and signature, pinned
  to a public key when one is supplied.
* ``init``     write a new specification from a template.
* ``compare``  compare the verdicts of several attestations.

Exit codes: 0 success; 1 failure or error; 2 a partial / inconsistent
result (``verify`` PARTIAL consensus, ``compare`` disagreeing verdicts).
Errors go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __protocol_version__, __version__
from .attestation import (
    Attestation,
    EnvironmentInfo,
    ProviderInfo,
    VerificationLevel,
    create_attestation,
)
from .cross_verification import (
    BlindingMode,
    CrossVerificationMatrix,
    CrossVerifier,
    IsolationLevel,
)
from .providers import get_provider
from .providers.base import AVIRProvider
from .specification import Specification
from .verdict import VerdictLevel

PROVIDER_NAMES = "claude, openai, gemini, claude-cli, codex-cli, gemini-cli"


class CLIError(Exception):
    """A user-facing error: printed to stderr without a traceback."""


def _describe(exc: BaseException) -> str:
    """Short, traceback-free description of an exception for stderr."""
    if isinstance(exc, KeyError):
        return f"missing field {exc.args[0]!r}"
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _err(message: str) -> None:
    print(f"avir: error: {message}", file=sys.stderr)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="avir",
        description="AVIR - AI-Verified Independent Replication Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cross-provider verification (at least two providers)
  avir verify spec.json --provider claude-cli --provider codex-cli

  # Pick a model per provider with NAME=MODEL
  avir verify spec.json --provider claude-cli --provider codex-cli=gpt-5.5

  # Write an Ed25519-signed attestation
  avir verify spec.json --provider claude-cli --provider codex-cli \\
      --attestation att.json --sign --key-file private.pem

  # Validate a specification file
  avir validate spec.json

  # Verify an attestation against a pinned public key
  avir check att.json --public-key public.pem

  # Initialize a new specification
  avir init my-system --output spec.json

Exit codes: 0 success, 1 failure or error, 2 partial / inconsistent result.
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"AVIR {__version__} (Protocol {__protocol_version__})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run double-blind cross-provider verification of a specification",
        description=(
            "Every provider answers the specification's benchmarks, then every "
            "other provider judges each answer blind. Self-verification is "
            "excluded from the consensus. Exit 0 on VERIFIED, 2 on PARTIAL, "
            "1 otherwise (FAILED, INCONCLUSIVE, INVALID, or an error)."
        ),
    )
    verify_parser.add_argument(
        "specification", type=Path, help="Path to specification JSON/YAML file"
    )
    verify_parser.add_argument(
        "--provider",
        action="append",
        metavar="NAME[=MODEL]",
        help=(
            f"Provider to include; repeat at least twice. NAME is one of "
            f"{PROVIDER_NAMES}. Optional =MODEL selects the provider's model. "
            "The -cli providers call headless CLIs (rubric benchmarks only); "
            "claude/openai/gemini use the vendor SDKs and need API keys."
        ),
    )
    verify_parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help=(
            "Runs per numeric benchmark per provider (default: 5). Rubric "
            "benchmarks are answered once per provider."
        ),
    )
    verify_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write results to this file instead of stdout",
    )
    verify_parser.add_argument(
        "--attestation",
        "-a",
        type=Path,
        help="Write an attestation of the run to this file",
    )
    verify_parser.add_argument(
        "--sign",
        action="store_true",
        help="Sign the attestation (requires --attestation and --key-file)",
    )
    verify_parser.add_argument(
        "--key-file",
        type=Path,
        help=(
            "Ed25519 private key for --sign: PEM (PKCS#8, unencrypted) or the "
            "hex-encoded raw key used by avir-verifier identities"
        ),
    )
    verify_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Progress output on stderr",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a specification file"
    )
    validate_parser.add_argument(
        "specification", type=Path, help="Path to specification JSON/YAML file"
    )

    # check command
    check_parser = subparsers.add_parser(
        "check",
        help="Verify an attestation's chain integrity and signature",
        description=(
            "Recomputes the attestation chain hash and verifies the Ed25519 "
            "signature. With --public-key the signature must verify against "
            "that key; an attestation signed by any other key fails. Without "
            "it, the signature is checked against the key embedded in the "
            "attestation, which only proves self-consistency, not origin. "
            "Unsigned attestations fail."
        ),
    )
    check_parser.add_argument(
        "attestation", type=Path, help="Path to attestation JSON file"
    )
    check_parser.add_argument(
        "--public-key",
        type=Path,
        help=(
            "Expected signer's Ed25519 public key: PEM (SubjectPublicKeyInfo), "
            "64 hex characters, or the base64 raw key as embedded in attestations"
        ),
    )

    # init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new specification"
    )
    init_parser.add_argument("name", type=str, help="System name")
    init_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("avir-spec.json"),
        help="Output file (default: avir-spec.json); never overwritten",
    )
    init_parser.add_argument(
        "--template",
        choices=["minimal", "standard", "comprehensive"],
        default="standard",
        help="Specification template (default: standard)",
    )

    # compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare the verdicts of several attestations",
        description=(
            "Checks each attestation's chain integrity and (self-asserted) "
            "signature, then compares verdicts. Exit 0 if all agree, 2 if they "
            "disagree, 1 on any load, integrity, or signature failure. Use "
            "'avir check --public-key' to authenticate who signed each file."
        ),
    )
    compare_parser.add_argument(
        "attestations",
        type=Path,
        nargs="+",
        help="Attestation files to compare (at least two)",
    )

    return parser


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def _fingerprint(raw_public_key: bytes) -> str:
    """Short key handle; same scheme as ``avir.network.keys.fingerprint``."""
    return hashlib.sha256(raw_public_key).hexdigest()[:16]


def load_private_key(path: Path) -> bytes:
    """Load an Ed25519 private key file and return the 32 raw key bytes."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CLIError(f"cannot read key file {path}: {_describe(exc)}") from exc

    if data.lstrip().startswith(b"-----BEGIN"):
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as exc:
            raise CLIError(
                "the 'cryptography' package is required: pip install avir-protocol[crypto]"
            ) from exc
        try:
            key = serialization.load_pem_private_key(data, password=None)
        except (ValueError, TypeError) as exc:
            raise CLIError(
                f"cannot load PEM private key {path} (encrypted keys are not "
                f"supported): {_describe(exc)}"
            ) from exc
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise CLIError(f"{path} is not an Ed25519 private key")
        raw: bytes = key.private_bytes_raw()
        return raw

    text = data.decode("ascii", errors="replace").strip()
    try:
        raw = bytes.fromhex(text)
    except ValueError:
        raw = b""
    if len(raw) != 32:
        raise CLIError(
            f"{path}: unrecognized private key format (expected Ed25519 PEM or "
            "64 hex characters)"
        )
    return raw


def load_public_key(path: Path) -> bytes:
    """Load an Ed25519 public key file and return the 32 raw key bytes."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CLIError(f"cannot read public key {path}: {_describe(exc)}") from exc

    if data.lstrip().startswith(b"-----BEGIN"):
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as exc:
            raise CLIError(
                "the 'cryptography' package is required: pip install avir-protocol[crypto]"
            ) from exc
        try:
            key = serialization.load_pem_public_key(data)
        except ValueError as exc:
            raise CLIError(
                f"cannot load PEM public key {path}: {_describe(exc)}"
            ) from exc
        if not isinstance(key, ed25519.Ed25519PublicKey):
            raise CLIError(f"{path} is not an Ed25519 public key")
        raw: bytes = key.public_bytes_raw()
        return raw

    text = data.decode("ascii", errors="replace").strip()
    candidates: list[bytes] = []
    try:
        candidates.append(bytes.fromhex(text))
    except ValueError:
        pass
    try:
        candidates.append(base64.b64decode(text, validate=True))
    except (binascii.Error, ValueError):
        pass
    for raw in candidates:
        if len(raw) == 32:
            return raw
    raise CLIError(
        f"{path}: unrecognized public key format (expected Ed25519 PEM, "
        "64 hex characters, or base64 of the 32-byte key)"
    )


def _embedded_public_key(attestation: Attestation) -> bytes | None:
    """Raw public key embedded in the attestation signature, if decodable."""
    if attestation.signature is None:
        return None
    try:
        raw = base64.b64decode(attestation.signature.public_key, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == 32 else None


def _signature_self_check(attestation: Attestation) -> bool:
    """Library check of the signature against the attestation's embedded key."""
    try:
        return attestation.verify_signature()
    except RuntimeError as exc:  # cryptography not installed
        raise CLIError(str(exc)) from exc
    except (ValueError, binascii.Error):
        # Malformed embedded key or signature encoding.
        return False


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def build_providers(entries: list[str]) -> list[AVIRProvider]:
    """Instantiate providers from ``NAME[=MODEL]`` entries."""
    providers: list[AVIRProvider] = []
    for entry in entries:
        name, sep, model = entry.partition("=")
        name = name.strip()
        kwargs: dict[str, Any] = {}
        if sep:
            if not model.strip():
                raise CLIError(f"--provider {entry!r}: empty model after '='")
            kwargs["model"] = model.strip()
        try:
            providers.append(get_provider(name, **kwargs))
        except ValueError as exc:
            raise CLIError(str(exc)) from exc

    ids = [p.provider_id for p in providers]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CLIError(
            f"providers must have distinct ids; duplicated: {', '.join(duplicates)} "
            "(e.g. 'claude' and 'claude-cli' are both id 'claude')"
        )
    return providers


def _attestation_from_matrix(
    spec: Specification,
    spec_path: Path,
    providers: list[AVIRProvider],
    matrix: CrossVerificationMatrix,
    started_at: datetime,
    completed_at: datetime,
) -> Attestation:
    """Build an attestation recording one entry per cross-verification cell."""
    cells = []
    for cell in matrix.cross_verifications:
        cells.append({
            "cell": f"{cell.verifier_id}->{cell.subject_id}",
            "verdict": "PASS" if cell.verdict == VerdictLevel.VERIFIED else "FAIL",
            "cell_verdict": cell.verdict.value,
            "pass_rate": cell.pass_rate,
            "abstain": cell.is_abstain,
            "abstain_count": cell.abstain_count,
            "canary_detected": cell.canary_detected,
            "verifier_locus": cell.verifier_locus,
        })

    matrix_dict = matrix.to_dict()
    results_hash = hashlib.sha256(
        json.dumps(matrix_dict, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

    attestation = create_attestation(
        spec_hash=spec.hash,
        env=EnvironmentInfo(cpu_cores=os.cpu_count()),
        verifier=ProviderInfo(
            provider="cross-verification",
            model=",".join(f"{p.provider_id}:{p.model}" for p in providers),
            instance_id=uuid.uuid4().hex[:8],
        ),
        benchmark_verdicts=cells,
        system_name=spec.system.name,
        system_version=spec.system.version,
        started_at=started_at,
        completed_at=completed_at,
        results_hash=results_hash,
        # Multi-provider (2+) cross-verification without a TEE.
        level=VerificationLevel.L3_ENHANCED,
        spec_source=str(spec_path),
    )
    # The attestation verdict is the matrix consensus, not the per-cell tally.
    consensus = matrix.consensus_verdict
    attestation.verdict = consensus.value
    attestation.verdict_details = (
        f"Cross-verification consensus {consensus.value} over "
        f"{len(cells)} cross cells (agreement {matrix.agreement_score:.0%}, "
        f"quorum {'met' if matrix.quorum_met else 'NOT met'})"
    )
    return attestation


async def cmd_verify(args: argparse.Namespace) -> int:
    """Execute verification command."""
    def progress(message: str) -> None:
        if args.verbose:
            print(message, file=sys.stderr)

    entries: list[str] = args.provider or []
    if len(entries) < 2:
        raise CLIError(
            "cross-provider verification needs at least two --provider options "
            f"(available: {PROVIDER_NAMES})"
        )
    if args.iterations < 1:
        raise CLIError("--iterations must be at least 1")
    if args.sign and (args.key_file is None or args.attestation is None):
        raise CLIError("--sign requires --key-file and --attestation")
    if args.key_file is not None and not args.sign:
        raise CLIError("--key-file is only used with --sign")

    # Load the signing key before running models, so a bad key fails fast.
    private_key = load_private_key(args.key_file) if args.sign else None

    try:
        spec = Specification.from_file(args.specification)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a user error
        raise CLIError(
            f"cannot load specification {args.specification}: {_describe(exc)}"
        ) from exc
    errors = spec.validate()
    if errors:
        raise CLIError(
            "invalid specification:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    providers = build_providers(entries)
    progress(f"Specification: {spec}")
    progress(f"Providers: {', '.join(f'{p.provider_id} ({p.model})' for p in providers)}")

    verifier = CrossVerifier(
        providers=providers,
        blinding_mode=BlindingMode.DOUBLE_BLIND,
        isolation_level=IsolationLevel.PROCESS,
        iterations=args.iterations,
    )
    started_at = datetime.utcnow()
    try:
        matrix = await verifier.verify(spec)
    except Exception as exc:  # noqa: BLE001 - provider/SDK errors are reported, not raised
        raise CLIError(f"verification failed: {_describe(exc)}") from exc
    completed_at = datetime.utcnow()
    consensus = matrix.consensus_verdict

    attestation = _attestation_from_matrix(
        spec, args.specification, providers, matrix, started_at, completed_at
    )

    if args.json:
        output_str = json.dumps(
            {
                "specification": str(args.specification),
                "system": {"name": spec.system.name, "version": spec.system.version},
                "consensus": {
                    "verdict": consensus.value,
                    "pass_rate": matrix.consensus_pass_rate,
                    "agreement_score": matrix.agreement_score,
                    "quorum_met": matrix.quorum_met,
                },
                "attestation_hash": attestation.hash,
                "matrix": matrix.to_dict(),
            },
            indent=2,
            default=str,
        )
    else:
        output_str = (
            f"{matrix.summary()}\n\nAttestation chain hash: {attestation.hash}"
        )

    if args.output:
        try:
            args.output.write_text(output_str + "\n")
        except OSError as exc:
            raise CLIError(f"cannot write {args.output}: {_describe(exc)}") from exc
        progress(f"Results written to: {args.output}")
    else:
        print(output_str)

    if args.attestation:
        if private_key is not None:
            try:
                attestation.sign(private_key)
            except (RuntimeError, ValueError) as exc:
                raise CLIError(f"signing failed: {_describe(exc)}") from exc
            if not _signature_self_check(attestation):
                raise CLIError("signing failed: signature does not verify")
            embedded = _embedded_public_key(attestation)
            if embedded is None:
                raise CLIError("signing failed: no decodable public key in signature")
            print(
                f"Attestation signed: Ed25519 key {_fingerprint(embedded)}",
                file=sys.stderr,
            )
        try:
            attestation.save(str(args.attestation))
        except OSError as exc:
            raise CLIError(
                f"cannot write attestation {args.attestation}: {_describe(exc)}"
            ) from exc
        progress(f"Attestation saved to: {args.attestation}")

    if consensus == VerdictLevel.VERIFIED:
        return 0
    if consensus == VerdictLevel.PARTIAL:
        return 2
    return 1


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

async def cmd_validate(args: argparse.Namespace) -> int:
    """Execute validate command."""
    try:
        spec = Specification.from_file(args.specification)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a user error
        raise CLIError(
            f"cannot load specification {args.specification}: {_describe(exc)}"
        ) from exc

    errors = spec.validate()
    if errors:
        print(f"Specification: {args.specification}", file=sys.stderr)
        print("Status: INVALID", file=sys.stderr)
        print(f"Errors ({len(errors)}):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    rubric = sum(1 for b in spec.benchmarks if b.is_rubric)
    print(f"Specification: {args.specification}")
    print("Status: VALID")
    print(f"AVIR version: {spec.avir_version}")
    print(f"System: {spec.system.name} v{spec.system.version}")
    print(
        f"Benchmarks: {len(spec.benchmarks)} "
        f"({len(spec.benchmarks) - rubric} numeric, {rubric} rubric)"
    )
    print(f"Capabilities: {len(spec.capabilities)}")
    print(f"Spec hash: {spec.hash}")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _load_attestation(path: Path) -> Attestation:
    try:
        return Attestation.load(str(path))
    except Exception as exc:  # noqa: BLE001 - any parse failure is a user error
        raise CLIError(f"cannot load attestation {path}: {_describe(exc)}") from exc


async def cmd_check(args: argparse.Namespace) -> int:
    """Execute check command."""
    attestation = _load_attestation(args.attestation)
    pinned = load_public_key(args.public_key) if args.public_key else None

    print(f"Attestation: {args.attestation}")
    print(f"System: {attestation.system_name} v{attestation.system_version}")
    print(f"Verification level: {attestation.verification_level.value}")
    print(f"Verifier: {attestation.verifier.provider} ({attestation.verifier.model})")
    print(f"Completed: {attestation.completed_at.isoformat()}Z")
    print(f"Verdict: {attestation.verdict} (not covered by the signature)")
    print(f"Chain hash: {attestation.chain.chain_hash}")

    if not attestation.verify_integrity():
        print("Chain integrity: INVALID")
        _err("chain hash does not match its components")
        return 1
    print("Chain integrity: VALID")

    if attestation.signature is None:
        print("Signature: MISSING")
        _err("attestation is unsigned; its origin cannot be verified")
        return 1

    embedded = _embedded_public_key(attestation)
    embedded_fp = _fingerprint(embedded) if embedded else "undecodable"

    if pinned is not None:
        pinned_fp = _fingerprint(pinned)
        if embedded is None or not hmac.compare_digest(embedded, pinned):
            print(f"Signature: INVALID (signed by key {embedded_fp}, expected {pinned_fp})")
            _err("attestation was not signed by the pinned public key")
            return 1
        if not _signature_self_check(attestation):
            print(f"Signature: INVALID (does not verify under pinned key {pinned_fp})")
            _err("signature verification failed")
            return 1
        print(f"Signature: VALID (pinned key {pinned_fp})")
        return 0

    if not _signature_self_check(attestation):
        print(f"Signature: INVALID (does not verify under embedded key {embedded_fp})")
        _err("signature verification failed")
        return 1
    print(f"Signature: VALID against embedded key {embedded_fp} (self-asserted, NOT pinned)")
    print(
        "warning: the key is self-asserted; anyone can re-sign with their own key. "
        "Pass --public-key to verify who signed it.",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def _numeric_benchmark(
    bench_id: str,
    description: str,
    target: float,
    unit: str,
    tolerance: float,
    runs: int,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    return {
        "id": bench_id,
        "description": description,
        "methodology": "measure",
        "target": target,
        "unit": unit,
        "tolerance": tolerance,
        "lower_is_better": lower_is_better,
        "runs": runs,
    }


def build_template(name: str, template: str) -> dict[str, Any]:
    """Return a specification template that ``Specification.validate`` accepts.

    Targets are starting values for the author to replace with the system's
    claimed figures.
    """
    created = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    spec: dict[str, Any] = {
        "avir_version": __protocol_version__,
        "metadata": {
            "name": name,
            "description": f"AVIR specification for {name}",
            "created": created,
        },
        "system": {"name": name, "version": "1.0.0"},
    }

    if template == "minimal":
        spec["benchmarks"] = [
            _numeric_benchmark("throughput", "Operations per second", 100, "ops/s", 0.2, 5),
        ]
        spec["capabilities"] = [
            {
                "name": "performance",
                "description": "Sustained throughput",
                "category": "performance",
                "benchmarks": ["throughput"],
            },
        ]
        return spec

    spec["system"].update({"description": "", "repository": ""})
    spec["metadata"].update({"author": "", "tags": []})

    if template == "standard":
        spec["benchmarks"] = [
            _numeric_benchmark("throughput", "Operations per second", 100, "ops/s", 0.2, 5),
            _numeric_benchmark(
                "latency", "Response time in milliseconds", 50, "ms", 0.25, 5,
                lower_is_better=True,
            ),
        ]
        spec["capabilities"] = [
            {
                "name": "performance",
                "description": "Throughput and latency",
                "category": "performance",
                "benchmarks": ["throughput", "latency"],
            },
        ]
        return spec

    # comprehensive: throughput, latency percentiles and memory
    spec["metadata"].update({"organization": "", "license": "MIT"})
    spec["benchmarks"] = [
        _numeric_benchmark(
            "throughput", "Operations per second under load", 100, "ops/s", 0.2, 10
        ),
        _numeric_benchmark(
            "latency_p50", "Median response time", 50, "ms", 0.2, 10,
            lower_is_better=True,
        ),
        _numeric_benchmark(
            "latency_p99", "99th percentile response time", 200, "ms", 0.3, 10,
            lower_is_better=True,
        ),
        _numeric_benchmark(
            "memory_usage", "Peak memory consumption", 512, "MB", 0.25, 5,
            lower_is_better=True,
        ),
    ]
    spec["capabilities"] = [
        {
            "name": "performance",
            "description": "Throughput and latency under load",
            "category": "performance",
            "benchmarks": ["throughput", "latency_p50", "latency_p99"],
        },
        {
            "name": "resources",
            "description": "Resource consumption",
            "category": "performance",
            "benchmarks": ["memory_usage"],
        },
    ]
    return spec


async def cmd_init(args: argparse.Namespace) -> int:
    """Execute init command."""
    if args.output.exists():
        raise CLIError(f"{args.output} already exists; choose another --output")

    spec = build_template(args.name, args.template)
    errors = Specification.from_dict(spec).validate()
    if errors:  # a template the validator rejects is a bug in this command
        raise CLIError(f"template {args.template!r} is invalid: {errors}")

    try:
        with args.output.open("x") as fh:
            fh.write(json.dumps(spec, indent=2) + "\n")
    except OSError as exc:
        raise CLIError(f"cannot write {args.output}: {_describe(exc)}") from exc

    print(f"Created specification: {args.output}")
    print(f"Template: {args.template}")
    print(f"Benchmarks: {len(spec['benchmarks'])}")
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

async def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare command."""
    if len(args.attestations) < 2:
        raise CLIError("need at least 2 attestations to compare")

    loaded = [(path, _load_attestation(path)) for path in args.attestations]

    print("AVIR Attestation Comparison")
    print("=" * 60)
    print()

    failures = 0
    print("Attestations:")
    for path, att in loaded:
        integrity = "VALID" if att.verify_integrity() else "INVALID"
        if att.signature is None:
            signature = "unsigned"
        elif _signature_self_check(att):
            embedded = _embedded_public_key(att)
            signature = (
                f"valid, key {_fingerprint(embedded)} (self-asserted)"
                if embedded else "valid"
            )
        else:
            signature = "INVALID"
        if integrity == "INVALID" or signature == "INVALID":
            failures += 1
        print(f"  {path.name}:")
        print(f"    verdict: {att.verdict}")
        print(f"    system: {att.system_name} v{att.system_version}")
        print(f"    verifier: {att.verifier.provider} ({att.verifier.model})")
        print(f"    completed: {att.completed_at.isoformat()}Z")
        print(f"    chain integrity: {integrity}")
        print(f"    signature: {signature}")

    if failures:
        _err(f"{failures} attestation(s) failed integrity or signature checks")
        return 1

    verdicts = {att.verdict for _, att in loaded}
    systems = {(att.system_name, att.system_version) for _, att in loaded}
    consistent = len(verdicts) == 1

    print()
    if len(systems) > 1:
        print("Warning: attestations cover different systems/versions")
    print(f"Consensus: {'YES' if consistent else 'NO'}")
    if not consistent:
        print("Verdicts are inconsistent across attestations")

    return 0 if consistent else 2


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "verify": cmd_verify,
        "validate": cmd_validate,
        "check": cmd_check,
        "init": cmd_init,
        "compare": cmd_compare,
    }

    try:
        return asyncio.run(commands[args.command](args))
    except CLIError as exc:
        _err(str(exc))
        return 1
    except KeyboardInterrupt:
        _err("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
