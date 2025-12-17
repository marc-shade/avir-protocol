#!/usr/bin/env python3
"""
AVIR CLI - Command Line Interface for AI-Verified Independent Replication

Main entry point for running AVIR verifications from the command line.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import __version__, __protocol_version__
from .verifier import AVIRVerifier, VerificationResult
from .specification import Specification
from .attestation import Attestation, VerificationLevel
from .verdict import VerdictLevel
from .providers import get_provider


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="avir",
        description="AVIR - AI-Verified Independent Replication Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run verification with default settings
  avir verify spec.json

  # Use specific provider
  avir verify spec.json --provider claude --model claude-sonnet-4-20250514

  # Run with multiple iterations for statistical rigor
  avir verify spec.json --iterations 10

  # Generate attestation with signing
  avir verify spec.json --sign --key-file private.pem

  # Validate a specification file
  avir validate spec.json

  # Verify an existing attestation
  avir check attestation.json

  # Initialize a new specification
  avir init my-system --output spec.json
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
        "verify", help="Run verification against a specification"
    )
    verify_parser.add_argument(
        "specification", type=Path, help="Path to specification JSON file"
    )
    verify_parser.add_argument(
        "--provider",
        choices=["claude", "openai", "gemini"],
        default="claude",
        help="AI provider to use (default: claude)",
    )
    verify_parser.add_argument(
        "--model", type=str, help="Specific model to use (provider-dependent)"
    )
    verify_parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    verify_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for results (default: stdout)",
    )
    verify_parser.add_argument(
        "--attestation",
        "-a",
        type=Path,
        help="Output file for attestation",
    )
    verify_parser.add_argument(
        "--sign",
        action="store_true",
        help="Sign the attestation",
    )
    verify_parser.add_argument(
        "--key-file",
        type=Path,
        help="Private key file for signing (Ed25519 PEM)",
    )
    verify_parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Verification level (1-4, default: 2)",
    )
    verify_parser.add_argument(
        "--container",
        action="store_true",
        help="Run verification in isolated container",
    )
    verify_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
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
        "specification", type=Path, help="Path to specification JSON file"
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict validation mode",
    )

    # check command
    check_parser = subparsers.add_parser(
        "check", help="Verify an existing attestation"
    )
    check_parser.add_argument(
        "attestation", type=Path, help="Path to attestation JSON file"
    )
    check_parser.add_argument(
        "--public-key",
        type=Path,
        help="Public key file for signature verification",
    )

    # init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new specification"
    )
    init_parser.add_argument(
        "name", type=str, help="System name"
    )
    init_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("avir-spec.json"),
        help="Output file (default: avir-spec.json)",
    )
    init_parser.add_argument(
        "--template",
        choices=["minimal", "standard", "comprehensive"],
        default="standard",
        help="Specification template (default: standard)",
    )

    # compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple verification results"
    )
    compare_parser.add_argument(
        "attestations",
        type=Path,
        nargs="+",
        help="Attestation files to compare",
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for comparison report",
    )

    return parser


async def cmd_verify(args: argparse.Namespace) -> int:
    """Execute verification command."""
    # Load specification
    try:
        spec = Specification.from_file(args.specification)
    except Exception as e:
        print(f"Error loading specification: {e}", file=sys.stderr)
        return 1

    # Validate specification
    errors = spec.validate()
    if errors:
        print("Specification validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded specification: {spec.metadata.get('name', 'unnamed')}")
        print(f"  Version: {spec.version}")
        print(f"  Benchmarks: {len(spec.benchmarks)}")

    # Create provider
    provider_kwargs = {}
    if args.model:
        provider_kwargs["model"] = args.model

    try:
        provider = get_provider(args.provider, **provider_kwargs)
    except ValueError as e:
        print(f"Error creating provider: {e}", file=sys.stderr)
        return 1

    # Create verifier
    verifier = AVIRVerifier(
        provider=provider,
        iterations=args.iterations,
        verification_level=VerificationLevel(args.level),
    )

    if args.verbose:
        print(f"Using provider: {args.provider}")
        print(f"Iterations: {args.iterations}")
        print(f"Verification level: L{args.level}")
        print()

    # Run verification
    try:
        if args.verbose:
            print("Running verification...")
        result = await verifier.verify(spec)
    except Exception as e:
        print(f"Verification error: {e}", file=sys.stderr)
        return 1

    # Output results
    if args.json:
        output = {
            "verdict": result.verdict.level.value,
            "pass_rate": result.verdict.pass_rate,
            "passed": result.verdict.passed_count,
            "total": result.verdict.total_count,
            "timestamp": result.attestation.timestamp if result.attestation else None,
            "provider": args.provider,
            "results": [
                {
                    "benchmark": r.benchmark_id,
                    "verdict": r.verdict.value,
                    "measured": r.statistics.mean if r.statistics else None,
                    "target": r.target,
                }
                for r in result.results
            ],
        }
        output_str = json.dumps(output, indent=2, default=str)
    else:
        output_str = result.summary()

    if args.output:
        args.output.write_text(output_str)
        if args.verbose:
            print(f"Results written to: {args.output}")
    else:
        print(output_str)

    # Save attestation if requested
    if args.attestation and result.attestation:
        if args.sign and args.key_file:
            try:
                key_data = args.key_file.read_bytes()
                result.attestation.sign(key_data, "file")
                if args.verbose:
                    print(f"Attestation signed with key: {args.key_file}")
            except Exception as e:
                print(f"Warning: Failed to sign attestation: {e}", file=sys.stderr)

        result.attestation.save(args.attestation)
        if args.verbose:
            print(f"Attestation saved to: {args.attestation}")

    # Return code based on verdict
    if result.is_verified:
        return 0
    elif result.verdict.level == VerdictLevel.PARTIAL:
        return 2
    else:
        return 1


async def cmd_validate(args: argparse.Namespace) -> int:
    """Execute validate command."""
    try:
        spec = Specification.from_file(args.specification)
    except Exception as e:
        print(f"Error loading specification: {e}", file=sys.stderr)
        return 1

    errors = spec.validate()

    if errors:
        print(f"Specification: {args.specification}")
        print(f"Status: INVALID")
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print(f"Specification: {args.specification}")
        print(f"Status: VALID")
        print(f"Version: {spec.version}")
        print(f"Benchmarks: {len(spec.benchmarks)}")
        if spec.capabilities:
            print(f"Capabilities: {len(spec.capabilities)}")
        return 0


async def cmd_check(args: argparse.Namespace) -> int:
    """Execute check command."""
    try:
        attestation = Attestation.load(args.attestation)
    except Exception as e:
        print(f"Error loading attestation: {e}", file=sys.stderr)
        return 1

    print(f"Attestation: {args.attestation}")
    print(f"Timestamp: {attestation.timestamp}")
    print(f"Verdict: {attestation.verdict}")
    print(f"Provider: {attestation.provider.provider_id if attestation.provider else 'unknown'}")
    print(f"Chain Hash: {attestation.chain.chain_hash[:16]}...")

    # Verify chain integrity
    if attestation.chain.verify():
        print(f"Chain Integrity: VALID")
    else:
        print(f"Chain Integrity: INVALID")
        return 1

    # Verify signature if public key provided
    if args.public_key and attestation.signature:
        try:
            key_data = args.public_key.read_bytes()
            if attestation.verify_signature(key_data):
                print(f"Signature: VALID")
            else:
                print(f"Signature: INVALID")
                return 1
        except Exception as e:
            print(f"Signature verification error: {e}", file=sys.stderr)
            return 1
    elif attestation.signature:
        print(f"Signature: Present (not verified - no public key provided)")

    return 0


async def cmd_init(args: argparse.Namespace) -> int:
    """Execute init command."""
    templates = {
        "minimal": {
            "version": "1.0.0",
            "metadata": {
                "name": args.name,
                "description": f"AVIR specification for {args.name}",
                "created": datetime.utcnow().isoformat() + "Z",
            },
            "system": {
                "name": args.name,
                "version": "1.0.0",
            },
            "benchmarks": [
                {
                    "id": "example_benchmark",
                    "name": "Example Benchmark",
                    "target": 100,
                    "unit": "ops/s",
                    "tolerance": 0.2,
                }
            ],
        },
        "standard": {
            "version": "1.0.0",
            "metadata": {
                "name": args.name,
                "description": f"AVIR specification for {args.name}",
                "author": "",
                "created": datetime.utcnow().isoformat() + "Z",
                "tags": [],
            },
            "system": {
                "name": args.name,
                "version": "1.0.0",
                "description": "",
                "repository": "",
            },
            "benchmarks": [
                {
                    "id": "throughput",
                    "name": "Throughput",
                    "description": "Operations per second",
                    "target": 100,
                    "unit": "ops/s",
                    "tolerance": 0.2,
                    "iterations": 5,
                },
                {
                    "id": "latency",
                    "name": "Latency",
                    "description": "Response time in milliseconds",
                    "target": 50,
                    "unit": "ms",
                    "tolerance": 0.25,
                    "lower_is_better": True,
                    "iterations": 5,
                },
            ],
            "capabilities": [],
            "requirements": {
                "min_iterations": 3,
                "pass_threshold": 0.8,
            },
        },
        "comprehensive": {
            "version": "1.0.0",
            "metadata": {
                "name": args.name,
                "description": f"Comprehensive AVIR specification for {args.name}",
                "author": "",
                "organization": "",
                "created": datetime.utcnow().isoformat() + "Z",
                "tags": [],
                "license": "MIT",
            },
            "system": {
                "name": args.name,
                "version": "1.0.0",
                "description": "",
                "repository": "",
                "documentation": "",
                "dependencies": [],
            },
            "benchmarks": [
                {
                    "id": "throughput",
                    "name": "Throughput",
                    "description": "Operations per second under load",
                    "category": "performance",
                    "target": 100,
                    "unit": "ops/s",
                    "tolerance": 0.2,
                    "iterations": 10,
                    "warmup": 2,
                    "outlier_policy": "iqr",
                },
                {
                    "id": "latency_p50",
                    "name": "P50 Latency",
                    "description": "Median response time",
                    "category": "performance",
                    "target": 50,
                    "unit": "ms",
                    "tolerance": 0.2,
                    "lower_is_better": True,
                    "iterations": 10,
                },
                {
                    "id": "latency_p99",
                    "name": "P99 Latency",
                    "description": "99th percentile response time",
                    "category": "performance",
                    "target": 200,
                    "unit": "ms",
                    "tolerance": 0.3,
                    "lower_is_better": True,
                    "iterations": 10,
                },
                {
                    "id": "memory_usage",
                    "name": "Memory Usage",
                    "description": "Peak memory consumption",
                    "category": "resource",
                    "target": 512,
                    "unit": "MB",
                    "tolerance": 0.25,
                    "lower_is_better": True,
                },
                {
                    "id": "accuracy",
                    "name": "Accuracy",
                    "description": "Correctness of results",
                    "category": "correctness",
                    "target": 0.99,
                    "unit": "ratio",
                    "tolerance": 0.01,
                },
            ],
            "capabilities": [
                {
                    "name": "feature_a",
                    "description": "Feature A capability",
                    "required": True,
                },
                {
                    "name": "feature_b",
                    "description": "Feature B capability",
                    "required": False,
                },
            ],
            "requirements": {
                "min_iterations": 5,
                "pass_threshold": 0.8,
                "outlier_removal": True,
                "statistical_significance": 0.95,
            },
            "environment": {
                "container": True,
                "isolation_level": "process",
                "timeout": 300,
            },
        },
    }

    spec = templates[args.template]

    try:
        args.output.write_text(json.dumps(spec, indent=2))
        print(f"Created specification: {args.output}")
        print(f"Template: {args.template}")
        print(f"Benchmarks: {len(spec['benchmarks'])}")
        return 0
    except Exception as e:
        print(f"Error writing specification: {e}", file=sys.stderr)
        return 1


async def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare command."""
    attestations = []

    for path in args.attestations:
        try:
            att = Attestation.load(path)
            attestations.append((path, att))
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            return 1

    if len(attestations) < 2:
        print("Need at least 2 attestations to compare", file=sys.stderr)
        return 1

    print("AVIR Attestation Comparison")
    print("=" * 60)
    print()

    # Compare verdicts
    verdicts = [att.verdict for _, att in attestations]
    consistent = len(set(verdicts)) == 1

    print("Verdicts:")
    for path, att in attestations:
        print(f"  {path.name}: {att.verdict}")

    print()
    print(f"Consensus: {'YES' if consistent else 'NO'}")

    if not consistent:
        print()
        print("Warning: Verdicts are inconsistent across verifications!")

    # Compare providers
    print()
    print("Providers:")
    for path, att in attestations:
        provider = att.provider.provider_id if att.provider else "unknown"
        model = att.provider.model if att.provider else "unknown"
        print(f"  {path.name}: {provider} ({model})")

    # Compare timestamps
    print()
    print("Timestamps:")
    for path, att in attestations:
        print(f"  {path.name}: {att.timestamp}")

    return 0 if consistent else 2


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Run appropriate command
    commands = {
        "verify": cmd_verify,
        "validate": cmd_validate,
        "check": cmd_check,
        "init": cmd_init,
        "compare": cmd_compare,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return asyncio.run(cmd_func(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
