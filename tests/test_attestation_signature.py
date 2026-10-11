"""
Tests for document-scope attestation signatures (avir.attestation).

The signature must cover the canonical serialization of the whole attestation
(``to_dict()`` without ``signature``): it verifies after save -> load, and fails
when any single field anywhere in the saved document changes. Legacy
signatures (no ``scope``; chain hash only) never verify.

Gaps / not covered:
- Only Ed25519 is exercised; no other algorithm is implemented.
- Attestations written by other implementations of the format are not tested.
"""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from avir.attestation import (
    SIGNATURE_SCOPE_DOCUMENT,
    Attestation,
    EnvironmentInfo,
    ProviderInfo,
    Signature,
    VerificationLevel,
    create_attestation,
)


def make_attestation() -> Attestation:
    started = datetime(2026, 10, 11, 12, 0, 0, 123456)
    return create_attestation(
        spec_hash="a" * 64,
        env=EnvironmentInfo(container=None, cpu_cores=8, memory_gb=31.5),
        verifier=ProviderInfo(provider="cross-verification", model="a,b", instance_id="1234abcd"),
        benchmark_verdicts=[
            {
                "benchmark_id": "throughput",
                "verdict": "PASS",
                "target": 100,
                "result": 101.337,
                "tolerance": 0.2,
                "abstain": False,
                "statistics": {"mean": 101.337, "std_dev": 0.25, "count": 5},
            },
            {"benchmark_id": "latency", "verdict": "FAIL", "result": 75.5, "note": None},
        ],
        system_name="sys",
        system_version="1.2.3",
        started_at=started,
        completed_at=started + timedelta(seconds=12, microseconds=345678),
        results_hash="b" * 64,
        level=VerificationLevel.L3_ENHANCED,
        spec_source="spec.json",
    )


@pytest.fixture
def key() -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.generate()


@pytest.fixture
def signed_document(key: ed25519.Ed25519PrivateKey, tmp_path: Path) -> dict[str, Any]:
    att = make_attestation()
    att.sign(key.private_bytes_raw())
    path = tmp_path / "att.json"
    att.save(str(path))
    document: dict[str, Any] = json.loads(path.read_text())
    return document


def leaf_paths(node: Any, prefix: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    """Every leaf path in a JSON document."""
    if isinstance(node, dict) and node:
        for k, v in node.items():
            yield from leaf_paths(v, prefix + (k,))
    elif isinstance(node, list) and node:
        for i, v in enumerate(node):
            yield from leaf_paths(v, prefix + (i,))
    else:
        yield prefix


def mutated(value: Any, path: tuple[Any, ...]) -> Any:
    """A different value of a type that still loads."""
    if path == ("verification_level",):
        return "L2" if value != "L2" else "L1"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        if value.endswith("Z") and "T" in value:  # ISO timestamp
            stamp = datetime.fromisoformat(value.rstrip("Z")) + timedelta(seconds=1)
            return stamp.isoformat() + "Z"
        return value + "x"
    if value is None:
        return "x" if path[-1] in ("container", "spec_source", "note") else 1
    raise AssertionError(f"unhandled leaf {path}: {value!r}")


def set_path(document: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    node: Any = document
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


def get_path(document: dict[str, Any], path: tuple[Any, ...]) -> Any:
    node: Any = document
    for part in path:
        node = node[part]
    return node


# ---------------------------------------------------------------------------

def test_sign_records_document_scope(key: ed25519.Ed25519PrivateKey) -> None:
    att = make_attestation()
    att.sign(key.private_bytes_raw())
    assert att.signature is not None
    assert att.signature.scope == SIGNATURE_SCOPE_DOCUMENT
    assert att.to_dict()["signature"]["scope"] == "document"
    assert att.verify_signature()


def test_round_trip_save_load_verify(
    key: ed25519.Ed25519PrivateKey, tmp_path: Path
) -> None:
    att = make_attestation()
    att.sign(key.private_bytes_raw())
    path = tmp_path / "att.json"
    att.save(str(path))
    loaded = Attestation.load(str(path))
    # to_dict round-trips exactly (datetimes with Z, floats, duration rounding)
    assert loaded.to_dict() == att.to_dict()
    assert loaded.to_dict() == json.loads(path.read_text())
    assert loaded.signing_payload() == att.signing_payload()
    assert loaded.verify_signature()


def test_signing_payload_is_canonical(key: ed25519.Ed25519PrivateKey) -> None:
    att = make_attestation()
    payload = att.signing_payload()
    document = att.to_dict()
    del document["signature"]
    assert payload == json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    att.sign(key.private_bytes_raw())
    assert att.signing_payload() == payload  # signature itself is excluded


def test_every_field_change_fails(signed_document: dict[str, Any]) -> None:
    paths = [
        p for p in leaf_paths(signed_document) if p and p[0] != "signature"
    ]
    assert len(paths) > 30
    for path in paths:
        document = copy.deepcopy(signed_document)
        set_path(document, path, mutated(get_path(document, path), path))
        att = Attestation.from_dict(document)
        assert not att.verify_signature(), f"change at {path} still verifies"


def test_unparsed_or_extra_fields_fail(signed_document: dict[str, Any]) -> None:
    # duration_seconds is recomputed by from_dict, not read back
    document = copy.deepcopy(signed_document)
    document["execution"]["duration_seconds"] += 100
    assert not Attestation.from_dict(document).verify_signature()

    document = copy.deepcopy(signed_document)
    document["extra"] = {"claim": "added after signing"}
    assert not Attestation.from_dict(document).verify_signature()

    assert Attestation.from_dict(copy.deepcopy(signed_document)).verify_signature()


def test_in_memory_change_after_load_fails(signed_document: dict[str, Any]) -> None:
    att = Attestation.from_dict(signed_document)
    att.verdict = "FAILED" if att.verdict != "FAILED" else "VERIFIED"
    assert not att.verify_signature()


def test_resign_after_change_verifies(
    signed_document: dict[str, Any], key: ed25519.Ed25519PrivateKey
) -> None:
    att = Attestation.from_dict(signed_document)
    att.verdict = "FAILED"
    att.sign(key.private_bytes_raw())
    assert att.verify_signature()
    reloaded = Attestation.from_dict(att.to_dict())
    assert reloaded.verify_signature()


def legacy_signature(att: Attestation, key: ed25519.Ed25519PrivateKey) -> Signature:
    """A signature in the pre-scope format: over the chain hash only."""
    return Signature(
        algorithm="Ed25519",
        public_key=base64.b64encode(key.public_key().public_bytes_raw()).decode(),
        signature=base64.b64encode(key.sign(att.chain.chain_hash.encode())).decode(),
    )


def test_legacy_signature_fails_closed(key: ed25519.Ed25519PrivateKey) -> None:
    att = make_attestation()
    att.signature = legacy_signature(att, key)
    assert att.signature.is_legacy
    assert "scope" not in att.to_dict()["signature"]
    assert not att.verify_signature()
    reloaded = Attestation.from_dict(att.to_dict())
    assert reloaded.signature is not None and reloaded.signature.is_legacy
    assert not reloaded.verify_signature()


@pytest.mark.parametrize("scope", ["document", "chain", "", "DOCUMENT"])
def test_legacy_signature_with_scope_set_fails(
    key: ed25519.Ed25519PrivateKey, scope: str
) -> None:
    att = make_attestation()
    document = att.to_dict()
    document["signature"] = legacy_signature(att, key).to_dict()
    document["signature"]["scope"] = scope
    assert not Attestation.from_dict(document).verify_signature()


def test_wrong_algorithm_and_malformed_encodings_fail(
    signed_document: dict[str, Any],
) -> None:
    for field, value in (
        ("algorithm", "RSA-SHA256"),
        ("public_key", "not base64!"),
        ("public_key", base64.b64encode(b"short").decode()),
        ("signature", "not base64!"),
    ):
        document = copy.deepcopy(signed_document)
        document["signature"][field] = value
        assert not Attestation.from_dict(document).verify_signature(), field


def test_unsigned_does_not_verify() -> None:
    assert not make_attestation().verify_signature()


def _signed_text(key: ed25519.Ed25519PrivateKey) -> str:
    att = make_attestation()
    att.sign(key.private_bytes_raw())
    return att.to_json()


def test_duplicate_key_is_rejected_on_load(key: ed25519.Ed25519PrivateKey) -> None:
    """A second 'verdict' that another JSON reader may show must not ride a valid signature."""
    text = _signed_text(key)
    assert Attestation.from_json(text).verify_signature()
    real = json.loads(text)["verdict"]
    forged = text.replace(
        f'"verdict": "{real}"', f'"verdict": "TAMPERED",\n  "verdict": "{real}"', 1)
    assert forged != text
    # Without strict parsing, json.loads keeps the last value and this verifies.
    assert Attestation.from_dict(json.loads(forged)).verify_signature()
    with pytest.raises(ValueError, match="duplicate key"):
        Attestation.from_json(forged)


def test_non_standard_constants_rejected_on_load(key: ed25519.Ed25519PrivateKey) -> None:
    text = _signed_text(key)
    nan_text = text.replace('"total": 2', '"total": NaN', 1)
    assert nan_text != text
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        Attestation.from_json(nan_text)


def test_non_finite_float_does_not_verify(key: ed25519.Ed25519PrivateKey) -> None:
    text = _signed_text(key)
    inf_text = text.replace('"total": 2', '"total": 1e400', 1)
    assert inf_text != text
    assert not Attestation.from_json(inf_text).verify_signature()
