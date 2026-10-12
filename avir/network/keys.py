"""
Ed25519 identity for verification-network participants.

Every participant (verifier or requester) has a long-lived Ed25519 keypair. The
public key is the participant's network identity; the short fingerprint is its
handle in the registry. Private keys never leave the operator's machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ed25519


def generate_keypair() -> tuple[bytes, bytes]:
    """Return (private_key_raw, public_key_raw) as 32-byte values."""
    sk = ed25519.Ed25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


def load_or_create_identity(path: str | Path) -> tuple[bytes, bytes]:
    """Load a hex-encoded private key from ``path``, creating one if absent.

    The key file is written with 0600 permissions. Returns (priv_raw, pub_raw).
    """
    path = Path(path)
    if path.exists():
        priv = bytes.fromhex(path.read_text().strip())
        pub = ed25519.Ed25519PrivateKey.from_private_bytes(priv).public_key().public_bytes_raw()
        return priv, pub
    priv, pub = generate_keypair()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(priv.hex())
    path.chmod(0o600)
    return priv, pub


def is_valid_pubkey(pub_hex: str) -> bool:
    """True if ``pub_hex`` is a hex-encoded 32-byte Ed25519 public key."""
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        return True
    except ValueError:
        return False


def fingerprint(pubkey: bytes | str) -> str:
    """Stable short handle for a public key (first 16 hex chars of SHA-256)."""
    raw = bytes.fromhex(pubkey) if isinstance(pubkey, str) else pubkey
    return hashlib.sha256(raw).hexdigest()[:16]


def canonical(payload: Any) -> bytes:
    """Deterministic JSON bytes for signing (sorted keys, no whitespace)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign(priv: bytes, message: bytes) -> str:
    """Ed25519-sign ``message``; return hex signature."""
    sk = ed25519.Ed25519PrivateKey.from_private_bytes(priv)
    sig_hex: str = sk.sign(message).hex()
    return sig_hex


def verify(pub_hex: str, message: bytes, sig_hex: str) -> bool:
    """Verify a hex Ed25519 signature over ``message``. False on any failure."""
    try:
        pk = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pk.verify(bytes.fromhex(sig_hex), message)
        return True
    except Exception:  # noqa: BLE001 - any failure is a non-verification
        return False
