"""
Transport encryption for the verification network.

Everything between clients and the coordinator travels over HTTPS/TLS. This
module provides:
- ``make_self_signed_cert`` -- generate a cert+key for a coordinator (turnkey
  encryption for LAN / Tailscale / private deployments).
- ``client_ssl_context`` / ``default_client_context`` -- build the SSL context
  clients use, honoring a custom CA bundle (for self-signed coordinators) via
  the ``AVIR_CA_BUNDLE`` env var, or ``AVIR_INSECURE=1`` for dev only.

What TLS does and does not cover (honest scope):
- Covers: confidentiality + integrity of all data in transit (jobs, responses,
  verdicts, the registry).
- Does NOT hide job content from the coordinator: the coordinator is a relay and
  verifiers must read the prompt/response to judge them, and the set of
  verifiers is unknown at submission time, so end-to-end encryption to specific
  verifiers is not possible. Verdicts are additionally Ed25519-signed for
  integrity/authenticity independent of TLS.
"""

from __future__ import annotations

import ipaddress
import os
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence


def make_self_signed_cert(
    cert_path: str | Path,
    key_path: str | Path,
    *,
    hosts: Sequence[str] = ("localhost", "127.0.0.1"),
    days: int = 825,
) -> tuple[str, str]:
    """Write a self-signed cert + EC private key covering ``hosts`` (SAN).

    Suitable for a private/LAN/Tailscale coordinator. Clients then trust it via
    AVIR_CA_BUNDLE=<cert_path>. The key is written 0600.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    if not hosts:
        raise ValueError("at least one host is required")

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, str(hosts[0]))])

    san: list[x509.GeneralName] = []
    for h in hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san.append(x509.DNSName(h))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path, key_path = Path(cert_path), Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)


def client_ssl_context(
    *,
    cafile: str | None = None,
    insecure: bool = False,
) -> ssl.SSLContext:
    """Build a client SSL context. ``cafile`` adds a CA for self-signed certs;
    ``insecure`` disables verification (dev only -- prints nothing, caller warns).
    """
    ctx = ssl.create_default_context()
    if cafile:
        ctx.load_verify_locations(cafile)
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def default_client_context() -> ssl.SSLContext:
    """Client SSL context from environment.

    The SECURE way to trust a self-signed coordinator is AVIR_CA_BUNDLE=<cert>
    (real verification against that CA). AVIR_INSECURE=1 disables verification
    entirely -- it enables MITM attacks and is for throwaway local dev ONLY; it
    emits a loud warning so it can never be silent.
    """
    import sys

    cafile = os.environ.get("AVIR_CA_BUNDLE") or None
    insecure = os.environ.get("AVIR_INSECURE", "").lower() in ("1", "true", "yes")
    if insecure:
        print(
            "SECURITY WARNING: AVIR_INSECURE disables TLS certificate verification "
            "(vulnerable to MITM). Use AVIR_CA_BUNDLE=<coordinator cert> instead.",
            file=sys.stderr,
        )
    return client_ssl_context(cafile=cafile, insecure=insecure)


def main() -> int:
    """CLI: generate a self-signed coordinator certificate."""
    import argparse

    ap = argparse.ArgumentParser(description="Generate a self-signed TLS cert for an AVIR coordinator.")
    ap.add_argument("--cert", default=str(Path(os.path.expanduser("~/.avir/coordinator.crt"))))
    ap.add_argument("--key", default=str(Path(os.path.expanduser("~/.avir/coordinator.key"))))
    ap.add_argument("--host", action="append", default=[],
                    help="hostname or IP to include in SAN (repeatable). Default: localhost,127.0.0.1")
    ap.add_argument("--days", type=int, default=825)
    args = ap.parse_args()
    hosts = args.host or ["localhost", "127.0.0.1"]
    cert, key = make_self_signed_cert(args.cert, args.key, hosts=hosts, days=args.days)
    print(f"wrote cert={cert}")
    print(f"wrote key={key} (0600)")
    print(f"SAN hosts: {', '.join(hosts)}")
    print(f"clients: set AVIR_CA_BUNDLE={cert} (or --cacert) to trust this coordinator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
