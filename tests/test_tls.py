"""Tests for verification-network transport encryption helpers."""

from __future__ import annotations

import ssl
from datetime import datetime, timezone

from avir.network.tls import client_ssl_context, make_self_signed_cert


def test_make_self_signed_cert(tmp_path):
    cert = tmp_path / "c.crt"
    key = tmp_path / "c.key"
    make_self_signed_cert(cert, key, hosts=["localhost", "127.0.0.1"])
    assert cert.exists() and key.exists()
    assert (key.stat().st_mode & 0o777) == 0o600

    from cryptography import x509
    c = x509.load_pem_x509_certificate(cert.read_bytes())
    san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    import ipaddress
    assert ipaddress.ip_address("127.0.0.1") in san.get_values_for_type(x509.IPAddress)
    assert c.not_valid_after_utc > datetime.now(timezone.utc)


def test_client_ssl_context_secure_by_default():
    ctx = client_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_client_ssl_context_with_cafile_still_verifies(tmp_path):
    cert = tmp_path / "c.crt"
    key = tmp_path / "c.key"
    make_self_signed_cert(cert, key, hosts=["127.0.0.1"])
    ctx = client_ssl_context(cafile=str(cert))
    assert ctx.verify_mode == ssl.CERT_REQUIRED  # trusts our CA but still verifies


def test_client_ssl_context_insecure_disables_verification():
    ctx = client_ssl_context(insecure=True)
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False
