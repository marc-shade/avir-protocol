"""
End-to-end tests for the ``avir`` command-line tool.

Every test runs the real CLI in a subprocess (``python -m avir.cli``) on the
files in ``examples/``. ``verify`` drives the real CrossVerifier and the real
CLI providers (``claude-cli``, ``gemini-cli``), which shell out to ``claude``
and ``gemini`` executables found on PATH. The tests put small stand-in
executables first on PATH, so no real model is called: the stand-ins answer
subject prompts with a fixed text and judge prompts with a fixed VERDICT line.

Gaps / not covered:
- No real model is invoked here; the SDK providers (claude/openai/gemini) and
  their API paths are not exercised.
- The stand-in judges always agree, so split or abstaining panels are only
  covered by the library's own cross-verification tests.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
BASIC_SPEC = EXAMPLES / "basic_spec.json"
RUBRIC_SPEC = EXAMPLES / "cluster_demonstrator_spec.json"

FAKE_MODEL_CLI = """\
import os
import sys

prompt = sys.argv[-1]
name = os.path.basename(sys.argv[0])
role = "judge" if "impartial verifier" in prompt else "subject"
with open(os.environ["AVIR_TEST_MODEL_LOG"], "a") as log:
    log.write(f"{name} {role}\\n")
if role == "judge":
    print("The answer was checked against the rubric.")
    print("CONFIDENCE: 0.9")
    print("VERDICT: " + os.environ.get("AVIR_TEST_VERDICT", "PASS"))
else:
    print("The ball costs 5 cents: ball $0.05, bat $1.05, total $1.10.")
"""


def run_cli(
    *args: str | Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "avir.cli", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=120,
        check=False,
    )


def assert_no_traceback(proc: subprocess.CompletedProcess[str]) -> None:
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stdout, proc.stdout


def write_keypair(directory: Path, name: str) -> tuple[Path, Path, bytes]:
    """Write PEM private/public key files; return their paths and raw public key."""
    sk = ed25519.Ed25519PrivateKey.generate()
    priv = directory / f"{name}.pem"
    pub = directory / f"{name}_pub.pem"
    priv.write_bytes(
        sk.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    pub.write_bytes(
        sk.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return priv, pub, sk.public_key().public_bytes_raw()


@pytest.fixture
def model_env(tmp_path: Path) -> dict[str, str]:
    """Environment whose PATH resolves ``claude`` and ``gemini`` to stand-ins."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("claude", "gemini"):
        exe = bindir / name
        exe.write_text(f"#!{sys.executable}\n{FAKE_MODEL_CLI}")
        exe.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["AVIR_TEST_MODEL_LOG"] = str(tmp_path / "model_calls.log")
    return env


def model_calls(env: dict[str, str]) -> list[str]:
    log = Path(env["AVIR_TEST_MODEL_LOG"])
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def signed_attestation(
    tmp_path: Path, model_env: dict[str, str]
) -> dict[str, Any]:
    """Run ``avir verify`` with signing; return the paths and the raw signer key."""
    priv, pub, raw_pub = write_keypair(tmp_path, "signer")
    att = tmp_path / "att.json"
    proc = run_cli(
        "verify", RUBRIC_SPEC,
        "--provider", "claude-cli", "--provider", "gemini-cli",
        "--attestation", att, "--sign", "--key-file", priv,
        env=model_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {"path": att, "pub": pub, "raw_pub": raw_pub, "proc": proc}


# ---------------------------------------------------------------------------
# general
# ---------------------------------------------------------------------------

def test_version_and_help() -> None:
    proc = run_cli("--version")
    assert proc.returncode == 0
    assert proc.stdout.startswith("AVIR ")
    proc = run_cli()
    assert proc.returncode == 0
    for command in ("verify", "validate", "check", "init", "compare"):
        assert command in proc.stdout


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [BASIC_SPEC, RUBRIC_SPEC])
def test_validate_valid_examples(spec: Path) -> None:
    proc = run_cli("validate", spec)
    assert proc.returncode == 0, proc.stderr
    assert "Status: VALID" in proc.stdout
    assert "AVIR version: 1.0.0" in proc.stdout


def test_validate_reports_spec_errors(tmp_path: Path) -> None:
    data = json.loads(BASIC_SPEC.read_text())
    data["benchmarks"][0]["tolerance"] = 2.0
    data["capabilities"][0]["benchmarks"].append("no_such_benchmark")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    proc = run_cli("validate", bad)
    assert proc.returncode == 1
    assert "Status: INVALID" in proc.stderr
    assert "invalid tolerance: 2.0" in proc.stderr
    assert "unknown benchmark 'no_such_benchmark'" in proc.stderr
    assert_no_traceback(proc)


@pytest.mark.parametrize(
    "content, expected",
    [
        ("{not json", "JSONDecodeError"),
        ('{"avir_version": "1.0.0"}', "Missing required fields"),
        (
            '{"avir_version": "1", "system": {"version": "1"},'
            ' "capabilities": [], "benchmarks": []}',
            "missing field 'name'",
        ),
    ],
)
def test_validate_unloadable_spec(tmp_path: Path, content: str, expected: str) -> None:
    path = tmp_path / "spec.json"
    path.write_text(content)
    proc = run_cli("validate", path)
    assert proc.returncode == 1
    assert expected in proc.stderr
    assert proc.stdout == ""
    assert_no_traceback(proc)


def test_validate_missing_file(tmp_path: Path) -> None:
    proc = run_cli("validate", tmp_path / "absent.json")
    assert proc.returncode == 1
    assert "not found" in proc.stderr
    assert_no_traceback(proc)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", ["minimal", "standard", "comprehensive"])
def test_init_templates_pass_validate(tmp_path: Path, template: str) -> None:
    out = tmp_path / f"{template}.json"
    proc = run_cli("init", "my-system", "--template", template, "--output", out)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["system"]["name"] == "my-system"
    proc = run_cli("validate", out)
    assert proc.returncode == 0, proc.stderr
    assert "Status: VALID" in proc.stdout


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "spec.json"
    out.write_text("keep me")
    proc = run_cli("init", "x", "--output", out)
    assert proc.returncode == 1
    assert "already exists" in proc.stderr
    assert out.read_text() == "keep me"


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def test_verify_rubric_pass(model_env: dict[str, str]) -> None:
    proc = run_cli(
        "verify", RUBRIC_SPEC, "--provider", "claude-cli", "--provider", "gemini-cli",
        env=model_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Verdict: VERIFIED" in proc.stdout
    assert "claude → gemini: VERIFIED" in proc.stdout
    assert "gemini → claude: VERIFIED" in proc.stdout
    calls = model_calls(model_env)
    # Each provider answers once and judges both answers (self cell recorded,
    # excluded from consensus): 2 subject + 4 judge calls.
    assert sorted(calls) == sorted(
        ["claude subject", "gemini subject"] + ["claude judge", "gemini judge"] * 2
    )


def test_verify_rubric_fail(model_env: dict[str, str]) -> None:
    model_env["AVIR_TEST_VERDICT"] = "FAIL"
    proc = run_cli(
        "verify", RUBRIC_SPEC, "--provider", "claude-cli", "--provider", "gemini-cli",
        "--json", env=model_env,
    )
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["consensus"]["verdict"] == "FAILED"
    assert result["consensus"]["quorum_met"] is True
    assert result["consensus"]["pass_rate"] == 0.0


def test_verify_numeric_spec_with_cli_providers_is_invalid(
    model_env: dict[str, str],
) -> None:
    # CLI providers abstain on numeric benchmarks (they cannot measure a
    # system), so quorum is not met and the run must not report success.
    proc = run_cli(
        "verify", BASIC_SPEC, "--provider", "claude-cli", "--provider", "gemini-cli",
        "--json", "--iterations", "2", env=model_env,
    )
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["consensus"]["verdict"] == "INVALID"
    assert result["consensus"]["quorum_met"] is False
    assert model_calls(model_env) == []


def test_verify_writes_output_file(tmp_path: Path, model_env: dict[str, str]) -> None:
    out = tmp_path / "result.json"
    proc = run_cli(
        "verify", RUBRIC_SPEC, "--provider", "claude-cli", "--provider", "gemini-cli",
        "--json", "--output", out, env=model_env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    result = json.loads(out.read_text())
    assert result["consensus"]["verdict"] == "VERIFIED"
    assert len(result["matrix"]["cells"]) == 4


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--provider", "claude-cli"], "at least two --provider"),
        (["--provider", "claude-cli", "--provider", "claude"], "distinct ids"),
        (["--provider", "claude-cli", "--provider", "nope"], "Unknown provider: nope"),
        (["--provider", "claude-cli", "--provider", "gemini-cli=",], "empty model"),
        (
            ["--provider", "claude-cli", "--provider", "gemini-cli", "--sign"],
            "--sign requires --key-file and --attestation",
        ),
        (
            ["--provider", "claude-cli", "--provider", "gemini-cli", "--iterations", "0"],
            "--iterations must be at least 1",
        ),
    ],
)
def test_verify_argument_errors(
    model_env: dict[str, str], extra: list[str], message: str
) -> None:
    proc = run_cli("verify", RUBRIC_SPEC, *extra, env=model_env)
    assert proc.returncode == 1
    assert message in proc.stderr
    assert_no_traceback(proc)
    assert model_calls(model_env) == []


def test_verify_bad_key_fails_before_running_models(
    tmp_path: Path, model_env: dict[str, str]
) -> None:
    key = tmp_path / "key.pem"
    key.write_text("not a key")
    proc = run_cli(
        "verify", RUBRIC_SPEC, "--provider", "claude-cli", "--provider", "gemini-cli",
        "--attestation", tmp_path / "att.json", "--sign", "--key-file", key,
        env=model_env,
    )
    assert proc.returncode == 1
    assert "unrecognized private key format" in proc.stderr
    assert model_calls(model_env) == []
    assert not (tmp_path / "att.json").exists()


def test_verify_invalid_spec(tmp_path: Path, model_env: dict[str, str]) -> None:
    data = json.loads(RUBRIC_SPEC.read_text())
    data["benchmarks"][0]["runs"] = 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    proc = run_cli(
        "verify", bad, "--provider", "claude-cli", "--provider", "gemini-cli",
        env=model_env,
    )
    assert proc.returncode == 1
    assert "invalid runs: 0" in proc.stderr
    assert_no_traceback(proc)
    assert model_calls(model_env) == []


# ---------------------------------------------------------------------------
# verify --sign + check round trip
# ---------------------------------------------------------------------------

def test_signed_attestation_contents(signed_attestation: dict[str, Any]) -> None:
    data = json.loads(signed_attestation["path"].read_text())
    assert data["verdict"] == "VERIFIED"
    assert data["verification_level"] == "L3"
    assert data["signature"]["algorithm"] == "Ed25519"
    assert data["signature"]["scope"] == "document"
    assert base64.b64decode(data["signature"]["public_key"]) == signed_attestation["raw_pub"]
    cells = {b["cell"] for b in data["results"]["benchmarks"]}
    assert cells == {"claude->gemini", "gemini->claude"}
    assert "Attestation signed: Ed25519 key" in signed_attestation["proc"].stderr


def _rechain(data: dict[str, Any]) -> None:
    """Recompute chain_hash so chain integrity passes and only the signature
    can catch the change."""
    import hashlib

    chain = data["attestation_chain"]
    chain["chain_hash"] = hashlib.sha256(
        f"{chain['spec_hash']}{chain['env_hash']}{chain['results_hash']}"
        f"{chain['timestamp']}".encode()
    ).hexdigest()


def _tamper(data: dict[str, Any], section: str) -> None:
    if section == "verdict":
        data["verdict"] = "FAILED"
    elif section == "results.benchmarks":
        data["results"]["benchmarks"][0]["verdict"] = "FAIL"
    elif section == "summary":
        data["results"]["summary"]["passed"] += 1
    elif section == "system":
        data["system"]["version"] = "9.9.9"
    elif section == "verifier":
        data["verifier"]["model"] = "someone-else"
    elif section == "environment":
        data["environment"]["os"] = "plan9"
    elif section == "execution":
        data["execution"]["duration_seconds"] += 60
    elif section == "verification_level":
        data["verification_level"] = "L4"
    elif section == "attestation_chain":
        data["attestation_chain"]["timestamp"] = "2000-01-01T00:00:00Z"
        _rechain(data)
    else:
        raise AssertionError(section)


TAMPER_SECTIONS = [
    "verdict", "results.benchmarks", "summary", "system", "verifier",
    "environment", "execution", "verification_level", "attestation_chain",
]


@pytest.mark.parametrize("section", TAMPER_SECTIONS)
def test_check_tampered_section_fails(
    tmp_path: Path, signed_attestation: dict[str, Any], section: str
) -> None:
    data = json.loads(signed_attestation["path"].read_text())
    _tamper(data, section)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data))
    for extra in ([], ["--public-key", str(signed_attestation["pub"])]):
        proc = run_cli("check", path, *extra)
        assert proc.returncode == 1, (section, extra, proc.stdout)
        assert "Chain integrity: VALID" in proc.stdout
        assert "Signature: INVALID" in proc.stdout
        assert "Signature: VALID" not in proc.stdout


def _legacy_file(tmp_path: Path, signed: dict[str, Any], scope: str | None) -> Path:
    """Re-sign the saved attestation in the legacy format (chain hash only)."""
    sk = ed25519.Ed25519PrivateKey.generate()
    data = json.loads(signed["path"].read_text())
    data["signature"] = {
        "algorithm": "Ed25519",
        "public_key": base64.b64encode(sk.public_key().public_bytes_raw()).decode(),
        "signature": base64.b64encode(
            sk.sign(data["attestation_chain"]["chain_hash"].encode())
        ).decode(),
    }
    if scope is not None:
        data["signature"]["scope"] = scope
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    return path


def test_check_legacy_signature_fails(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    path = _legacy_file(tmp_path, signed_attestation, scope=None)
    proc = run_cli("check", path)
    assert proc.returncode == 1
    assert "Signature: LEGACY (content not authenticated)" in proc.stdout
    assert "legacy signature" in proc.stderr
    assert "does not authenticate the attestation content" in proc.stderr
    assert "Signature: VALID" not in proc.stdout

    proc = run_cli("compare", signed_attestation["path"], path)
    assert proc.returncode == 1
    assert "LEGACY" in proc.stdout


def test_check_legacy_signature_with_scope_flipped_fails(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    path = _legacy_file(tmp_path, signed_attestation, scope="document")
    proc = run_cli("check", path)
    assert proc.returncode == 1
    assert "Signature: INVALID" in proc.stdout
    assert "Signature: VALID" not in proc.stdout


def test_check_unknown_scope_fails(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    path = _legacy_file(tmp_path, signed_attestation, scope="chain")
    proc = run_cli("check", path)
    assert proc.returncode == 1
    assert "Signature: UNSUPPORTED" in proc.stdout
    assert "unsupported signature scope 'chain'" in proc.stderr


def test_check_pinned_key_passes(signed_attestation: dict[str, Any]) -> None:
    proc = run_cli(
        "check", signed_attestation["path"], "--public-key", signed_attestation["pub"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Chain integrity: VALID" in proc.stdout
    assert "Signature: VALID (pinned key" in proc.stdout


@pytest.mark.parametrize("encoding", ["hex", "base64"])
def test_check_pinned_key_raw_encodings(
    tmp_path: Path, signed_attestation: dict[str, Any], encoding: str
) -> None:
    raw = signed_attestation["raw_pub"]
    key = tmp_path / "pub.txt"
    key.write_text(raw.hex() if encoding == "hex" else base64.b64encode(raw).decode())
    proc = run_cli("check", signed_attestation["path"], "--public-key", key)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Signature: VALID (pinned key" in proc.stdout


def test_check_wrong_pinned_key_fails(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    _, other_pub, _ = write_keypair(tmp_path, "other")
    proc = run_cli("check", signed_attestation["path"], "--public-key", other_pub)
    assert proc.returncode == 1
    assert "Signature: INVALID" in proc.stdout
    assert "Signature: VALID" not in proc.stdout
    assert "not signed by the pinned public key" in proc.stderr


def test_check_resigned_by_other_key_fails_against_pinned_key(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    """A self-consistent attestation re-signed by another key must not pass
    when the original signer's key is pinned."""
    from avir.attestation import Attestation

    other_priv = ed25519.Ed25519PrivateKey.generate()
    forged = Attestation.load(str(signed_attestation["path"]))
    forged.sign(other_priv.private_bytes_raw())
    assert forged.verify_signature()  # valid under its own embedded key
    forged_path = tmp_path / "forged.json"
    forged.save(str(forged_path))

    # Unpinned: the signature is self-consistent, reported as self-asserted.
    proc = run_cli("check", forged_path)
    assert proc.returncode == 0
    assert "self-asserted, NOT pinned" in proc.stdout
    assert "Pass --public-key" in proc.stderr

    # Pinned to the original signer: must fail.
    proc = run_cli("check", forged_path, "--public-key", signed_attestation["pub"])
    assert proc.returncode == 1
    assert "Signature: INVALID" in proc.stdout
    assert "Signature: VALID" not in proc.stdout


def test_check_tampered_signature_fails(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    data = json.loads(signed_attestation["path"].read_text())
    sig = bytearray(base64.b64decode(data["signature"]["signature"]))
    sig[0] ^= 0x01
    data["signature"]["signature"] = base64.b64encode(bytes(sig)).decode()
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data))

    for extra in ([], ["--public-key", str(signed_attestation["pub"])]):
        proc = run_cli("check", path, *extra)
        assert proc.returncode == 1, extra
        assert "Signature: INVALID" in proc.stdout


def test_check_tampered_chain_fails(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    data = json.loads(signed_attestation["path"].read_text())
    data["attestation_chain"]["results_hash"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data))
    proc = run_cli("check", path, "--public-key", signed_attestation["pub"])
    assert proc.returncode == 1
    assert "Chain integrity: INVALID" in proc.stdout


def test_check_unsigned_fails(tmp_path: Path, signed_attestation: dict[str, Any]) -> None:
    data = json.loads(signed_attestation["path"].read_text())
    data["signature"] = None
    path = tmp_path / "unsigned.json"
    path.write_text(json.dumps(data))
    for extra in ([], ["--public-key", str(signed_attestation["pub"])]):
        proc = run_cli("check", path, *extra)
        assert proc.returncode == 1, extra
        assert "Signature: MISSING" in proc.stdout
        assert "unsigned" in proc.stderr


@pytest.mark.parametrize("content", ["{}", "not json", ""])
def test_check_unloadable_attestation(tmp_path: Path, content: str) -> None:
    path = tmp_path / "att.json"
    path.write_text(content)
    proc = run_cli("check", path)
    assert proc.returncode == 1
    assert "cannot load attestation" in proc.stderr
    assert_no_traceback(proc)


def test_check_bad_public_key_file(
    tmp_path: Path, signed_attestation: dict[str, Any]
) -> None:
    key = tmp_path / "key.txt"
    key.write_text("abc")
    proc = run_cli("check", signed_attestation["path"], "--public-key", key)
    assert proc.returncode == 1
    assert "unrecognized public key format" in proc.stderr
    assert_no_traceback(proc)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare(tmp_path: Path, signed_attestation: dict[str, Any]) -> None:
    first = signed_attestation["path"]
    second = tmp_path / "second.json"
    second.write_text(first.read_text())

    proc = run_cli("compare", first, second)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Consensus: YES" in proc.stdout

    from avir.attestation import Attestation

    # A genuinely signed attestation with a different verdict: disagreement.
    other = Attestation.load(str(first))
    other.verdict = "FAILED"
    other.sign(ed25519.Ed25519PrivateKey.generate().private_bytes_raw())
    disagree = tmp_path / "disagree.json"
    other.save(str(disagree))
    proc = run_cli("compare", first, disagree)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Consensus: NO" in proc.stdout

    # An edited verdict without re-signing is a signature failure.
    data = json.loads(first.read_text())
    data["verdict"] = "FAILED"
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(data))
    proc = run_cli("compare", first, edited)
    assert proc.returncode == 1
    assert "signature: INVALID" in proc.stdout

    data["attestation_chain"]["spec_hash"] = "0" * 64
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data))
    proc = run_cli("compare", first, broken)
    assert proc.returncode == 1
    assert "chain integrity: INVALID" in proc.stdout

    proc = run_cli("compare", first)
    assert proc.returncode == 1
    assert "at least 2" in proc.stderr
