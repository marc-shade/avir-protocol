"""
Tests for the AVIR verification network: identity, protocol, coordinator.

These run with deterministic test keypairs and fixed verdicts (no real models),
proving the integrity properties that matter: verdicts are signed and verified,
forged/tampered verdicts are rejected, consensus respects quorum and distinct
keys, and the requester's own key is excluded.
"""

from __future__ import annotations

import pytest

from avir.network import keys
from avir.network.protocol import (
    aggregate_consensus,
    sign_verdict,
    verify_verdict,
)

# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------

def test_keypair_sign_verify_roundtrip():
    priv, pub = keys.generate_keypair()
    msg = b"hello avir"
    sig = keys.sign(priv, msg)
    assert keys.verify(pub.hex(), msg, sig)
    assert not keys.verify(pub.hex(), b"tampered", sig)


def test_fingerprint_stable_and_short():
    _, pub = keys.generate_keypair()
    fp = keys.fingerprint(pub)
    assert fp == keys.fingerprint(pub.hex())
    assert len(fp) == 16


def test_identity_persisted_0600(tmp_path):
    p = tmp_path / "id"
    priv1, pub1 = keys.load_or_create_identity(p)
    priv2, pub2 = keys.load_or_create_identity(p)
    assert priv1 == priv2 and pub1 == pub2
    assert (p.stat().st_mode & 0o777) == 0o600


# --------------------------------------------------------------------------
# protocol: verdict signing + consensus
# --------------------------------------------------------------------------

def _verdicts(pass_=True):
    return [{"benchmark_id": "q1", "verdict": "PASS" if pass_ else "FAIL", "confidence": 0.9}]


def test_verdict_signature_roundtrip_and_tamper():
    priv, pub = keys.generate_keypair()
    v = _verdicts()
    sig = sign_verdict(priv, "job1", v)
    assert verify_verdict(pub.hex(), "job1", v, sig)
    # tamper the verdict -> signature no longer valid
    assert not verify_verdict(pub.hex(), "job1", _verdicts(pass_=False), sig)
    # wrong job id -> invalid
    assert not verify_verdict(pub.hex(), "job2", v, sig)


def _record(pub, org, verdict):
    return {"verifier_pubkey": pub, "verifier_org": org,
            "verdicts": [{"benchmark_id": "q1", "verdict": verdict}]}


def test_consensus_requires_quorum():
    recs = [_record("k1", "a", "PASS"), _record("k2", "b", "PASS")]
    c = aggregate_consensus(recs, min_verifiers=3)
    assert c["overall_verdict"] == "INCONCLUSIVE"
    assert c["quorum_met"] is False
    assert c["distinct_verifiers"] == 2


def test_consensus_majority_pass_verified():
    recs = [_record("k1", "a", "PASS"), _record("k2", "b", "PASS"), _record("k3", "c", "PASS")]
    c = aggregate_consensus(recs, min_verifiers=3)
    assert c["overall_verdict"] == "VERIFIED"
    assert c["benchmarks"]["q1"]["verdict"] == "PASS"
    assert c["agreement_score"] == 1.0


def test_consensus_excludes_requester_key():
    # requester's own key should not count toward quorum
    recs = [_record("req", "a", "PASS"), _record("k2", "b", "PASS"), _record("k3", "c", "PASS")]
    c = aggregate_consensus(recs, min_verifiers=3, exclude_pubkeys=("req",))
    assert c["distinct_verifiers"] == 2
    assert c["quorum_met"] is False


def test_consensus_distinct_orgs_required():
    # 3 keys but only 2 orgs -> distinct-org quorum of 3 not met
    recs = [_record("k1", "a", "PASS"), _record("k2", "a", "PASS"), _record("k3", "b", "PASS")]
    c = aggregate_consensus(recs, min_verifiers=3, require_distinct_orgs=True)
    assert c["distinct_orgs"] == 2
    assert c["quorum_met"] is False


def test_consensus_abstain_only_inconclusive():
    recs = [_record("k1", "a", "ABSTAIN"), _record("k2", "b", "ABSTAIN"), _record("k3", "c", "ABSTAIN")]
    c = aggregate_consensus(recs, min_verifiers=3)
    assert c["overall_verdict"] == "INCONCLUSIVE"
    assert c["quorum_met"] is False


def test_consensus_abstentions_do_not_count_toward_quorum():
    # 4 keys submitted, but only 2 judged: the 2026-10-10 fleet job that reported
    # quorum_met=True with half the verifiers abstaining.
    recs = [_record("k1", "a", "PASS"), _record("k2", "b", "PASS"),
            _record("k3", "c", "ABSTAIN"), _record("k4", "d", "ABSTAIN")]
    c = aggregate_consensus(recs, min_verifiers=3)
    assert c["distinct_verifiers"] == 4
    assert c["quorum_met"] is False
    assert c["overall_verdict"] == "INCONCLUSIVE"
    assert c["benchmarks"]["q1"]["quorum_met"] is False
    # one more decisive vote reaches quorum
    c = aggregate_consensus(recs + [_record("k5", "e", "PASS")], min_verifiers=3)
    assert c["quorum_met"] is True
    assert c["overall_verdict"] == "VERIFIED"


def test_consensus_quorum_is_per_benchmark():
    # q1 has 3 decisive votes, q2 only 2: the job as a whole is below quorum.
    def rec(pub, org, v1, v2):
        return {"verifier_pubkey": pub, "verifier_org": org,
                "verdicts": [{"benchmark_id": "q1", "verdict": v1},
                             {"benchmark_id": "q2", "verdict": v2}]}
    recs = [rec("k1", "a", "PASS", "PASS"), rec("k2", "b", "PASS", "FAIL"),
            rec("k3", "c", "PASS", "ABSTAIN")]
    c = aggregate_consensus(recs, min_verifiers=3)
    assert c["benchmarks"]["q1"]["quorum_met"] is True
    assert c["benchmarks"]["q2"]["quorum_met"] is False
    assert c["quorum_met"] is False


def test_consensus_distinct_orgs_counts_only_decisive_orgs():
    # 3 orgs submitted, but org c abstained: 2 decisive orgs < 3.
    recs = [_record("k1", "a", "PASS"), _record("k2", "b", "PASS"),
            _record("k3", "c", "ABSTAIN"), _record("k4", "a", "PASS")]
    c = aggregate_consensus(recs, min_verifiers=3, require_distinct_orgs=True)
    assert c["distinct_orgs"] == 3
    assert c["benchmarks"]["q1"]["decisive_orgs"] == 2
    assert c["quorum_met"] is False


def test_verifier_logs_abstain_reason(capsys):
    """A judge call that fails must leave its cause in the verifier log, not just ABSTAIN."""
    import asyncio

    from avir.network.verifier import _judge_job
    from avir.providers.base import VerdictResponse

    class FailingJudge:
        async def judge_output(self, **_kw):
            return VerdictResponse(value=None, verdict="ABSTAIN",
                                   abstain_reason="http:m judge call failed: HTTP Error 503")

    job = {"benchmarks": [{"benchmark_id": "q1", "prompt": "p", "rubric": "r"}],
           "subject_responses": {"q1": ["answer"]}}
    verdicts = asyncio.run(_judge_job(FailingJudge(), job))
    assert verdicts == [{"benchmark_id": "q1", "verdict": "ABSTAIN", "confidence": None}]
    assert "ABSTAIN on q1: http:m judge call failed: HTTP Error 503" in capsys.readouterr().err


# --------------------------------------------------------------------------
# coordinator end-to-end (FastAPI TestClient)
# --------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from avir.network.coordinator import create_app
    app = create_app(str(tmp_path / "coord.db"))
    return TestClient(app)


def _spec():
    return {
        "avir_version": "1.0.0",
        "system": {"name": "net-test", "version": "1.0.0"},
        "capabilities": [],
        "benchmarks": [{"id": "q1", "kind": "rubric", "prompt": "p", "rubric": "r"}],
    }


def test_coordinator_full_lifecycle(client):
    # three independent verifiers register
    parties = {}
    orgs = {"alice": "org-a", "bob": "org-b", "carol": "org-c"}
    for name, org in orgs.items():
        priv, pub = keys.generate_keypair()
        parties[name] = (priv, pub.hex())
        r = client.post("/verifiers/register", json={
            "pubkey": pub.hex(), "label": name, "org": org, "models": ["m"]})
        assert r.status_code == 200

    # registry lists all participants
    vs = client.get("/verifiers").json()
    assert vs["count"] == 3

    # requester submits a job
    job_id = client.post("/jobs", json={
        "spec": _spec(),
        "subject_responses": {"q1": ["the answer is 5 cents"]},
        "min_verifiers": 3,
    }).json()["job_id"]

    # each verifier pulls the (blinded) job and submits a signed PASS
    for _name, (priv, pub_hex) in parties.items():
        nxt = client.get(f"/jobs/next?pubkey={pub_hex}").json()["job"]
        assert nxt is not None and nxt["job_id"] == job_id
        assert "requester_pubkey" not in nxt  # blinded
        verdicts = [{"benchmark_id": "q1", "verdict": "PASS", "confidence": 0.9}]
        sig = sign_verdict(priv, job_id, verdicts)
        # the verifier client sends the org it registered with
        rr = client.post(f"/jobs/{job_id}/verdicts", json={
            "verifier_pubkey": pub_hex, "verifier_org": orgs[_name], "verifier_model": "m",
            "verdicts": verdicts, "signature": sig})
        assert rr.status_code == 200

    # result: quorum met, VERIFIED, job closed
    res = client.get(f"/jobs/{job_id}").json()
    assert res["status"] == "complete"
    assert res["consensus"]["overall_verdict"] == "VERIFIED"
    assert res["consensus"]["distinct_verifiers"] == 3
    assert res["consensus"]["distinct_orgs"] == 3


def test_coordinator_rejects_forged_signature(client):
    priv, pub = keys.generate_keypair()
    _, other_pub = keys.generate_keypair()
    job_id = client.post("/jobs", json={
        "spec": _spec(), "subject_responses": {"q1": ["x"]}, "min_verifiers": 1}).json()["job_id"]
    verdicts = [{"benchmark_id": "q1", "verdict": "PASS", "confidence": 1.0}]
    sig = sign_verdict(priv, job_id, verdicts)
    # claim a DIFFERENT pubkey than the one that signed -> must be rejected
    r = client.post(f"/jobs/{job_id}/verdicts", json={
        "verifier_pubkey": other_pub.hex(), "verifier_org": "x", "verifier_model": "m",
        "verdicts": verdicts, "signature": sig})
    assert r.status_code == 400


def test_coordinator_rejects_duplicate_verdict(client):
    priv, pub = keys.generate_keypair()
    assert _register(client, pub.hex(), "x").status_code == 200
    job_id = client.post("/jobs", json={
        "spec": _spec(), "subject_responses": {"q1": ["x"]}, "min_verifiers": 1}).json()["job_id"]
    verdicts = [{"benchmark_id": "q1", "verdict": "PASS", "confidence": 1.0}]
    sig = sign_verdict(priv, job_id, verdicts)
    body = {"verifier_pubkey": pub.hex(), "verifier_org": "x", "verifier_model": "m",
            "verdicts": verdicts, "signature": sig}
    assert client.post(f"/jobs/{job_id}/verdicts", json=body).status_code == 200
    assert client.post(f"/jobs/{job_id}/verdicts", json=body).status_code == 409


def test_verifier_registration_waits_for_unreachable_coordinator(monkeypatch):
    """At boot the tunnel/coordinator may be down: a daemon retries, --once fails fast."""
    import urllib.error

    from avir.network import verifier as vmod
    calls = []

    def fake_post(base, path, body, timeout=30.0, headers=None):
        calls.append(path)
        if len(calls) < 3:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        return {"ok": True}

    monkeypatch.setattr(vmod, "_post", fake_post)
    monkeypatch.setattr(vmod.time, "sleep", lambda s: None)
    vmod._register("https://x", {"pubkey": "p"}, retry_interval=5.0)
    assert calls == ["/verifiers/register"] * 3

    calls.clear()
    with pytest.raises(urllib.error.URLError):
        vmod._register("https://x", {"pubkey": "p"}, retry_interval=None)
    assert len(calls) == 1

    def refused(base, path, body, timeout=30.0, headers=None):
        raise urllib.error.HTTPError(base + path, 400, "bad", {}, None)
    monkeypatch.setattr(vmod, "_post", refused)
    with pytest.raises(urllib.error.HTTPError):
        vmod._register("https://x", {"pubkey": "p"}, retry_interval=5.0)


def _age_job(db_path, job_id, seconds):
    import sqlite3
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with sqlite3.connect(db_path) as c:
        c.execute("UPDATE jobs SET created_at=? WHERE job_id=?", (old, job_id))


def test_job_below_quorum_expires_after_ttl(tmp_path):
    """A job stuck below decisive quorum is expired, not served or accepted forever."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from avir.network.coordinator import create_app
    db = str(tmp_path / "c.db")
    tc = TestClient(create_app(db, job_ttl_s=60))
    priv, pub = keys.generate_keypair()
    _, other = keys.generate_keypair()
    priv2, pub2 = keys.generate_keypair()
    for key, org in ((pub, "a"), (other, "c"), (pub2, "b")):
        assert _register(tc, key.hex(), org).status_code == 200
    job_id = tc.post("/jobs", json={"spec": _spec(), "subject_responses": {"q1": ["x"]},
                                    "min_verifiers": 2}).json()["job_id"]
    # one verifier abstains: below quorum, still open and still served to others
    verdicts = [{"benchmark_id": "q1", "verdict": "ABSTAIN", "confidence": None}]
    assert tc.post(f"/jobs/{job_id}/verdicts", json={
        "verifier_pubkey": pub.hex(), "verifier_org": "a", "verifier_model": "m",
        "verdicts": verdicts, "signature": sign_verdict(priv, job_id, verdicts)}).json()["quorum_met"] is False
    assert tc.get(f"/jobs/{job_id}").json()["status"] == "open"
    assert tc.get(f"/jobs/next?pubkey={other.hex()}").json()["job"]["job_id"] == job_id

    _age_job(db, job_id, 61)
    assert tc.get(f"/jobs/{job_id}").json()["status"] == "expired"
    assert tc.get(f"/jobs/next?pubkey={other.hex()}").json()["job"] is None
    late = [{"benchmark_id": "q1", "verdict": "PASS", "confidence": 1.0}]
    r = tc.post(f"/jobs/{job_id}/verdicts", json={
        "verifier_pubkey": pub2.hex(), "verifier_org": "b", "verifier_model": "m",
        "verdicts": late, "signature": sign_verdict(priv2, job_id, late)})
    assert r.status_code == 410


def test_job_ttl_zero_disables_expiry(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from avir.network.coordinator import create_app
    db = str(tmp_path / "c.db")
    tc = TestClient(create_app(db, job_ttl_s=0))
    job_id = tc.post("/jobs", json={"spec": _spec(), "subject_responses": {"q1": ["x"]},
                                    "min_verifiers": 2}).json()["job_id"]
    _age_job(db, job_id, 10 * 86400)
    assert tc.get(f"/jobs/{job_id}").json()["status"] == "open"


def test_job_submission_token_gate(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from avir.network.coordinator import create_app
    app = create_app(str(tmp_path / "c.db"), token="secret")
    tc = TestClient(app)
    # no token -> 401
    assert tc.post("/jobs", json={"spec": _spec(), "subject_responses": {}}).status_code == 401
    # correct token -> ok
    assert tc.post("/jobs", json={"spec": _spec(), "subject_responses": {}},
                   headers={"X-AVIR-Token": "secret"}).status_code == 200


# --------------------------------------------------------------------------
# quorum counts only registered verifiers, with their registered org
# --------------------------------------------------------------------------

def _register(tc, pub_hex, org, headers=None):
    return tc.post("/verifiers/register", json={
        "pubkey": pub_hex, "label": org, "org": org, "models": ["m"]}, headers=headers or {})


def _signed_body(priv, pub_hex, job_id, org, verdict="PASS"):
    verdicts = [{"benchmark_id": "q1", "verdict": verdict, "confidence": 1.0}]
    return {"verifier_pubkey": pub_hex, "verifier_org": org, "verifier_model": "m",
            "verdicts": verdicts, "signature": sign_verdict(priv, job_id, verdicts)}


def _job(tc, min_verifiers, distinct_orgs=False):
    return tc.post("/jobs", json={
        "spec": _spec(), "subject_responses": {"q1": ["x"]},
        "min_verifiers": min_verifiers, "require_distinct_orgs": distinct_orgs,
    }).json()["job_id"]


def test_unregistered_keys_cannot_reach_quorum(client):
    """Fresh keys with valid signatures but no registry entry are refused."""
    job_id = _job(client, 2, distinct_orgs=True)
    for org in ("org-a", "org-b"):
        priv, pub = keys.generate_keypair()
        r = client.post(f"/jobs/{job_id}/verdicts",
                        json=_signed_body(priv, pub.hex(), job_id, org))
        assert r.status_code == 403
        assert "not registered" in r.json()["detail"]
    res = client.get(f"/jobs/{job_id}").json()
    assert res["status"] == "open"
    assert res["verdicts"] == []
    assert res["consensus"]["quorum_met"] is False


def test_body_org_cannot_satisfy_distinct_orgs(client):
    """Two keys registered under one org cannot claim two orgs in their verdicts."""
    job_id = _job(client, 2, distinct_orgs=True)
    parties = []
    for _ in range(2):
        priv, pub = keys.generate_keypair()
        assert _register(client, pub.hex(), "org-a").status_code == 200
        parties.append((priv, pub.hex()))
    (p1, k1), (p2, k2) = parties
    assert client.post(f"/jobs/{job_id}/verdicts",
                       json=_signed_body(p1, k1, job_id, "org-a")).status_code == 200
    r = client.post(f"/jobs/{job_id}/verdicts", json=_signed_body(p2, k2, job_id, "org-b"))
    assert r.status_code == 400
    assert "registered org" in r.json()["detail"]
    res = client.get(f"/jobs/{job_id}").json()
    assert res["status"] == "open"
    assert res["consensus"]["quorum_met"] is False
    assert [v["verifier_org"] for v in res["verdicts"]] == ["org-a"]


def test_verdict_records_the_registered_org(client):
    """An empty body org is accepted; the record carries the org on file."""
    job_id = _job(client, 2, distinct_orgs=True)
    for org in ("org-a", "org-b"):
        priv, pub = keys.generate_keypair()
        assert _register(client, pub.hex(), org).status_code == 200
        r = client.post(f"/jobs/{job_id}/verdicts", json=_signed_body(priv, pub.hex(), job_id, ""))
        assert r.status_code == 200
    res = client.get(f"/jobs/{job_id}").json()
    assert sorted(v["verifier_org"] for v in res["verdicts"]) == ["org-a", "org-b"]
    assert res["consensus"]["distinct_orgs"] == 2
    assert res["status"] == "complete"


def test_unregistered_key_is_not_served_jobs(client):
    _job(client, 1)
    _, pub = keys.generate_keypair()
    assert client.get(f"/jobs/next?pubkey={pub.hex()}").status_code == 403
    assert _register(client, pub.hex(), "org-a").status_code == 200
    assert client.get(f"/jobs/next?pubkey={pub.hex()}").json()["job"] is not None


def test_register_rejects_a_malformed_pubkey(client):
    for bad in ("zz", "ab" * 31, "not hex at all"):
        assert _register(client, bad, "org-a").status_code == 400


def test_registration_token_gate(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from avir.network.coordinator import create_app
    tc = TestClient(create_app(str(tmp_path / "c.db"), token="[REDACTED]",
                               registration_token="reg-secret"))
    _, pub = keys.generate_keypair()
    assert _register(tc, pub.hex(), "org-a").status_code == 401
    assert _register(tc, pub.hex(), "org-a",
                     headers={"X-AVIR-Registration-Token": "wrong"}).status_code == 401
    # the job-submission token does not authorize registration
    assert _register(tc, pub.hex(), "org-a",
                     headers={"X-AVIR-Token": "submit-secret"}).status_code == 401
    assert tc.get("/verifiers").json()["count"] == 0
    assert _register(tc, pub.hex(), "org-a",
                     headers={"X-AVIR-Registration-Token": "reg-secret"}).status_code == 200
    health = tc.get("/health").json()
    assert health["registration_required_token"] is True


def test_registration_is_open_without_a_registration_token(client):
    """Localhost-only deployments keep the original open registration."""
    _, pub = keys.generate_keypair()
    assert _register(client, pub.hex(), "org-a").status_code == 200
    assert client.get("/health").json()["registration_required_token"] is False


def test_verifier_sends_the_registration_token(monkeypatch):
    from avir.network import verifier as vmod
    seen = []
    monkeypatch.setattr(vmod, "_post", lambda base, path, body, timeout=30.0, headers=None:
                        seen.append(headers) or {"ok": True})
    vmod._register("https://x", {"pubkey": "p"}, retry_interval=None, token="reg-secret")
    vmod._register("https://x", {"pubkey": "p"}, retry_interval=None, token=None)
    assert seen == [{"X-AVIR-Registration-Token": "reg-secret"}, None]


def test_verifier_registers_again_when_the_coordinator_forgets_it(monkeypatch):
    """A 403 means the coordinator has no record of the key (e.g. a reset
    registry); the daemon registers again instead of judging jobs it cannot
    submit."""
    import urllib.error

    from avir.network import verifier as vmod
    registered = []

    def forbidden(*a, **k):
        raise urllib.error.HTTPError("https://x/jobs/next", 403, "Forbidden", {}, None)

    monkeypatch.setattr(vmod, "run_once", forbidden)
    did = vmod._poll_step("https://x", b"k", "p", None, "org-a", "m",
                          reregister=lambda: registered.append(True))
    assert did is False and registered == [True]

    def gone(*a, **k):
        raise urllib.error.HTTPError("https://x/jobs/j/verdicts", 410, "Gone", {}, None)

    monkeypatch.setattr(vmod, "run_once", gone)
    assert vmod._poll_step("https://x", b"k", "p", None, "org-a", "m",
                           reregister=lambda: registered.append(True)) is False
    assert registered == [True]


# --------------------------------------------------------------------------
# outbound URL scheme guard (bandit B310)
# --------------------------------------------------------------------------

def test_require_http_url_allows_only_http_and_https():
    from avir.urls import require_http_url
    assert require_http_url("http://h:8770/jobs") == "http://h:8770/jobs"
    assert require_http_url("HTTPS://h/x") == "HTTPS://h/x"
    for bad in ("file:///etc/passwd", "ftp://h/x", "data:,x", "/relative/path", ""):
        with pytest.raises(ValueError):
            require_http_url(bad)


def test_network_clients_refuse_file_urls_before_urlopen(monkeypatch):
    import urllib.request

    from avir.network import requester as rmod
    from avir.network import verifier as vmod
    opened = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: opened.append(a))
    with pytest.raises(ValueError):
        rmod._get("file:///etc", "/passwd")
    with pytest.raises(ValueError):
        rmod._post("file:///etc", "/passwd", {})
    with pytest.raises(ValueError):
        vmod._get("file:///etc", "/passwd")
    with pytest.raises(ValueError):
        vmod._post("file:///etc", "/passwd", {})
    assert opened == []


def test_http_model_provider_guard_abstains_on_file_endpoint_and_passes_http(monkeypatch):
    import asyncio
    import io
    import json
    import urllib.request

    from avir.providers.cli import HTTPModelProvider
    opened = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        opened.append(req.full_url)
        body = {"choices": [{"message": {"content": "VERDICT: PASS\nCONFIDENCE: 0.9"}}]}
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    bad = HTTPModelProvider(endpoint="file:///etc", model="m")
    jv = asyncio.run(bad.judge_output(prompt="p", response="r", rubric="x"))
    assert jv.verdict == "ABSTAIN" and "scheme" in (jv.abstain_reason or "")
    assert opened == []

    good = HTTPModelProvider(endpoint="http://127.0.0.1:11434", model="m")
    jv = asyncio.run(good.judge_output(prompt="p", response="r", rubric="x"))
    assert jv.verdict == "PASS"
    assert opened == ["http://127.0.0.1:11434/v1/chat/completions"]
