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
    for name, org in [("alice", "org-a"), ("bob", "org-b"), ("carol", "org-c")]:
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
        rr = client.post(f"/jobs/{job_id}/verdicts", json={
            "verifier_pubkey": pub_hex, "verifier_org": "x", "verifier_model": "m",
            "verdicts": verdicts, "signature": sig})
        assert rr.status_code == 200

    # result: quorum met, VERIFIED, job closed
    res = client.get(f"/jobs/{job_id}").json()
    assert res["status"] == "complete"
    assert res["consensus"]["overall_verdict"] == "VERIFIED"
    assert res["consensus"]["distinct_verifiers"] == 3


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
    job_id = client.post("/jobs", json={
        "spec": _spec(), "subject_responses": {"q1": ["x"]}, "min_verifiers": 1}).json()["job_id"]
    verdicts = [{"benchmark_id": "q1", "verdict": "PASS", "confidence": 1.0}]
    sig = sign_verdict(priv, job_id, verdicts)
    body = {"verifier_pubkey": pub.hex(), "verifier_org": "x", "verifier_model": "m",
            "verdicts": verdicts, "signature": sig}
    assert client.post(f"/jobs/{job_id}/verdicts", json=body).status_code == 200
    assert client.post(f"/jobs/{job_id}/verdicts", json=body).status_code == 409


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
