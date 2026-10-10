"""
Verification-network protocol: what gets signed, and how verdicts aggregate.

A verifier signs the canonical bytes of {job_id, verdicts}. The coordinator and
any requester can verify that signature against the verifier's public key, so a
verdict cannot be forged or altered in transit. Consensus is computed over
distinct verifier keys (optionally distinct organizations), excluding the
requester's own key.
"""

from __future__ import annotations

from typing import Any

from .keys import canonical, sign, verify


def verdict_signing_payload(job_id: str, verdicts: list[dict[str, Any]]) -> bytes:
    """Canonical bytes a verifier signs for a job."""
    normalized = [
        {
            "benchmark_id": str(v.get("benchmark_id", "")),
            "verdict": str(v.get("verdict", "ABSTAIN")).upper(),
            "confidence": v.get("confidence"),
        }
        for v in verdicts
    ]
    return canonical({"job_id": job_id, "verdicts": normalized})


def sign_verdict(priv: bytes, job_id: str, verdicts: list[dict[str, Any]]) -> str:
    return sign(priv, verdict_signing_payload(job_id, verdicts))


def verify_verdict(
    pub_hex: str,
    job_id: str,
    verdicts: list[dict[str, Any]],
    sig_hex: str,
) -> bool:
    return verify(pub_hex, verdict_signing_payload(job_id, verdicts), sig_hex)


def _benchmark_consensus(
    judgments: list[str],
) -> tuple[str, float, int]:
    """Given verdict strings for one benchmark, return (verdict, pass_rate, n).

    Only PASS/FAIL count toward the rate; ABSTAIN is excluded. ``n`` is the
    number of non-abstaining verifiers.
    """
    decisive = [j for j in judgments if j in ("PASS", "FAIL")]
    n = len(decisive)
    if n == 0:
        return "INCONCLUSIVE", 0.0, 0
    passes = sum(1 for j in decisive if j == "PASS")
    pass_rate = passes / n
    verdict = "PASS" if pass_rate > 0.5 else ("FAIL" if pass_rate < 0.5 else "TIE")
    return verdict, pass_rate, n


def aggregate_consensus(
    verdict_records: list[dict[str, Any]],
    *,
    min_verifiers: int = 3,
    require_distinct_orgs: bool = False,
    exclude_pubkeys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Aggregate signed verdict records into a network consensus.

    Each record: {verifier_pubkey, verifier_org, verdicts: [{benchmark_id,
    verdict, confidence}]}. Records from ``exclude_pubkeys`` (e.g. the
    requester's own key) are dropped before counting.

    Quorum counts only decisive (PASS/FAIL) votes: every benchmark needs
    ``min_verifiers`` distinct keys (and, with ``require_distinct_orgs``, that
    many distinct orgs) voting PASS or FAIL on it. An ABSTAIN is a submission,
    not a judgment, so it never counts toward quorum.

    Returns a dict with per-benchmark results, the overall verdict, the number
    of distinct verifiers/orgs that participated, and an agreement score.
    """
    records = [
        r for r in verdict_records
        if str(r.get("verifier_pubkey", "")) not in set(exclude_pubkeys)
    ]

    distinct_keys = {str(r.get("verifier_pubkey", "")) for r in records if r.get("verifier_pubkey")}
    distinct_orgs = {str(r.get("verifier_org", "")) for r in records if r.get("verifier_org")}
    org_of = {str(r.get("verifier_pubkey", "")): str(r.get("verifier_org", "")) for r in records}

    # Collect judgments per benchmark, one vote per distinct verifier key.
    per_benchmark: dict[str, dict[str, str]] = {}
    for r in records:
        pub = str(r.get("verifier_pubkey", ""))
        for v in r.get("verdicts", []):
            bm = str(v.get("benchmark_id", ""))
            verdict = str(v.get("verdict", "ABSTAIN")).upper()
            per_benchmark.setdefault(bm, {})[pub] = verdict

    benchmarks: dict[str, Any] = {}
    total_votes = 0
    agreeing_votes = 0
    passed = 0
    decided = 0
    for bm, votes in per_benchmark.items():
        verdict, pass_rate, n = _benchmark_consensus(list(votes.values()))
        decisive_keys = {pub for pub, j in votes.items() if pub and j in ("PASS", "FAIL")}
        decisive_orgs = {org_of[pub] for pub in decisive_keys if org_of.get(pub)}
        bm_quorum = len(decisive_keys) >= min_verifiers
        if require_distinct_orgs:
            bm_quorum = bm_quorum and len(decisive_orgs) >= min_verifiers
        benchmarks[bm] = {
            "verdict": verdict,
            "pass_rate": pass_rate,
            "decisive_votes": n,
            "total_votes": len(votes),
            "decisive_orgs": len(decisive_orgs),
            "quorum_met": bm_quorum,
        }
        if verdict in ("PASS", "FAIL"):
            decided += 1
            if verdict == "PASS":
                passed += 1
            majority = "PASS" if pass_rate > 0.5 else "FAIL"
            for j in votes.values():
                if j in ("PASS", "FAIL"):
                    total_votes += 1
                    if j == majority:
                        agreeing_votes += 1

    agreement_score = (agreeing_votes / total_votes) if total_votes else 0.0

    quorum_ok = bool(benchmarks) and all(b["quorum_met"] for b in benchmarks.values())

    if not quorum_ok or decided == 0:
        overall = "INCONCLUSIVE"
    else:
        bench_pass_rate = passed / decided
        if bench_pass_rate >= 0.8:
            overall = "VERIFIED"
        elif bench_pass_rate >= 0.5:
            overall = "PARTIAL"
        else:
            overall = "FAILED"

    return {
        "overall_verdict": overall,
        "benchmarks": benchmarks,
        "distinct_verifiers": len(distinct_keys),
        "distinct_orgs": len(distinct_orgs),
        "min_verifiers": min_verifiers,
        "require_distinct_orgs": require_distinct_orgs,
        "quorum_met": quorum_ok,
        "agreement_score": agreement_score,
        "benchmarks_passed": passed,
        "benchmarks_decided": decided,
    }
