"""
AVIR requester client: ask the network to verify, then trust only signatures.

Submits a blinded verification job (a spec with rubric benchmarks + the subject
responses to judge) and polls for the aggregated result. Critically, it
RE-VERIFIES every verdict's Ed25519 signature locally, so the coordinator is
never trusted for integrity -- only for transport and queueing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from .keys import load_or_create_identity
from .protocol import aggregate_consensus, verify_verdict
from .tls import default_client_context

_CTX = None


def _ctx():
    """Cached client SSL context (used for https:// coordinators)."""
    global _CTX
    if _CTX is None:
        _CTX = default_client_context()
    return _CTX


def _post(base: str, path: str, body: dict[str, Any], token: str | None = None,
          timeout: float = 30.0) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-AVIR-Token"] = token
    req = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(body).encode(),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as resp:
        return json.loads(resp.read().decode())


def _get(base: str, path: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as resp:
        return json.loads(resp.read().decode())


def reverify_result(result: dict[str, Any]) -> dict[str, Any]:
    """Re-check every verdict signature locally and recompute consensus.

    Returns {valid_records, invalid_signatures, consensus}. Only signature-valid
    records feed the recomputed consensus -- never trust the coordinator's math.
    """
    job_id = result["job_id"]
    valid: list[dict[str, Any]] = []
    invalid: list[str] = []
    for rec in result.get("verdicts", []):
        ok = verify_verdict(
            rec["verifier_pubkey"], job_id, rec["verdicts"], rec["signature"],
        )
        (valid if ok else invalid).append(rec if ok else rec["verifier_pubkey"])
    consensus = aggregate_consensus(
        valid,
        min_verifiers=result.get("min_verifiers", 3),
        require_distinct_orgs=result.get("require_distinct_orgs", False),
    )
    return {"valid_records": valid, "invalid_signatures": invalid, "consensus": consensus}


def submit_job(
    coordinator: str,
    *,
    spec: dict[str, Any],
    subject_responses: dict[str, list[str]],
    min_verifiers: int = 3,
    require_distinct_orgs: bool = False,
    requester_pubkey: str = "",
    token: str | None = None,
) -> str:
    resp = _post(coordinator, "/jobs", {
        "spec": spec,
        "subject_responses": subject_responses,
        "min_verifiers": min_verifiers,
        "require_distinct_orgs": require_distinct_orgs,
        "requester_pubkey": requester_pubkey,
    }, token=token)
    return resp["job_id"]


def poll_result(coordinator: str, job_id: str, *, timeout_s: float = 600.0,
                interval_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        result = _get(coordinator, f"/jobs/{job_id}")
        if result.get("status") == "complete":
            return result
        if time.monotonic() >= deadline:
            return result  # return whatever we have (may be sub-quorum)
        time.sleep(interval_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit a verification job to the AVIR network.")
    ap.add_argument("--coordinator", required=True)
    ap.add_argument("--spec", required=True, help="path to an AVIR spec JSON (rubric benchmarks)")
    ap.add_argument("--responses", required=True,
                    help="path to JSON: {benchmark_id: [response strings]} to be judged")
    ap.add_argument("--min-verifiers", type=int, default=3)
    ap.add_argument("--require-distinct-orgs", action="store_true")
    ap.add_argument("--identity", default=str(Path(os.path.expanduser("~/.avir/requester_identity"))))
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    token = os.environ.get("AVIR_COORDINATOR_TOKEN")
    _, pub = load_or_create_identity(args.identity)
    spec = json.loads(Path(args.spec).read_text())
    responses = json.loads(Path(args.responses).read_text())

    job_id = submit_job(
        args.coordinator, spec=spec, subject_responses=responses,
        min_verifiers=args.min_verifiers, require_distinct_orgs=args.require_distinct_orgs,
        requester_pubkey=pub.hex(), token=token,
    )
    print(f"[requester] submitted job {job_id}; waiting for {args.min_verifiers} verifiers...")
    result = poll_result(args.coordinator, job_id, timeout_s=args.timeout)
    checked = reverify_result(result)
    c = checked["consensus"]
    print(f"[requester] status={result['status']} "
          f"verifiers={c['distinct_verifiers']} orgs={c['distinct_orgs']}")
    print(f"[requester] CONSENSUS={c['overall_verdict']} agreement={c['agreement_score']:.2f} "
          f"(quorum_met={c['quorum_met']})")
    # Quorum counts only decisive votes, so show per benchmark how many there were.
    for bm_id, b in sorted(c.get("benchmarks", {}).items()):
        short = "" if b["quorum_met"] else "  [below quorum]"
        print(f"[requester]   {bm_id}: {b['verdict']} "
              f"({b['decisive_votes']}/{b['total_votes']} decisive votes){short}")
    abstained = sum(b["total_votes"] - b["decisive_votes"] for b in c.get("benchmarks", {}).values())
    if abstained:
        print(f"[requester] WARNING: {abstained} abstention(s), not counted toward quorum; "
              f"see the verifiers' logs for the cause")
    if checked["invalid_signatures"]:
        print(f"[requester] WARNING: {len(checked['invalid_signatures'])} invalid signature(s) excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
