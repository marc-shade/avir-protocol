#!/usr/bin/env python3
"""End-to-end demo of the AVIR verification NETWORK on the local cluster.

Starts a real coordinator daemon, registers three independent verifier nodes
backed by DIFFERENT models on DIFFERENT machines (fedora.local + ai-lab.local),
submits a blinded verification job, has each node judge it and submit an
Ed25519-signed verdict, then polls the result and RE-VERIFIES every signature
locally. This is the distributed analogue of examples/cluster_cross_verify.py:
instead of one process orchestrating the panel, independent nodes pull work from
a job board and sign their own verdicts.

Usage: python examples/network_local_demo.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avir.network import keys  # noqa: E402
from avir.network.requester import _get, _post, poll_result, reverify_result, submit_job  # noqa: E402
from avir.network.verifier import run_once  # noqa: E402
from avir.providers import ailab_model, fedora_ollama  # noqa: E402

PORT = 8771
BASE = f"http://127.0.0.1:{PORT}"

SPEC = {
    "avir_version": "1.0.0",
    "system": {"name": "network-demonstrator", "version": "1.0.0"},
    "capabilities": [],
    "benchmarks": [{
        "id": "bat_and_ball",
        "kind": "rubric",
        "prompt": "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball, in cents?",
        "rubric": "PASS iff the final answer is 5 cents ($0.05); 10 cents or anything else is FAIL.",
        "expected": "5 cents (bat $1.05, ball $0.05).",
    }],
}
RESPONSES = {
    "bat_and_ball": [
        "Let ball = x. Bat = x + 1.00. x + (x+1.00) = 1.10 -> 2x = 0.10 -> x = 0.05. "
        "The ball costs 5 cents (bat $1.05, ball $0.05)."
    ],
}


def main() -> int:
    db = Path("/tmp/avir_netdemo.db")
    if db.exists():
        db.unlink()

    coord = subprocess.Popen(
        [sys.executable, "-m", "avir.network.coordinator", "--port", str(PORT), "--db", str(db)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # wait for the daemon
        for _ in range(40):
            try:
                _get(BASE, "/health")
                break
            except urllib.error.URLError:
                time.sleep(0.5)
        else:
            print("coordinator did not start")
            return 1
        print(f"[demo] coordinator up at {BASE}")

        job_id = submit_job(BASE, spec=SPEC, subject_responses=RESPONSES, min_verifiers=3)
        print(f"[demo] submitted job {job_id}")

        # Independent verifier nodes: distinct keys, distinct models, distinct machines.
        nodes = [
            ("deepseek", fedora_ollama("deepseek-v4-pro:cloud", organization="deepseek")),
            ("deepreinforce", ailab_model("ornith-9b", organization="deepreinforce")),
            ("zhipu", fedora_ollama("glm-5.1:cloud", organization="zhipu")),
        ]
        for org, provider in nodes:
            priv, pub = keys.generate_keypair()
            _post(BASE, "/verifiers/register", {
                "pubkey": pub.hex(), "label": f"{org}-node", "org": org,
                "models": [provider.model],
            })
            run_once(BASE, priv, pub.hex(), provider, org, provider.model)

        registry = _get(BASE, "/verifiers")
        print(f"[demo] participant registry: {registry['count']} verifiers")

        result = poll_result(BASE, job_id, timeout_s=30)
        checked = reverify_result(result)
        c = checked["consensus"]
        print(f"\n[demo] job status: {result['status']}")
        print(f"[demo] signatures re-verified locally: "
              f"{len(checked['valid_records'])} valid, {len(checked['invalid_signatures'])} invalid")
        for rec in checked["valid_records"]:
            v = rec["verdicts"][0]
            print(f"  {rec['verifier_org']:14s} {rec['verifier_model']:24s} "
                  f"-> {v['verdict']}  (key {rec['verifier_pubkey'][:12]}...)")
        print(f"\n[demo] NETWORK CONSENSUS: {c['overall_verdict']}  "
              f"verifiers={c['distinct_verifiers']} orgs={c['distinct_orgs']} "
              f"agreement={c['agreement_score']:.2f} quorum_met={c['quorum_met']}")
        return 0
    finally:
        coord.terminate()
        try:
            coord.wait(timeout=5)
        except subprocess.TimeoutExpired:
            coord.kill()


if __name__ == "__main__":
    raise SystemExit(main())
