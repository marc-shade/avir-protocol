"""
AVIR verifier client: contribute YOUR compute and AI providers to the network.

Runs as a small daemon (stdlib HTTP only -- no heavy deps): it registers in the
coordinator's participant registry, pulls blinded jobs, judges them with YOUR
local provider (claude/codex/gemini CLI, or any OpenAI-compatible endpoint),
signs the verdict with your Ed25519 key, and submits it. Your provider
credentials never leave your machine.

Run it:
  avir-verifier --coordinator http://host:8770 --provider claude-cli --org alice
  avir-verifier --coordinator http://host:8770 --provider http \
                --endpoint http://localhost:11434 --model llama3 --org alice
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..providers import (
    ClaudeCLIProvider,
    CodexCLIProvider,
    GeminiCLIProvider,
    HTTPModelProvider,
)
from .keys import fingerprint, load_or_create_identity
from .protocol import sign_verdict
from .tls import default_client_context

_CTX = None


def _ctx():
    """Cached client SSL context (used for https:// coordinators)."""
    global _CTX
    if _CTX is None:
        _CTX = default_client_context()
    return _CTX


def _get(base: str, path: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as resp:
        return json.loads(resp.read().decode())


def _post(base: str, path: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as resp:
        return json.loads(resp.read().decode())


def build_provider(args: argparse.Namespace):
    if args.provider == "claude-cli":
        return ClaudeCLIProvider()
    if args.provider == "codex-cli":
        return CodexCLIProvider()
    if args.provider == "gemini-cli":
        return GeminiCLIProvider()
    if args.provider == "http":
        if not args.endpoint or not args.model:
            raise SystemExit("--provider http requires --endpoint and --model")
        return HTTPModelProvider(
            endpoint=args.endpoint, model=args.model,
            provider_id=f"http:{args.model}", organization=args.org or "independent",
            node=args.endpoint,
        )
    raise SystemExit(f"unknown provider: {args.provider}")


async def _judge_job(provider, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Judge every rubric benchmark in the job; one verdict per benchmark.

    Multiple subject responses for a benchmark are each judged and reduced to a
    majority PASS/FAIL (ABSTAIN if no decisive judgments).
    """
    responses = job.get("subject_responses", {})
    verdicts: list[dict[str, Any]] = []
    for bm in job.get("benchmarks", []):
        bm_id = bm.get("benchmark_id")
        texts = responses.get(bm_id, []) or []
        decisive: list[str] = []
        confidences: list[float] = []
        for text in texts:
            jv = await provider.judge_output(
                prompt=bm.get("prompt", ""), response=text,
                rubric=bm.get("rubric", ""), expected=bm.get("expected", ""),
            )
            v = getattr(jv, "verdict", "ABSTAIN")
            if v in ("PASS", "FAIL"):
                decisive.append(v)
                if jv.confidence is not None:
                    confidences.append(float(jv.confidence))
            else:
                # An abstention carries its cause; without this line a failing
                # judge endpoint is indistinguishable from an undecided one.
                reason = getattr(jv, "abstain_reason", None) or "no reason given"
                print(f"[verifier] ABSTAIN on {bm_id}: {reason}", file=sys.stderr, flush=True)
        if not decisive:
            verdict = "ABSTAIN"
        else:
            passes = sum(1 for d in decisive if d == "PASS")
            verdict = "PASS" if passes > len(decisive) / 2 else "FAIL"
        verdicts.append({
            "benchmark_id": bm_id,
            "verdict": verdict,
            "confidence": (sum(confidences) / len(confidences)) if confidences else None,
        })
    return verdicts


def run_once(base: str, priv: bytes, pub_hex: str, provider, org: str, model_label: str) -> bool:
    """Pull one job, judge, sign, submit. Returns True if a job was processed."""
    resp = _get(base, f"/jobs/next?pubkey={pub_hex}")
    job = resp.get("job")
    if not job:
        return False
    job_id = job["job_id"]
    verdicts = asyncio.run(_judge_job(provider, job))
    signature = sign_verdict(priv, job_id, verdicts)
    result = _post(base, f"/jobs/{job_id}/verdicts", {
        "verifier_pubkey": pub_hex,
        "verifier_org": org,
        "verifier_model": model_label,
        "verdicts": verdicts,
        "signature": signature,
    })
    print(f"[verifier] job {job_id[:8]} judged "
          f"({sum(1 for v in verdicts if v['verdict']!='ABSTAIN')}/{len(verdicts)} decisive); "
          f"accepted={result.get('accepted')} quorum_met={result.get('quorum_met')}")
    return True


def _register(base: str, body: dict[str, Any], *, retry_interval: float | None) -> None:
    """Register with the coordinator.

    A daemon (``retry_interval`` set) waits out an unreachable coordinator, as it
    does when polling: at boot its SSH tunnel or the coordinator may not be up
    yet, and exiting would leave recovery to the service manager's restart delay.
    An HTTP refusal is a configuration error and raises at once.
    """
    while True:
        try:
            _post(base, "/verifiers/register", body)
            return
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            if retry_interval is None:
                raise
            print(f"[verifier] coordinator unreachable, retrying registration: {exc}")
            time.sleep(retry_interval)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an AVIR verifier node (contribute to the network).")
    ap.add_argument("--coordinator", required=True, help="coordinator base URL, e.g. http://host:8770")
    ap.add_argument("--provider", required=True,
                    choices=["claude-cli", "codex-cli", "gemini-cli", "http"])
    ap.add_argument("--endpoint", help="for --provider http: OpenAI-compatible base URL")
    ap.add_argument("--model", help="for --provider http: model name")
    ap.add_argument("--org", default="independent", help="your organization label (diversity matters)")
    ap.add_argument("--label", default="", help="human label for your node")
    ap.add_argument("--identity", default=str(Path(os.path.expanduser("~/.avir/verifier_identity"))))
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="process at most one job then exit")
    args = ap.parse_args()

    priv, pub = load_or_create_identity(args.identity)
    pub_hex = pub.hex()
    provider = build_provider(args)
    model_label = getattr(provider, "model", args.provider)
    org = args.org or provider.get_organization()
    label = args.label or fingerprint(pub_hex)

    _register(args.coordinator, {
        "pubkey": pub_hex, "label": label, "org": org,
        "models": [model_label],
    }, retry_interval=None if args.once else args.poll_interval)
    print(f"[verifier] registered {label} ({fingerprint(pub_hex)}) org={org} model={model_label}")

    if args.once:
        did = run_once(args.coordinator, priv, pub_hex, provider, org, model_label)
        if not did:
            print("[verifier] no open jobs")
        return 0

    print(f"[verifier] polling {args.coordinator} every {args.poll_interval}s (Ctrl-C to stop)")
    while True:
        try:
            did = run_once(args.coordinator, priv, pub_hex, provider, org, model_label)
        except urllib.error.HTTPError as exc:
            # The coordinator answered but refused (e.g. 410: job expired while judging).
            print(f"[verifier] coordinator rejected request: HTTP {exc.code} {exc.reason}")
            did = False
        except urllib.error.URLError as exc:
            print(f"[verifier] coordinator unreachable: {exc}")
            did = False
        if not did:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
