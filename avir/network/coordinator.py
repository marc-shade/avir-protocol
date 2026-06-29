"""
AVIR coordinator: the verification-network job board (a persistent daemon).

The coordinator queues blinded verification jobs and collects Ed25519-signed
verdicts from independent verifier nodes. It is UNTRUSTED for integrity: it
verifies and stores signatures but cannot forge them, and any party can
re-check every signature in the result.

Run it:  avir-coordinator --host 127.0.0.1 --port 8770 --db ~/.avir/coordinator.db
Bind 0.0.0.0 only deliberately, and set AVIR_COORDINATOR_TOKEN to gate job
submission when exposed beyond localhost.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .protocol import aggregate_consensus, verify_verdict
from .registry import Registry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RegisterBody(BaseModel):
    pubkey: str
    label: str = ""
    org: str = ""
    models: list[str] = Field(default_factory=list)


class JobBody(BaseModel):
    spec: dict[str, Any]
    subject_responses: dict[str, list[str]]
    min_verifiers: int = 3
    require_distinct_orgs: bool = False
    requester_pubkey: str = ""


class VerdictItem(BaseModel):
    benchmark_id: str
    verdict: str
    confidence: Optional[float] = None


class VerdictBody(BaseModel):
    verifier_pubkey: str
    verifier_org: str = ""
    verifier_model: str = ""
    verdicts: list[VerdictItem]
    signature: str


def create_app(db_path: str, *, token: Optional[str] = None) -> FastAPI:
    registry = Registry(db_path)
    app = FastAPI(title="AVIR Coordinator", version="1.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "avir-coordinator", "auth_required": bool(token)}

    # --- participant registry (the directory verifiers use to find each other) ---
    @app.post("/verifiers/register")
    def register(body: RegisterBody) -> dict[str, Any]:
        if not body.pubkey:
            raise HTTPException(400, "pubkey required")
        return registry.register_verifier(
            pubkey=body.pubkey, label=body.label, org=body.org,
            models=body.models, now=_now(),
        )

    @app.get("/verifiers")
    def verifiers() -> dict[str, Any]:
        items = registry.list_verifiers()
        return {"count": len(items), "verifiers": items}

    @app.get("/verifiers/{fp}")
    def verifier_by_fp(fp: str) -> dict[str, Any]:
        for v in registry.list_verifiers():
            if v["fingerprint"] == fp or v["pubkey"] == fp:
                return v
        raise HTTPException(404, "verifier not found")

    # --- jobs ---
    @app.post("/jobs")
    def submit_job(body: JobBody, x_avir_token: Optional[str] = Header(default=None)) -> dict[str, Any]:
        if token and x_avir_token != token:
            raise HTTPException(401, "invalid or missing X-AVIR-Token")
        job_id = uuid.uuid4().hex
        registry.create_job(
            job_id=job_id, spec=body.spec, subject_responses=body.subject_responses,
            min_verifiers=body.min_verifiers, require_distinct_orgs=body.require_distinct_orgs,
            requester_pubkey=body.requester_pubkey, now=_now(),
        )
        return {"job_id": job_id, "status": "open"}

    @app.get("/jobs/next")
    def next_job(pubkey: str) -> dict[str, Any]:
        if not pubkey:
            raise HTTPException(400, "pubkey query param required")
        registry.touch_verifier(pubkey, _now())
        job = registry.next_open_job_for(pubkey)
        if job is None:
            return {"job": None}
        # Blinded view: only the spec benchmarks + responses, no requester identity.
        spec = job["spec"]
        benchmarks = [
            {
                "benchmark_id": b.get("id"),
                "kind": b.get("kind", "numeric"),
                "prompt": b.get("prompt", ""),
                "rubric": b.get("rubric", ""),
                "expected": b.get("expected", ""),
            }
            for b in spec.get("benchmarks", [])
            if b.get("kind") == "rubric"
        ]
        return {
            "job": {
                "job_id": job["job_id"],
                "benchmarks": benchmarks,
                "subject_responses": job["subject_responses"],
                "min_verifiers": job["min_verifiers"],
            }
        }

    @app.post("/jobs/{job_id}/verdicts")
    def submit_verdict(job_id: str, body: VerdictBody) -> dict[str, Any]:
        job = registry.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        verdicts = [v.model_dump() for v in body.verdicts]
        if not verify_verdict(body.verifier_pubkey, job_id, verdicts, body.signature):
            raise HTTPException(400, "signature verification failed")
        stored = registry.add_verdict(
            job_id=job_id, verifier_pubkey=body.verifier_pubkey,
            verifier_org=body.verifier_org, verifier_model=body.verifier_model,
            verdicts=verdicts, signature=body.signature, now=_now(),
        )
        if not stored:
            raise HTTPException(409, "this verifier already submitted for this job")
        registry.increment_judged(body.verifier_pubkey, _now())
        # Close the job once quorum of distinct verifiers is reached.
        records = registry.list_verdicts(job_id)
        consensus = aggregate_consensus(
            records, min_verifiers=job["min_verifiers"],
            require_distinct_orgs=job["require_distinct_orgs"],
            exclude_pubkeys=(job["requester_pubkey"],) if job["requester_pubkey"] else (),
        )
        if consensus["quorum_met"]:
            registry.set_job_status(job_id, "complete")
        return {"accepted": True, "verdicts_so_far": len(records), "quorum_met": consensus["quorum_met"]}

    @app.get("/jobs/{job_id}")
    def job_result(job_id: str) -> dict[str, Any]:
        job = registry.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        records = registry.list_verdicts(job_id)
        consensus = aggregate_consensus(
            records, min_verifiers=job["min_verifiers"],
            require_distinct_orgs=job["require_distinct_orgs"],
            exclude_pubkeys=(job["requester_pubkey"],) if job["requester_pubkey"] else (),
        )
        return {
            "job_id": job_id,
            "status": job["status"],
            "min_verifiers": job["min_verifiers"],
            "require_distinct_orgs": job["require_distinct_orgs"],
            "consensus": consensus,
            "verdicts": records,  # each carries its verifier pubkey + signature
        }

    return app


def main() -> int:
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Run the AVIR coordinator daemon.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to expose, deliberately)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--db", default=str(os.path.expanduser("~/.avir/coordinator.db")))
    ap.add_argument("--tls-cert", default=os.environ.get("AVIR_TLS_CERT"),
                    help="TLS certificate (PEM); enables HTTPS. Generate with avir-gen-cert.")
    ap.add_argument("--tls-key", default=os.environ.get("AVIR_TLS_KEY"),
                    help="TLS private key (PEM)")
    args = ap.parse_args()
    token = os.environ.get("AVIR_COORDINATOR_TOKEN")
    tls = bool(args.tls_cert and args.tls_key)
    exposed = args.host not in ("127.0.0.1", "localhost", "::1")
    if exposed and not token:
        print("WARNING: binding beyond localhost without AVIR_COORDINATOR_TOKEN; "
              "job submission is ungated.")
    if exposed and not tls:
        print("WARNING: binding beyond localhost WITHOUT TLS; traffic is unencrypted. "
              "Generate a cert with `avir-gen-cert` and pass --tls-cert/--tls-key.")
    scheme = "https" if tls else "http"
    print(f"[coordinator] {scheme}://{args.host}:{args.port}  tls={tls}  auth={bool(token)}")
    uvicorn.run(
        app=create_app(args.db, token=token),
        host=args.host, port=args.port,
        ssl_certfile=args.tls_cert if tls else None,
        ssl_keyfile=args.tls_key if tls else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
