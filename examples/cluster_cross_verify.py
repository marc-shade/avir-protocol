#!/usr/bin/env python3
"""Run a REAL cross-provider, cross-node AVIR verification on the local cluster.

Panel (genuinely different providers, isolated by process and by physical node):
  * claude  -- Anthropic, headless CLI (separate OS process)
  * codex   -- OpenAI,    headless CLI (separate OS process)
  * gemini  -- Google,    headless CLI (separate OS process)
  * fedora:deepseek-v4-pro -- DeepSeek family, runs on fedora.local (GPU node)
  * ailab:ornith-9b        -- DeepReinforce family, runs on ai-lab.local (GPU node)

Each subject answers the benchmark prompt; every OTHER provider independently
judges the (blinded) answer. No provider judges its own work in consensus.
Produces a signed Ed25519 attestation and verifies it.

Usage: python examples/cluster_cross_verify.py [spec.json] [out.json]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avir import Specification, BlindingMode, IsolationLevel  # noqa: E402
from avir.cross_verification import CrossVerifier  # noqa: E402
from avir.providers import (  # noqa: E402
    ClaudeCLIProvider,
    CodexCLIProvider,
    GeminiCLIProvider,
    fedora_ollama,
    ailab_model,
)
from avir.attestation import (  # noqa: E402
    create_attestation,
    EnvironmentInfo,
    ProviderInfo,
    VerificationLevel,
)


def build_panel():
    return [
        ClaudeCLIProvider(timeout=180),
        CodexCLIProvider(timeout=200),
        GeminiCLIProvider(timeout=150),
        fedora_ollama("deepseek-v4-pro:cloud", organization="deepseek", timeout=180),
        ailab_model("ornith-9b", organization="deepreinforce", timeout=180),
    ]


async def main() -> int:
    here = Path(__file__).resolve().parent
    spec_path = sys.argv[1] if len(sys.argv) > 1 else str(here / "cluster_demonstrator_spec.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/avir_cluster_result.json"

    spec = Specification.from_file(spec_path)
    providers = build_panel()
    print(f"[avir] spec={spec.system.name} providers={[p.provider_id for p in providers]}")

    started = datetime.utcnow()
    cv = CrossVerifier(
        providers=providers,
        blinding_mode=BlindingMode.DOUBLE_BLIND,
        isolation_level=IsolationLevel.DISTRIBUTED,
        iterations=1,
    )
    matrix = await cv.verify(spec, require_minimum_providers=2)
    completed = datetime.utcnow()

    # Per-cell summary (the real cross-verification matrix)
    cells = {}
    print("\n[avir] cross-verification cells (verifier -> subject : verdict):")
    for (verifier_id, subject_id), c in sorted(matrix.cells.items()):
        tag = "(self)" if verifier_id == subject_id else ""
        print(f"  {verifier_id:28s} -> {subject_id:28s} {c.verdict.value:11s} "
              f"pass={c.pass_rate:.2f} abstain={c.abstain_count} {tag} [{c.verifier_locus}]")
        cells[f"{verifier_id}|{subject_id}"] = {
            "verdict": c.verdict.value,
            "pass_rate": c.pass_rate,
            "abstain_count": c.abstain_count,
            "canary_detected": c.canary_detected,
            "verifier_locus": c.verifier_locus,
            "is_self": verifier_id == subject_id,
        }

    md = matrix.to_dict()
    results_hash = hashlib.sha256(
        json.dumps(md, sort_keys=True, default=str).encode()
    ).hexdigest()

    consensus_pass = matrix.consensus_pass_rate
    bench_verdicts = [{
        "benchmark_id": "bat_and_ball",
        "verdict": "PASS" if consensus_pass >= 0.5 else "FAIL",
        "consensus_pass_rate": consensus_pass,
    }]

    env = EnvironmentInfo(cpu_cores=os.cpu_count())
    verifier_info = ProviderInfo(
        provider="avir-cluster-panel",
        model="+".join(p.provider_id for p in providers),
        instance_id=platform.node(),
    )
    att = create_attestation(
        spec_hash=spec.hash,
        env=env,
        verifier=verifier_info,
        benchmark_verdicts=bench_verdicts,
        system_name=spec.system.name,
        system_version=spec.system.version,
        started_at=started,
        completed_at=completed,
        results_hash=results_hash,
        level=VerificationLevel.L4_COMPREHENSIVE,
        spec_source=spec_path,
    )

    signed = False
    sign_error = None
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        sk = ed25519.Ed25519PrivateKey.generate()
        att.sign(sk.private_bytes_raw())
        signed = att.verify_signature()
    except Exception as exc:  # noqa: BLE001
        sign_error = str(exc)

    result = {
        "consensus": {
            "verdict": matrix.consensus_verdict.value,
            "pass_rate": consensus_pass,
            "agreement_score": matrix.agreement_score,
        },
        "cells": cells,
        "attestation": att.to_dict(),
        "signature_verified": signed,
        "sign_error": sign_error,
        "chain_verified": att.chain.verify(),
        "matrix": md,
    }
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))

    print("\n[avir] CONSENSUS:", result["consensus"])
    print(f"[avir] attestation chain_verified={result['chain_verified']} "
          f"signature_verified={signed} (sign_error={sign_error})")
    print(f"[avir] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
