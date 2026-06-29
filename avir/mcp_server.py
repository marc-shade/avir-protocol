"""
AVIR MCP server: drive the verification network from an agent/harness session.

Exposes the network as MCP tools so a Claude (or other) session can submit a
verification job to independent external verifiers and read back the
signature-reverified consensus -- without shelling out. Thin wrapper over
``avir.network.requester``; the coordinator URL is configured via the
``AVIR_COORDINATOR`` environment variable (or passed per call).

Run as an MCP stdio server:  python -m avir.mcp_server
Requires the ``[mcp]`` extra:  pip install avir-protocol[mcp,network]
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .network.requester import (
    _get,
    poll_result,
    reverify_result,
    submit_job,
)

mcp = FastMCP("avir")


def _coordinator(url: str | None) -> str:
    base = url or os.environ.get("AVIR_COORDINATOR")
    if not base:
        raise ValueError("no coordinator URL (pass coordinator_url or set AVIR_COORDINATOR)")
    return base


@mcp.tool()
def avir_network_health(coordinator_url: str | None = None) -> dict[str, Any]:
    """Check the AVIR coordinator is reachable and whether job submission is gated."""
    return _get(_coordinator(coordinator_url), "/health")


@mcp.tool()
def avir_list_participants(coordinator_url: str | None = None) -> dict[str, Any]:
    """List the registered verifier participants (the network directory)."""
    return _get(_coordinator(coordinator_url), "/verifiers")


@mcp.tool()
def avir_submit_verification(
    spec: dict[str, Any],
    subject_responses: dict[str, list[str]],
    min_verifiers: int = 3,
    require_distinct_orgs: bool = False,
    coordinator_url: str | None = None,
) -> dict[str, Any]:
    """Submit a blinded verification job to the external verifier network.

    spec: an AVIR specification dict containing rubric benchmarks.
    subject_responses: {benchmark_id: [answer strings]} to be judged by others.
    Returns the job_id to poll with avir_get_result.
    """
    token = os.environ.get("AVIR_COORDINATOR_TOKEN")
    job_id = submit_job(
        _coordinator(coordinator_url), spec=spec, subject_responses=subject_responses,
        min_verifiers=min_verifiers, require_distinct_orgs=require_distinct_orgs, token=token,
    )
    return {"job_id": job_id, "min_verifiers": min_verifiers}


@mcp.tool()
def avir_get_result(
    job_id: str,
    wait: bool = False,
    timeout_s: float = 300.0,
    coordinator_url: str | None = None,
) -> dict[str, Any]:
    """Get a job's external consensus, with every verdict signature re-verified.

    If wait=True, block (up to timeout_s) until quorum completes. The returned
    consensus is recomputed locally from signature-valid verdicts only.
    """
    base = _coordinator(coordinator_url)
    if wait:
        result = poll_result(base, job_id, timeout_s=timeout_s)
    else:
        result = _get(base, f"/jobs/{job_id}")
    checked = reverify_result(result)
    return {
        "job_id": job_id,
        "status": result.get("status"),
        "consensus": checked["consensus"],
        "invalid_signatures": checked["invalid_signatures"],
        "verdict_count": len(checked["valid_records"]),
    }


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
