"""
Persistent store for the verification network (sqlite).

Holds three things:
- ``verifiers`` -- the participant REGISTRY: who is in the network, their public
  key, organization, advertised models, and activity (registered_at, last_seen,
  jobs_judged). This is the directory participants use to find and verify each
  other.
- ``jobs``      -- submitted verification jobs (blinded spec + subject responses).
- ``verdicts``  -- signed verdicts, one row per (job, verifier).

All access is parameterized; connections use a 30s busy timeout so concurrent
verifier submissions do not fail under contention.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .keys import fingerprint


_SCHEMA = """
CREATE TABLE IF NOT EXISTS verifiers (
    pubkey        TEXT PRIMARY KEY,
    fingerprint   TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT '',
    org           TEXT NOT NULL DEFAULT '',
    models        TEXT NOT NULL DEFAULT '[]',
    registered_at TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    jobs_judged   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id               TEXT PRIMARY KEY,
    spec                 TEXT NOT NULL,
    subject_responses    TEXT NOT NULL,
    min_verifiers        INTEGER NOT NULL DEFAULT 3,
    require_distinct_orgs INTEGER NOT NULL DEFAULT 0,
    requester_pubkey     TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'open',
    created_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    job_id           TEXT NOT NULL,
    verifier_pubkey  TEXT NOT NULL,
    verifier_org     TEXT NOT NULL DEFAULT '',
    verifier_model   TEXT NOT NULL DEFAULT '',
    verdicts         TEXT NOT NULL,
    signature        TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (job_id, verifier_pubkey)
);
"""


class Registry:
    """sqlite-backed participant/job/verdict store."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- participant registry ---
    def register_verifier(
        self,
        *,
        pubkey: str,
        label: str,
        org: str,
        models: list[str],
        now: str,
    ) -> dict[str, Any]:
        fp = fingerprint(pubkey)
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO verifiers (pubkey, fingerprint, label, org, models, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pubkey) DO UPDATE SET
                    label=excluded.label, org=excluded.org,
                    models=excluded.models, last_seen=excluded.last_seen
                """,
                (pubkey, fp, label, org, json.dumps(models), now, now),
            )
        return self.get_verifier(pubkey)  # type: ignore[return-value]

    def get_verifier(self, pubkey: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM verifiers WHERE pubkey=?", (pubkey,)).fetchone()
        return _verifier_row(row) if row else None

    def list_verifiers(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM verifiers ORDER BY jobs_judged DESC, registered_at ASC"
            ).fetchall()
        return [_verifier_row(r) for r in rows]

    def touch_verifier(self, pubkey: str, now: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE verifiers SET last_seen=? WHERE pubkey=?", (now, pubkey))

    def increment_judged(self, pubkey: str, now: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE verifiers SET jobs_judged=jobs_judged+1, last_seen=? WHERE pubkey=?",
                (now, pubkey),
            )

    # --- jobs ---
    def create_job(
        self,
        *,
        job_id: str,
        spec: dict[str, Any],
        subject_responses: dict[str, Any],
        min_verifiers: int,
        require_distinct_orgs: bool,
        requester_pubkey: str,
        now: str,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO jobs (job_id, spec, subject_responses, min_verifiers,
                                  require_distinct_orgs, requester_pubkey, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (
                    job_id,
                    json.dumps(spec),
                    json.dumps(subject_responses),
                    int(min_verifiers),
                    1 if require_distinct_orgs else 0,
                    requester_pubkey,
                    now,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _job_row(row) if row else None

    def next_open_job_for(self, verifier_pubkey: str) -> dict[str, Any] | None:
        """An open job this verifier has not judged and did not submit."""
        with self._conn() as c:
            row = c.execute(
                """
                SELECT * FROM jobs
                WHERE status='open'
                  AND requester_pubkey != ?
                  AND job_id NOT IN (SELECT job_id FROM verdicts WHERE verifier_pubkey=?)
                ORDER BY created_at ASC LIMIT 1
                """,
                (verifier_pubkey, verifier_pubkey),
            ).fetchone()
        return _job_row(row) if row else None

    def expire_open_jobs(self, created_before: str) -> int:
        """Mark open jobs created before ``created_before`` (ISO UTC) as expired."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status='expired' WHERE status='open' AND created_at < ?",
                (created_before,),
            )
            return cur.rowcount

    def set_job_status(self, job_id: str, status: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, job_id))

    # --- verdicts ---
    def add_verdict(
        self,
        *,
        job_id: str,
        verifier_pubkey: str,
        verifier_org: str,
        verifier_model: str,
        verdicts: list[dict[str, Any]],
        signature: str,
        now: str,
    ) -> bool:
        """Insert a verdict. Returns False if this verifier already voted."""
        with self._conn() as c:
            exists = c.execute(
                "SELECT 1 FROM verdicts WHERE job_id=? AND verifier_pubkey=?",
                (job_id, verifier_pubkey),
            ).fetchone()
            if exists:
                return False
            c.execute(
                """
                INSERT INTO verdicts (job_id, verifier_pubkey, verifier_org,
                                      verifier_model, verdicts, signature, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    verifier_pubkey,
                    verifier_org,
                    verifier_model,
                    json.dumps(verdicts),
                    signature,
                    now,
                ),
            )
        return True

    def list_verdicts(self, job_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM verdicts WHERE job_id=? ORDER BY created_at ASC", (job_id,)
            ).fetchall()
        return [_verdict_row(r) for r in rows]


def _verifier_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "pubkey": r["pubkey"],
        "fingerprint": r["fingerprint"],
        "label": r["label"],
        "org": r["org"],
        "models": json.loads(r["models"]),
        "registered_at": r["registered_at"],
        "last_seen": r["last_seen"],
        "jobs_judged": r["jobs_judged"],
    }


def _job_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": r["job_id"],
        "spec": json.loads(r["spec"]),
        "subject_responses": json.loads(r["subject_responses"]),
        "min_verifiers": r["min_verifiers"],
        "require_distinct_orgs": bool(r["require_distinct_orgs"]),
        "requester_pubkey": r["requester_pubkey"],
        "status": r["status"],
        "created_at": r["created_at"],
    }


def _verdict_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": r["job_id"],
        "verifier_pubkey": r["verifier_pubkey"],
        "verifier_org": r["verifier_org"],
        "verifier_model": r["verifier_model"],
        "verdicts": json.loads(r["verdicts"]),
        "signature": r["signature"],
        "created_at": r["created_at"],
    }
