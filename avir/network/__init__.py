"""
AVIR verification network.

A distributed, double-blind verification network where independent operators
contribute their OWN compute and AI providers as external verification sources.

Components:
- ``keys``        -- Ed25519 identity (keypair, sign, verify, fingerprint)
- ``protocol``    -- verdict signing payloads and consensus aggregation
- ``registry``    -- persistent participant + job + verdict store (sqlite)
- ``coordinator`` -- FastAPI job board (queues work, aggregates signed verdicts)
- ``verifier``    -- stdlib client: pull jobs, judge locally, sign, submit
- ``requester``   -- stdlib client: submit a job, poll the verified result

Trust model: the coordinator only queues and aggregates. Every verdict is
Ed25519-signed by the verifier that produced it, so the coordinator cannot forge
or alter verdicts. Consensus is computed over distinct verifier keys (and
optionally distinct organizations); the requester's own key is excluded.
"""

from .keys import (
    generate_keypair,
    load_or_create_identity,
    fingerprint,
    sign,
    verify,
    canonical,
)
from .protocol import (
    sign_verdict,
    verify_verdict,
    aggregate_consensus,
)

__all__ = [
    "generate_keypair",
    "load_or_create_identity",
    "fingerprint",
    "sign",
    "verify",
    "canonical",
    "sign_verdict",
    "verify_verdict",
    "aggregate_consensus",
]
