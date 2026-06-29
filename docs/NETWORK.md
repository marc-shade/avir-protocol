# AVIR Verification Network

The network turns AVIR from a single-operator tool into a **distributed
verification service**: independent operators run verifier nodes that contribute
their own compute and AI providers, so the parties grading a claim are people
you do not control. That attacks the "an AI grades its own test" problem at the
*operator* level, not just the model level.

## Components

```
                          ┌─────────────────────────┐
   requester ──submit──▶  │      COORDINATOR        │  ◀──register── verifier A (Anthropic)
   (poll + re-verify  ◀── │  (job board, untrusted  │  ◀──pull/sign─ verifier B (ollama, node X)
    every signature)      │   for integrity)        │  ◀──pull/sign─ verifier C (your model)
                          └─────────────────────────┘
```

- **Coordinator** (`avir-coordinator`, a persistent FastAPI daemon) — queues
  blinded jobs and stores signed verdicts. It is **untrusted for integrity**: it
  verifies signatures but cannot forge them, and the requester re-checks every
  signature anyway.
- **Verifier node** (`avir-verifier`, stdlib-only client) — registers in the
  participant registry, pulls blinded jobs, judges them with the operator's own
  provider, signs the verdict with an Ed25519 key, and submits it. Provider
  credentials never leave the operator's machine.
- **Requester** (`avir-verify-remote`, or the MCP tools) — submits a job and
  polls the result, re-verifying each verdict signature locally.
- **MCP server** (`avir-mcp`) — exposes the requester operations as MCP tools so
  an agent/harness can request external verification programmatically.

## Participant registry

The coordinator keeps a persistent registry of participants (sqlite): public
key, fingerprint, organization, advertised models, `registered_at`,
`last_seen`, and `jobs_judged`. It is the directory participants use to find and
verify each other:

```
GET /verifiers            -> list all participants (sorted by jobs_judged)
GET /verifiers/{fp}       -> one participant by fingerprint or pubkey
```

## Trust model

- Every verdict is **Ed25519-signed** over the canonical bytes of
  `{job_id, verdicts}`. The signer's public key IS their network identity.
- The coordinator verifies signatures on submission and rejects forgeries
  (HTTP 400) and double-votes (HTTP 409).
- The requester (and anyone) **re-verifies every signature locally** and
  recomputes consensus from signature-valid verdicts only — the coordinator's
  aggregate is a convenience, never trusted.
- Consensus = majority PASS/FAIL per benchmark across **distinct verifier keys**;
  optionally across **distinct organizations**; the **requester's own key is
  excluded**. Quorum (`min_verifiers`) must be met or the verdict is
  `INCONCLUSIVE`.

## Running it

Coordinator (host):
```bash
pip install avir-protocol[network,crypto]
avir-coordinator --host 127.0.0.1 --port 8770 --db ~/.avir/coordinator.db
# expose deliberately: --host 0.0.0.0 and set AVIR_COORDINATOR_TOKEN to gate submission
```

Contribute a verifier node (any operator):
```bash
pip install avir-protocol[crypto]   # verifier client is stdlib + providers only
avir-verifier --coordinator http://HOST:8770 --provider claude-cli --org alice
# or any OpenAI-compatible endpoint you run:
avir-verifier --coordinator http://HOST:8770 --provider http \
              --endpoint http://localhost:11434 --model llama3 --org alice
```

Request external verification:
```bash
avir-verify-remote --coordinator http://HOST:8770 --spec spec.json \
                   --responses responses.json --min-verifiers 3 --require-distinct-orgs
```

From an agent (MCP):
```bash
pip install avir-protocol[network,mcp,crypto]
AVIR_COORDINATOR=http://HOST:8770 avir-mcp     # stdio MCP server
# tools: avir_network_health, avir_list_participants, avir_submit_verification, avir_get_result
```

## Security

- Default bind is `127.0.0.1`. Exposing the coordinator is an explicit
  `--host 0.0.0.0`; do so only with `AVIR_COORDINATOR_TOKEN` set to gate job
  submission.
- Identity private keys are stored 0600 and never transmitted.
- All DB access is parameterized; connections use a 30s busy timeout.

## Verified run (2026-06-29)

`examples/network_local_demo.py` against a live coordinator daemon, three
independent verifier nodes with distinct keys, organizations, models, and
machines:

| Org | Model | Machine | Verdict |
|-----|-------|---------|---------|
| deepseek | deepseek-v4-pro | fedora.local | PASS |
| deepreinforce | ornith-9b | ai-lab.local | PASS |
| zhipu | glm-5.1 | fedora.local | PASS |

Job auto-closed on quorum; all 3 verdict signatures re-verified locally (3
valid, 0 invalid); **NETWORK CONSENSUS: VERIFIED, 3 distinct verifiers, 3
distinct orgs, agreement 1.00**. This was a local-cluster stand-in for external
operators (same protocol, same signing, same registry) -- a true multi-operator
deployment is the same commands run by different people.

## Honest limits

- **Operator decentralization, not formal independent-lab replication.** The
  requester still authors the spec; the network supplies independent graders.
  That breaks single-operator self-grading and cancels cross-org bias, but it is
  not a credentialed third-party lab reproducing a result under controlled
  conditions.
- **Sybil / collusion is only partially mitigated.** One operator can run many
  nodes. Distinct-key and distinct-organization quorum thresholds raise the bar,
  but real defenses (stake, reputation, vetted identities, proof-of-distinct-
  hardware) are future work. Do not treat a high verifier count from unknown
  keys as trust on its own.
- **Coordinator availability is a single point** (not integrity — verdicts are
  signed). A federated/P2P coordinator is future work.
- **Verifier honesty is not enforced**, only attributed. A verifier can vote
  badly; the signature ties that vote to its key for reputation over time.
