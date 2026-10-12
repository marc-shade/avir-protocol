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
  `INCONCLUSIVE`. Quorum counts **decisive votes only**: every benchmark needs
  `min_verifiers` distinct keys (and distinct orgs, when required) voting PASS
  or FAIL on it; an ABSTAIN never counts. A job below quorum stays open for
  new verifiers until `--job-ttl` (default 3600 s) passes, then it is marked
  `expired`: no longer served, late verdicts get HTTP 410, and its consensus
  reads `INCONCLUSIVE`.
- **Only registered verifiers count.** `GET /jobs/next` and
  `POST /jobs/{id}/verdicts` refuse a key with no registry entry (HTTP 403),
  so a validly signed verdict from a fresh key never reaches quorum. The
  verifier client registers at startup and registers again when it gets a 403
  (for example after the coordinator's database was reset).
- **The org that counts is the registered org.** `verifier_org` in a verdict
  submission is outside the signature, so the coordinator records the org on
  file for the key, and that is what `require_distinct_orgs` counts. The body
  field may be empty; a value that differs from the registered org is refused
  (HTTP 400) rather than silently replaced, so a misconfigured or spoofing
  client finds out.
- **The registry is the coordinator's word.** Distinct-org quorum rests on the
  registry, which the requester cannot check from signatures alone: its local
  recomputation confirms each signature and that the keys are distinct, and
  takes each record's org from the coordinator. Registration is open unless
  `AVIR_REGISTRATION_TOKEN` is set on the coordinator; with it set, only
  parties holding that token (sent as `X-AVIR-Registration-Token`, read by the
  verifier from the `AVIR_REGISTRATION_TOKEN` environment variable) can add or
  change registry entries. Without it, anyone who can reach the coordinator
  can register many keys under many org labels, so an open registry is only as
  strong as the network boundary in front of the coordinator (fine for
  127.0.0.1 behind SSH tunnels; not for an exposed port). The registration
  token is separate from the job-submission token so verifier nodes never hold
  the right to submit jobs.

## Running it

Coordinator (host):
```bash
pip install avir-protocol[network,crypto]
avir-coordinator --host 127.0.0.1 --port 8770 --db ~/.avir/coordinator.db
# expose deliberately: --host 0.0.0.0 and set AVIR_COORDINATOR_TOKEN to gate
# submission and AVIR_REGISTRATION_TOKEN to gate verifier registration
```

Contribute a verifier node (any operator):
```bash
pip install avir-protocol[crypto]   # verifier client is stdlib + providers only
avir-verifier --coordinator http://HOST:8770 --provider claude-cli --org alice
# or any OpenAI-compatible endpoint you run:
avir-verifier --coordinator http://HOST:8770 --provider http \
              --endpoint http://localhost:11434 --model llama3 --org alice
# if the coordinator gates registration, export AVIR_REGISTRATION_TOKEN first
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
  submission and `AVIR_REGISTRATION_TOKEN` set to gate verifier registration.
  The coordinator prints a warning at startup for each one missing while
  exposed. `GET /health` reports both (`auth_required`,
  `registration_required_token`).
- Registration does not prove possession of the private key, and a
  re-registration of an existing key replaces its label, org and models. With a
  registration token set, that power is limited to token holders.
- Identity private keys are stored 0600 and never transmitted.
- All DB access is parameterized; connections use a 30s busy timeout.

## Encryption (TLS)

All client/coordinator traffic travels over HTTPS/TLS. Generate a certificate
and run the coordinator with it:

```bash
avir-gen-cert --host coordinator.example.com --host 100.x.y.z   # SAN entries
avir-coordinator --host 0.0.0.0 --port 8770 \
                 --tls-cert ~/.avir/coordinator.crt --tls-key ~/.avir/coordinator.key
```

Clients trust a self-signed coordinator by pointing at its certificate (this is
real verification, not a bypass):

```bash
export AVIR_CA_BUNDLE=/path/to/coordinator.crt
avir-verifier --coordinator https://coordinator.example.com:8770 --provider claude-cli --org alice
```

With a publicly-issued certificate (Let's Encrypt, etc.), clients need nothing
extra. `AVIR_INSECURE=1` disables verification and is for throwaway local dev
only -- it enables MITM and prints a loud warning; prefer `AVIR_CA_BUNDLE`.

The coordinator warns if bound beyond localhost without TLS. Verified end-to-end:
HTTPS succeeds with the CA bundle, is correctly rejected without it
(`CERTIFICATE_VERIFY_FAILED`), and plain HTTP is refused on the TLS port.

Scope: TLS encrypts everything in transit (jobs, responses, verdicts, registry).
It does not hide job content from the coordinator itself -- the coordinator is a
relay and verifiers must read the prompt/response to judge them, and the set of
verifiers is unknown at submission time, so end-to-end encryption to specific
verifiers is not possible. Verdicts are additionally Ed25519-signed.

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
  keys as trust on its own. Org labels are whatever the registering party
  declared; a registration token limits who may declare them, it does not
  check that they are true.
- **Coordinator availability is a single point** (not integrity — verdicts are
  signed). A federated/P2P coordinator is future work.
- **Verifier honesty is not enforced**, only attributed. A verifier can vote
  badly; the signature ties that vote to its key for reputation over time.
