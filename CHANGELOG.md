# Changelog

All notable changes to AVIR Protocol are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [Unreleased]

### Changed (attestation format)
- **Attestation signatures cover the full document.** `Attestation.sign()`
  signs `signing_payload()`: the canonical JSON (sorted keys, compact
  separators) of `to_dict()` without the `signature` field, and records
  `"scope": "document"` in the signature. Previously only
  `attestation_chain.chain_hash` was signed, so verdict, results, summary,
  system, verifier, environment, execution and verification level could be
  edited without affecting the signature. `verify_signature()` also rejects a
  loaded document whose fields differ from what the attestation serializes to
  (for example an edited `execution.duration_seconds` or an added key).
- **Legacy signatures fail closed.** A signature without `scope` (the old
  chain-hash-only format) makes `verify_signature()` return False, and
  `avir check` reports it as LEGACY and exits 1. Re-sign to upgrade. The
  network verdict signatures (`avir.network`) are unchanged.
- **Strict attestation parsing.** `Attestation.from_json()`/`load()` reject
  duplicate object keys and NaN/Infinity, so a file cannot carry a second
  value for a field that a different JSON reader would show while the
  signature verifies against the first.

### Fixed
- **CI installs what the tests import.** The `dev` extra now includes
  `cryptography`. `pip install -e ".[dev]"` (the CI install step) left it out,
  so collecting `test_attestation_signature.py`, `test_cli.py`,
  `test_network.py` and `test_tls.py` failed with `ModuleNotFoundError`.
  `dev` already carried `fastapi` and `httpx` for the network tests; the
  library still imports `cryptography` lazily or only from the network/TLS
  modules, so runtime dependencies are unchanged.
- **`avir` CLI works against the library APIs.** Every command crashed before
  (`Specification.version`, `VerificationLevel(2)`, `Attestation.timestamp`,
  `verify_signature(key)` did not exist).
  - `verify` runs `CrossVerifier` (double-blind, two or more `--provider`,
    `NAME=MODEL` form) and writes an attestation of the matrix; a signing
    failure is an error instead of a warning next to an unsigned file.
  - `check` pins the signature to `--public-key` (PEM, hex or base64); a file
    signed by any other key fails. Without a key it says the key is
    self-asserted. Unsigned and legacy-signed attestations fail.
  - `init` templates now pass `avir validate`; `init` no longer overwrites.
  - `compare` checks chain integrity and signatures before comparing.
  - `verify --model` is replaced by `--provider NAME=MODEL` (one model per
    provider). Removed flags whose value was never used or would be an
    unbacked claim: `verify --level` (the level is L3 by construction),
    `verify --container`, `validate --strict`, `compare --output`.

### Changed (network registry)
- **Quorum counts only registered verifiers.** `POST /jobs/{id}/verdicts` and
  `GET /jobs/next` refuse a key that is not in the coordinator's registry
  (HTTP 403). Before, any key with a valid signature counted, so fresh keys
  could fill `min_verifiers` without ever registering.
- **The registered org is the one recorded and counted.** The verdict record
  carries the org on file for the key, and `require_distinct_orgs` counts that.
  `verifier_org` in the request body (not covered by the signature) may be
  empty; a different value is refused with HTTP 400 instead of being stored.
  Before, the body value was stored as-is, so one party could claim several
  orgs.
- **Optional registration token.** With `AVIR_REGISTRATION_TOKEN` set on the
  coordinator, `POST /verifiers/register` requires it in
  `X-AVIR-Registration-Token` (401 otherwise). It is separate from the
  job-submission token. Unset keeps registration open, unchanged for a
  coordinator only trusted parties can reach; the coordinator warns at
  startup when exposed beyond localhost without it, and `/health` reports
  `registration_required_token`. The verifier reads the token from the
  `AVIR_REGISTRATION_TOKEN` environment variable; `deploy/setup_launchd.sh`
  takes `GATE_REGISTRATION=1` (coordinator) and `REGISTRATION_TOKEN_FILE`
  (verifier).
- Registration rejects a pubkey that is not a hex 32-byte Ed25519 key (400;
  it raised a server error before).
- The verifier daemon registers again when the coordinator answers 403 (for
  example after its registry was reset) instead of judging jobs it cannot
  submit.
- Wire format and verdict signatures are unchanged; verifiers on the previous
  release keep working against this coordinator while registration is open.

### Changed
- **Network quorum counts decisive votes only.** `aggregate_consensus` now
  requires, for every benchmark, `min_verifiers` distinct keys (and distinct
  orgs under `require_distinct_orgs`) voting PASS or FAIL. Previously any
  submission counted, so a job could report `quorum_met=True` with most
  verifiers abstaining. Each benchmark result carries `quorum_met` and
  `decisive_orgs`; the coordinator keeps such a job open until enough decisive
  votes arrive.

### Added
- **Job expiry.** The coordinator marks an open job `expired` once it is older
  than `--job-ttl` seconds (default 3600; 0 disables). Expired jobs are not
  served, late verdicts get HTTP 410, and the requester stops polling. Without
  it a job below decisive quorum stayed open forever and new verifiers worked
  through every stale job first. Verifiers log a refused request as such
  instead of "coordinator unreachable".
- **Verifier waits for the coordinator at startup.** Registration retried
  every `--poll-interval` when the coordinator is unreachable (daemon mode),
  instead of exiting and relying on the service manager's restart delay; at
  boot the SSH tunnel is often not up yet. `--once` still fails fast.
- Verifiers log the cause of every ABSTAIN; the requester prints decisive/total
  votes per benchmark and warns on abstentions.
- Coordinator `--no-access-log` flag.

### Fixed
- `pyyaml` declared as a dependency (clean installs failed on import).

## [1.2.1]

Transport encryption and deployment.

### Added
- **TLS for all client/coordinator traffic** (`avir.network.tls`). The
  coordinator accepts `--tls-cert/--tls-key` (or `AVIR_TLS_CERT/KEY`) and serves
  HTTPS; it warns when bound beyond localhost without TLS.
- **`avir-gen-cert`** -- generate a self-signed coordinator certificate (SAN per
  `--host`). Clients trust a self-signed coordinator via `AVIR_CA_BUNDLE`
  (real verification); `AVIR_INSECURE=1` disables verification (dev only) and
  prints a loud MITM warning.
- Verifier and requester clients use the configured SSL context for all calls.
- `deploy/setup_launchd.sh` + `deploy/README.md`: stand up a persistent,
  TLS-enabled, token-gated coordinator and verifier as launchd services
  (RunAtLoad + KeepAlive + reboot-survival).
- `tests/test_tls.py`. Verified end-to-end: HTTPS succeeds with the CA bundle,
  is rejected without it (`CERTIFICATE_VERIFY_FAILED`), and plain HTTP is refused
  on the TLS port. Full suite 95 passing.

### Security
- Coordinator warns on non-localhost bind without TLS and/or without
  `AVIR_COORDINATOR_TOKEN`. Certs/keys/tokens written `0600`.
- TLS encrypts all data in transit (jobs, responses, verdicts, registry). It
  does not hide content from the coordinator relay (verifiers must read it);
  verdicts remain Ed25519-signed independent of TLS.

## [1.2.0]

Distributed verification network: independent operators contribute their own
compute and AI providers as genuinely external verification sources.

### Added
- **`avir.network`** package and a verification network:
  - `avir-coordinator` -- a persistent FastAPI job-board daemon. Untrusted for
    integrity: it verifies and stores Ed25519-signed verdicts but cannot forge
    them. Rejects forged signatures (400) and double-votes (409).
  - `avir-verifier` -- a stdlib-only client. Registers in the participant
    registry, pulls blinded jobs, judges with the operator's own provider
    (CLI or any OpenAI-compatible endpoint), signs, and submits. Provider
    credentials never leave the operator's machine.
  - `avir-verify-remote` -- requester client that submits a job and
    re-verifies every verdict signature locally before trusting consensus.
- **Persistent participant registry** (sqlite): public key, fingerprint, org,
  advertised models, registered_at, last_seen, jobs_judged. `GET /verifiers`
  is the directory participants use to find and verify each other.
- **Ed25519 network identity** (`avir.network.keys`): keypair, sign/verify,
  fingerprint, 0600 on-disk identity files.
- **Consensus over distinct keys / distinct orgs**, with the requester's own
  key excluded and a quorum (`min_verifiers`) gate -> INCONCLUSIVE below quorum.
- **`avir-mcp`** -- an MCP stdio server (tools: `avir_network_health`,
  `avir_list_participants`, `avir_submit_verification`, `avir_get_result`) so an
  agent/harness can request external verification programmatically.
- New extras: `[network]` (fastapi+uvicorn), `[mcp]` (mcp). The verifier and
  requester clients are stdlib-only.
- `examples/network_local_demo.py` + `docs/NETWORK.md`. Verified live on the
  cluster: 3 independent nodes (deepseek/fedora, ornith/ai-lab, glm/fedora),
  distinct keys+orgs, consensus VERIFIED, all signatures re-verified.
- `tests/test_network.py` (13 tests): signing/verify, forgery + double-vote
  rejection, quorum/distinct-org/requester-exclusion consensus, token gate.

### Security
- Coordinator binds `127.0.0.1` by default; exposing it is an explicit
  `--host 0.0.0.0` and warns unless `AVIR_COORDINATOR_TOKEN` gates submission.
- Parameterized sqlite throughout; 30s busy timeout.

### Honest limits
- Operator-level decentralization, NOT formal independent-lab replication.
  Sybil/collusion is only partially mitigated (distinct-key/org thresholds);
  stake/reputation is future work. Coordinator availability (not integrity) is a
  single point. See docs/NETWORK.md.

## [1.1.0]

Real-model verification, cluster execution, and removal of simulated paths.

### Added
- **Rubric (model-judged) benchmarks** (`BenchmarkSpec.kind = "rubric"`). A
  subject model answers `prompt`; verifier models from *other* providers
  independently judge the blinded answer against `rubric`, returning
  PASS/FAIL/ABSTAIN. This is real cross-provider judgment that can disagree --
  the property that makes consensus meaningful. Numeric benchmarks are
  unchanged (deterministic arithmetic, where that is the correct check).
- **Real-model providers with no SDK dependency** (`avir/providers/cli.py`):
  - `ClaudeCLIProvider`, `CodexCLIProvider`, `GeminiCLIProvider` shell out to
    the headless CLIs (`claude --print`, `codex exec`, `gemini -p`). Each call
    is a separate OS process, so `IsolationLevel.PROCESS` is true by
    construction.
  - `HTTPModelProvider` (+ `fedora_ollama`, `ailab_model` factories) posts to
    an OpenAI-compatible `/v1/chat/completions` endpoint, i.e. a model running
    on a different physical machine, making the new `IsolationLevel.DISTRIBUTED`
    level true and auditable.
- **`IsolationLevel.DISTRIBUTED`** for separate-physical-node verification.
- **`VerificationCell.verifier_locus`** records where each verifier physically
  ran (`subprocess:...@host` or `node:host/model`), so the attested isolation
  level is auditable, not a bare label.
- Provider base gained `generate_response()` and `judge_output()` (the rubric
  primitives), plus an `execution_locus` property.
- `examples/cluster_cross_verify.py` and `examples/cluster_demonstrator_spec.json`:
  an end-to-end cross-provider, cross-node run that emits a signed attestation.
- `tests/test_rubric_cross_verification.py`: proves the verdict depends on the
  verifier identity and on response content (impossible under the old arithmetic
  path).

### Changed
- **Removed simulated benchmark execution.** The SDK providers
  (`ClaudeProvider`, `OpenAIProvider`, `GeminiProvider`) previously returned
  `target + random.uniform(...)` from `execute_benchmark`, i.e. fabricated
  numbers. They now ABSTAIN on numeric benchmarks (an LLM cannot physically
  measure throughput/latency) and implement real `generate_response` /
  `judge_output` via their SDKs (requires an API key; not exercised in CI --
  the tested real path is the CLI/cluster providers).
- `Specification` parser accepts rubric benchmarks (only `id` + `prompt` +
  `rubric` required; `target`/`unit`/`tolerance`/`methodology` optional) and
  reads `iterations` as an alias for `runs`.
- Corrected repository URLs from `marcshade` to `marc-shade` in `pyproject.toml`
  and the citation.

### Fixed
- A verification cell where the verifier abstained on **every** benchmark is now
  labeled `INCONCLUSIVE` instead of `FAILED`. Absence of a verdict is not
  evidence the subject failed. (Surfaced by a real cluster run where the codex
  CLI emitted no parseable `VERDICT:` marker.)

### Known issues
- The codex CLI's judgment output is not reliably parsed into a `VERDICT:`
  marker yet, so codex tends to ABSTAIN as a verifier. It still works as a
  subject. Improving codex judge-output parsing is a follow-up.

### Notes
- SDK provider rubric methods are real but not covered by CI (no keys). The
  CLI and cluster (HTTP) providers are the tested real-model path.
- Numeric-benchmark verification remains deterministic arithmetic by design; no
  model judgment is used or implied there.
- Verified end-to-end on the local cluster (5 providers, 3 processes + 2 GPU
  nodes): consensus VERIFIED, agreement 1.0, signed attestation verified. See
  `docs/REAL_MODEL_VERIFICATION.md`.

## [1.0.0]

Initial release: cross-provider double-blind verification matrix, BLAKE2b
salted blinding with canary leak detection, statistical consensus (Fleiss'
kappa / agreement), Ed25519 cryptographic attestation chains, JSON schemas,
and the specification/verdict/benchmark core.
