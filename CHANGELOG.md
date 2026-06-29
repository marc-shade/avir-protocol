# Changelog

All notable changes to AVIR Protocol are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

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
