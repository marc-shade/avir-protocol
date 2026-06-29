# Changelog

All notable changes to AVIR Protocol are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

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

### Notes
- SDK provider rubric methods are real but not covered by CI (no keys). The
  CLI and cluster (HTTP) providers are the tested real-model path.
- Numeric-benchmark verification remains deterministic arithmetic by design; no
  model judgment is used or implied there.

## [1.0.0]

Initial release: cross-provider double-blind verification matrix, BLAKE2b
salted blinding with canary leak detection, statistical consensus (Fleiss'
kappa / agreement), Ed25519 cryptographic attestation chains, JSON schemas,
and the specification/verdict/benchmark core.
