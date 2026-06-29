# Real-Model Verification (rubric benchmarks + cluster providers)

AVIR v1.1.0 adds verification that actually invokes models, with isolation that
is true rather than labeled. This document explains the two kinds of benchmark,
the real-model providers, and how to run a cross-provider, cross-node check.

## Two kinds of benchmark

| Kind | What is measured | How it is verified |
|------|------------------|--------------------|
| `numeric` (default) | A measured value (throughput, latency, ...) | Deterministic arithmetic: `target` within `tolerance`. No model judgment is used or implied -- that is the correct check for a number. |
| `rubric` | A model's free-text answer to `prompt` | A *verifier model from another provider* reads the blinded answer and decides PASS/FAIL/ABSTAIN against `rubric`. Real judgment that can disagree. |

The rubric path is what makes "no AI grades its own test" meaningful: the verdict
depends on which verifier judged and on the content of the answer, not on a
formula that every verifier would compute identically.

```json
{
  "id": "bat_and_ball",
  "kind": "rubric",
  "prompt": "A bat and a ball cost $1.10 ... how much does the ball cost?",
  "rubric": "PASS iff the final answer is 5 cents; 10 cents or anything else is FAIL.",
  "expected": "5 cents (bat $1.05, ball $0.05)."
}
```

## Real-model providers (no SDK required)

`avir/providers/cli.py`:

- **`ClaudeCLIProvider` / `CodexCLIProvider` / `GeminiCLIProvider`** shell out to
  the headless CLIs. Each call is a separate OS process, so
  `IsolationLevel.PROCESS` is true by construction. No API key handling and no
  vendor SDK import -- the CLI owns auth.
- **`HTTPModelProvider`** (and the `fedora_ollama` / `ailab_model` factories)
  post to an OpenAI-compatible `/v1/chat/completions` endpoint. The model runs
  on a *different physical machine*, making `IsolationLevel.DISTRIBUTED` true.
  Reasoning models that leave `content` empty (text in `reasoning`/`thinking`)
  are handled.

The SDK providers (`ClaudeProvider`, `OpenAIProvider`, `GeminiProvider`) also
implement `generate_response` / `judge_output` via their SDKs, but require an API
key and are not exercised in CI; the CLI/cluster providers are the tested path.

### Auditable isolation

Every `VerificationCell` records `verifier_locus` -- e.g.
`subprocess:claude@host` or `node:fedora.local/deepseek-v4-pro:cloud` -- so the
isolation level in the attestation can be checked against where work actually
ran, instead of being trusted as a bare label.

## Running a cross-provider, cross-node check

```bash
pip install -e ".[crypto]"
python examples/cluster_cross_verify.py examples/cluster_demonstrator_spec.json out.json
```

This builds a panel of genuinely different providers across separate processes
and physical nodes, runs the double-blind matrix (each provider judges every
other; self-cells excluded from consensus), and writes a signed Ed25519
attestation. Programmatically:

```python
from avir import Specification, BlindingMode, IsolationLevel
from avir.cross_verification import CrossVerifier
from avir.providers import ClaudeCLIProvider, GeminiCLIProvider, fedora_ollama, ailab_model

spec = Specification.from_file("spec.json")  # rubric benchmarks
cv = CrossVerifier(
    providers=[
        ClaudeCLIProvider(),                                  # Anthropic, local process
        GeminiCLIProvider(),                                  # Google, local process
        fedora_ollama("deepseek-v4-pro:cloud", organization="deepseek"),  # node
        ailab_model("ornith-9b", organization="deepreinforce"),           # node
    ],
    blinding_mode=BlindingMode.DOUBLE_BLIND,
    isolation_level=IsolationLevel.DISTRIBUTED,
    iterations=1,
)
matrix = await cv.verify(spec)
print(matrix.consensus_verdict, matrix.agreement_score)
```

## Honest scope

Cross-provider verification breaks the *single-model* self-grading loop and
cancels some single-org bias. It is **not** a substitute for independent
third-party replication: the operator still chooses the panel and the spec.
Diversity of providers (and organizations) is what gives the result weight; a
panel of same-lineage models shares blind spots. Use the `organization` field
and mix families.
