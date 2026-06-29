# Contributing to AVIR Protocol

Thanks for your interest. AVIR aims to be a small, honest, well-tested
verification protocol. Contributions are welcome in these areas: new real-model
providers, benchmark types, isolation methods, and documentation.

## Ground rules

- **No simulated work presented as real.** A provider method named for a model
  call must make a model call (or honestly ABSTAIN). Do not return fabricated
  numbers, hardcoded verdicts, or placeholder data from a function whose name
  implies it did real work. (v1.1.0 removed exactly this from the SDK
  providers.)
- **Tests must exercise the claim.** Assert on the actual contract, not on type
  or truthiness. A test must fail when the behavior is broken.
- **Isolation claims must be real.** If you add an isolation level, it must
  correspond to an actual boundary (process, container, node, TEE) and record a
  `verifier_locus` that makes it auditable.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,crypto]"
pytest -q                      # full suite
ruff check avir/               # lint
```

## Adding a provider

Subclass `avir.providers.base.AVIRProvider` and implement at minimum:

- `provider_id`, `model`, `get_organization()` (organization matters: cross-org
  panels cancel more bias than same-org ones)
- `generate_response(prompt, *, system="")` -- the subject answer
- `judge_output(*, prompt, response, rubric, expected="")` -> `VerdictResponse`
  with verdict PASS / FAIL / ABSTAIN (never silently pass)
- `execution_locus` -- where it physically runs

See `avir/providers/cli.py` for the headless-CLI and HTTP/cluster reference
implementations (no vendor SDK required).

## Pull requests

- Keep changes focused; update `CHANGELOG.md`.
- Run `pytest` and `ruff` before submitting.
- New behavior needs a test that fails without it.
