"""
Tests for rubric (model-judged) cross-verification.

These tests prove the property that distinguishes real cross-provider
verification from the old deterministic-arithmetic path: the verdict for a
given subject output depends on WHICH verifier judged it and on the CONTENT of
the response. Under arithmetic tolerance checking that is impossible (every
verifier would compute the same number), so these assertions would fail on the
pre-rubric implementation.
"""

from typing import Callable

import pytest

from avir.providers.base import AVIRProvider, VerdictResponse
from avir.providers.cli import parse_judgment, build_judge_prompt
from avir.specification import Specification
from avir.cross_verification import CrossVerifier, BlindingMode, IsolationLevel


class FakeProvider(AVIRProvider):
    """Deterministic provider: fixed answer + configurable judge policy."""

    def __init__(self, pid: str, answer: str, judge: Callable[[str], str]):
        self._pid = pid
        self._answer = answer
        self._judge = judge

    @property
    def provider_id(self) -> str:
        return self._pid

    @property
    def model(self) -> str:
        return f"fake-{self._pid}"

    @property
    def execution_locus(self) -> str:
        return f"fake:{self._pid}"

    async def initialize(self, config):
        del config

    async def generate_response(self, prompt: str, *, system: str = "") -> str:
        del prompt, system
        return self._answer

    async def judge_output(self, *, prompt, response, rubric, expected=""):
        del prompt, rubric, expected
        return VerdictResponse(value=None, verdict=self._judge(response))

    async def execute_benchmark(self, benchmark_spec, context):
        del benchmark_spec, context
        return VerdictResponse(value=None, verdict="ABSTAIN")

    async def analyze_results(self, results):
        return {"provider": self._pid, "count": len(results)}


def _rubric_spec() -> Specification:
    return Specification.from_dict({
        "avir_version": "1.0.0",
        "system": {"name": "rubric-test", "version": "1.0.0"},
        "capabilities": [],
        "benchmarks": [{
            "id": "q1",
            "kind": "rubric",
            "description": "demo rubric benchmark",
            "prompt": "Answer the question.",
            "rubric": "Pass iff the answer contains the token CORRECT.",
        }],
    })


def _providers():
    # alpha: judges everything PASS; beta: everything FAIL;
    # gamma: PASS iff the response contains 'CORRECT' (content-dependent)
    alpha = FakeProvider("alpha", "alpha says CORRECT", lambda r: "PASS")
    beta = FakeProvider("beta", "beta says wrong", lambda r: "FAIL")
    gamma = FakeProvider("gamma", "gamma says CORRECT",
                         lambda r: "PASS" if "CORRECT" in r else "FAIL")
    return alpha, beta, gamma


@pytest.mark.asyncio
async def test_verdict_depends_on_verifier_identity():
    """Same subject, different verifiers -> different verdicts (real judgment)."""
    alpha, beta, gamma = _providers()
    cv = CrossVerifier(
        providers=[alpha, beta, gamma],
        blinding_mode=BlindingMode.DOUBLE_BLIND,
        isolation_level=IsolationLevel.PROCESS,
        iterations=1,
    )
    matrix = await cv.verify(_rubric_spec())

    # alpha (always-PASS) and beta (always-FAIL) judging the SAME subject gamma
    cell_alpha = matrix.get_cell("alpha", "gamma")
    cell_beta = matrix.get_cell("beta", "gamma")
    assert cell_alpha is not None and cell_beta is not None
    assert cell_alpha.pass_rate == 1.0
    assert cell_beta.pass_rate == 0.0
    # The crux: verdict changes with the verifier. Impossible under arithmetic.
    assert cell_alpha.pass_rate != cell_beta.pass_rate


@pytest.mark.asyncio
async def test_verdict_depends_on_response_content():
    """One verifier (gamma) returns different verdicts based on content read."""
    alpha, beta, gamma = _providers()
    cv = CrossVerifier(providers=[alpha, beta, gamma], iterations=1)
    matrix = await cv.verify(_rubric_spec())

    # gamma judges alpha's answer (contains CORRECT) -> PASS
    # gamma judges beta's answer  (no CORRECT)       -> FAIL
    cell_on_alpha = matrix.get_cell("gamma", "alpha")
    cell_on_beta = matrix.get_cell("gamma", "beta")
    assert cell_on_alpha.pass_rate == 1.0
    assert cell_on_beta.pass_rate == 0.0


@pytest.mark.asyncio
async def test_self_verification_excluded_and_locus_recorded():
    alpha, beta, gamma = _providers()
    cv = CrossVerifier(providers=[alpha, beta, gamma], iterations=1)
    matrix = await cv.verify(_rubric_spec())

    # Self cells exist but are flagged for exclusion from consensus
    self_cell = matrix.get_cell("alpha", "alpha")
    assert self_cell is not None and self_cell.is_self_verification
    cross = [c for c in matrix.cells.values() if not c.is_self_verification]
    assert len(cross) == 6  # 3x3 minus 3 diagonal
    # Real isolation locus is recorded per cell
    assert matrix.get_cell("alpha", "beta").verifier_locus == "fake:alpha"


def test_parse_judgment_markers():
    assert parse_judgment("reasoning...\nCONFIDENCE: 0.9\nVERDICT: PASS").verdict == "PASS"
    assert parse_judgment("VERDICT: FAIL").verdict == "FAIL"
    assert parse_judgment("VERDICT: ABSTAIN").verdict == "ABSTAIN"
    # No marker -> ABSTAIN (never a silent pass)
    assert parse_judgment("I think it is fine").verdict == "ABSTAIN"
    assert parse_judgment("").verdict == "ABSTAIN"
    # Confidence captured
    v = parse_judgment("CONFIDENCE: 0.75\nVERDICT: PASS")
    assert v.confidence == 0.75


def test_build_judge_prompt_blind_and_structured():
    p = build_judge_prompt("task here", "the response", "must be correct", "expected x")
    assert "TASK" in p and "RESPONSE" in p and "RUBRIC" in p
    assert "VERDICT: PASS | FAIL | ABSTAIN" in p
    # The verifier is told it does not know the provider (blinding)
    assert "do not know which provider" in p.lower()
