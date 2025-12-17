"""Tests for AVIR cross-provider verification."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from avir.cross_verification import (
    BlindingMode,
    IsolationLevel,
    BlindedContext,
    VerificationCell,
    CrossVerificationMatrix,
    CrossVerifier,
)
from avir.specification import Specification
from avir.verdict import VerdictLevel
from avir.providers.base import AVIRProvider


class MockProvider(AVIRProvider):
    """Mock provider for testing."""

    def __init__(self, provider_id: str, pass_rate: float = 0.8):
        self._provider_id = provider_id
        self._pass_rate = pass_rate

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model(self) -> str:
        return f"{self._provider_id}-model"

    async def initialize(self, config):
        pass

    async def execute_benchmark(self, benchmark_spec, context):
        import random
        target = benchmark_spec.get("target", 100)
        tolerance = benchmark_spec.get("tolerance", 0.2)
        # Simulate pass/fail based on pass_rate
        if random.random() < self._pass_rate:
            return target  # Pass
        else:
            return target * 2  # Fail (outside tolerance)

    async def analyze_results(self, results):
        return {"analysis": "mock analysis"}


class TestBlindedContext:
    """Test BlindedContext class."""

    def test_create_blinded_context(self):
        """Test creating a blinded context."""
        spec_data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test", "category": "performance", "benchmarks": ["test_bm"]}
            ],
            "benchmarks": [
                {"id": "test_bm", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }
        spec = Specification.from_dict(spec_data)

        blinded = BlindedContext.create(spec)

        assert blinded.context_id is not None
        assert len(blinded.context_id) == 32  # 16 bytes = 32 hex chars
        assert blinded.specification_hash == spec.hash
        assert len(blinded.benchmark_specs) == 1
        assert blinded.nonce is not None

    def test_benchmark_ids_anonymized(self):
        """Test that benchmark IDs are anonymized."""
        spec_data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test", "category": "performance", "benchmarks": ["original_id"]}
            ],
            "benchmarks": [
                {"id": "original_id", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }
        spec = Specification.from_dict(spec_data)

        blinded = BlindedContext.create(spec)

        # Original ID should not appear
        assert blinded.benchmark_specs[0]["id"] != "original_id"
        # Should be a hash (8 bytes = 16 hex chars)
        assert len(blinded.benchmark_specs[0]["id"]) == 16


class TestVerificationCell:
    """Test VerificationCell class."""

    def test_is_self_verification(self):
        """Test self-verification detection."""
        cell_self = VerificationCell(
            verifier_id="claude",
            subject_id="claude",
            verdict=VerdictLevel.VERIFIED,
            pass_rate=1.0,
            results=[],
            execution_time=1.0,
            context_hash="abc123",
        )

        cell_cross = VerificationCell(
            verifier_id="claude",
            subject_id="openai",
            verdict=VerdictLevel.VERIFIED,
            pass_rate=1.0,
            results=[],
            execution_time=1.0,
            context_hash="abc123",
        )

        assert cell_self.is_self_verification is True
        assert cell_cross.is_self_verification is False


class TestCrossVerificationMatrix:
    """Test CrossVerificationMatrix class."""

    def test_consensus_unanimous_verified(self):
        """Test unanimous verified consensus."""
        matrix = CrossVerificationMatrix(
            providers=["claude", "openai", "gemini"]
        )

        # Add cross-verification cells (excluding self-verification)
        for verifier in ["claude", "openai", "gemini"]:
            for subject in ["claude", "openai", "gemini"]:
                if verifier != subject:
                    matrix.add_cell(VerificationCell(
                        verifier_id=verifier,
                        subject_id=subject,
                        verdict=VerdictLevel.VERIFIED,
                        pass_rate=1.0,
                        results=[],
                        execution_time=1.0,
                        context_hash="abc",
                    ))

        assert matrix.consensus_verdict == VerdictLevel.VERIFIED
        assert matrix.agreement_score == 1.0

    def test_consensus_majority_failed(self):
        """Test majority failed consensus."""
        matrix = CrossVerificationMatrix(
            providers=["claude", "openai"]
        )

        # Claude verifies OpenAI: FAILED
        matrix.add_cell(VerificationCell(
            verifier_id="claude",
            subject_id="openai",
            verdict=VerdictLevel.FAILED,
            pass_rate=0.3,
            results=[],
            execution_time=1.0,
            context_hash="abc",
        ))

        # OpenAI verifies Claude: FAILED
        matrix.add_cell(VerificationCell(
            verifier_id="openai",
            subject_id="claude",
            verdict=VerdictLevel.FAILED,
            pass_rate=0.3,
            results=[],
            execution_time=1.0,
            context_hash="abc",
        ))

        assert matrix.consensus_verdict == VerdictLevel.FAILED

    def test_consensus_mixed_results(self):
        """Test mixed results consensus."""
        matrix = CrossVerificationMatrix(
            providers=["claude", "openai", "gemini"]
        )

        verdicts = [
            ("claude", "openai", VerdictLevel.VERIFIED),
            ("claude", "gemini", VerdictLevel.VERIFIED),
            ("openai", "claude", VerdictLevel.PARTIAL),
            ("openai", "gemini", VerdictLevel.VERIFIED),
            ("gemini", "claude", VerdictLevel.VERIFIED),
            ("gemini", "openai", VerdictLevel.PARTIAL),
        ]

        for verifier, subject, verdict in verdicts:
            matrix.add_cell(VerificationCell(
                verifier_id=verifier,
                subject_id=subject,
                verdict=verdict,
                pass_rate=0.7 if verdict == VerdictLevel.PARTIAL else 1.0,
                results=[],
                execution_time=1.0,
                context_hash="abc",
            ))

        # Majority are VERIFIED, so should be VERIFIED
        assert matrix.consensus_verdict == VerdictLevel.VERIFIED
        # Agreement is 4/6 = 0.67
        assert 0.6 < matrix.agreement_score < 0.7

    def test_cross_verifications_excludes_self(self):
        """Test that cross_verifications excludes self-verification."""
        matrix = CrossVerificationMatrix(
            providers=["claude", "openai"]
        )

        # Add self-verification
        matrix.add_cell(VerificationCell(
            verifier_id="claude",
            subject_id="claude",
            verdict=VerdictLevel.VERIFIED,
            pass_rate=1.0,
            results=[],
            execution_time=1.0,
            context_hash="abc",
        ))

        # Add cross-verification
        matrix.add_cell(VerificationCell(
            verifier_id="claude",
            subject_id="openai",
            verdict=VerdictLevel.VERIFIED,
            pass_rate=1.0,
            results=[],
            execution_time=1.0,
            context_hash="abc",
        ))

        cross = matrix.cross_verifications

        assert len(cross) == 1
        assert cross[0].verifier_id == "claude"
        assert cross[0].subject_id == "openai"

    def test_to_dict(self):
        """Test matrix serialization."""
        matrix = CrossVerificationMatrix(
            providers=["claude", "openai"],
            blinding_mode=BlindingMode.DOUBLE_BLIND,
        )

        matrix.add_cell(VerificationCell(
            verifier_id="claude",
            subject_id="openai",
            verdict=VerdictLevel.VERIFIED,
            pass_rate=1.0,
            results=[],
            execution_time=1.0,
            context_hash="abc",
        ))

        data = matrix.to_dict()

        assert "providers" in data
        assert "consensus" in data
        assert "cells" in data
        assert data["blinding_mode"] == "double_blind"


class TestCrossVerifier:
    """Test CrossVerifier class."""

    @pytest.mark.asyncio
    async def test_verify_minimum_providers(self):
        """Test that verification requires minimum providers."""
        providers = [MockProvider("claude")]

        verifier = CrossVerifier(
            providers=providers,
            blinding_mode=BlindingMode.DOUBLE_BLIND,
        )

        spec_data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test benchmark", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }
        spec = Specification.from_dict(spec_data)

        with pytest.raises(ValueError, match="at least 2"):
            await verifier.verify(spec, require_minimum_providers=2)

    @pytest.mark.asyncio
    async def test_verify_cross_provider(self):
        """Test cross-provider verification."""
        providers = [
            MockProvider("claude", pass_rate=0.9),
            MockProvider("openai", pass_rate=0.9),
        ]

        verifier = CrossVerifier(
            providers=providers,
            blinding_mode=BlindingMode.DOUBLE_BLIND,
            iterations=3,
        )

        spec_data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test benchmark", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }
        spec = Specification.from_dict(spec_data)

        matrix = await verifier.verify(spec)

        assert len(matrix.providers) == 2
        # Should have 4 cells: 2 self + 2 cross
        assert len(matrix.cells) == 4
        # Cross verifications should be 2
        assert len(matrix.cross_verifications) == 2
