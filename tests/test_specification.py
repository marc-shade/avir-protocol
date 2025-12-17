"""Tests for AVIR specification parsing and validation."""

import json
import pytest
from pathlib import Path
import tempfile

from avir.specification import Specification, BenchmarkSpec


class TestSpecification:
    """Test Specification class."""

    def test_from_dict_minimal(self):
        """Test creating specification from minimal dict."""
        data = {
            "version": "1.0.0",
            "metadata": {"name": "Test Spec"},
            "system": {"name": "test-system", "version": "1.0.0"},
            "benchmarks": [
                {"id": "test_benchmark", "target": 100, "tolerance": 0.2}
            ],
        }

        spec = Specification.from_dict(data)

        assert spec.version == "1.0.0"
        assert spec.metadata["name"] == "Test Spec"
        assert len(spec.benchmarks) == 1
        assert spec.benchmarks[0].id == "test_benchmark"

    def test_from_dict_full(self):
        """Test creating specification from full dict."""
        data = {
            "version": "1.0.0",
            "metadata": {
                "name": "Full Spec",
                "description": "A complete specification",
                "author": "Test Author",
            },
            "system": {
                "name": "test-system",
                "version": "2.0.0",
                "description": "Test system description",
            },
            "benchmarks": [
                {
                    "id": "throughput",
                    "name": "Throughput Test",
                    "target": 100,
                    "unit": "ops/s",
                    "tolerance": 0.2,
                    "iterations": 5,
                },
                {
                    "id": "latency",
                    "name": "Latency Test",
                    "target": 50,
                    "unit": "ms",
                    "tolerance": 0.25,
                    "lower_is_better": True,
                },
            ],
            "capabilities": [
                {"name": "feature_a", "required": True},
                {"name": "feature_b", "required": False},
            ],
        }

        spec = Specification.from_dict(data)

        assert len(spec.benchmarks) == 2
        assert spec.benchmarks[1].lower_is_better is True
        assert len(spec.capabilities) == 2

    def test_from_file(self):
        """Test loading specification from file."""
        data = {
            "version": "1.0.0",
            "metadata": {"name": "File Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "benchmarks": [
                {"id": "test", "target": 100, "tolerance": 0.2}
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            spec = Specification.from_file(Path(f.name))

            assert spec.metadata["name"] == "File Spec"

    def test_validate_valid_spec(self):
        """Test validation of valid specification."""
        data = {
            "version": "1.0.0",
            "metadata": {"name": "Valid Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "benchmarks": [
                {"id": "test", "target": 100, "tolerance": 0.2}
            ],
        }

        spec = Specification.from_dict(data)
        errors = spec.validate()

        assert len(errors) == 0

    def test_validate_missing_benchmarks(self):
        """Test validation catches missing benchmarks."""
        data = {
            "version": "1.0.0",
            "metadata": {"name": "Invalid Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "benchmarks": [],
        }

        spec = Specification.from_dict(data)
        errors = spec.validate()

        assert len(errors) > 0
        assert any("benchmark" in e.lower() for e in errors)

    def test_hash_deterministic(self):
        """Test specification hash is deterministic."""
        data = {
            "version": "1.0.0",
            "metadata": {"name": "Hash Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "benchmarks": [
                {"id": "test", "target": 100, "tolerance": 0.2}
            ],
        }

        spec1 = Specification.from_dict(data)
        spec2 = Specification.from_dict(data)

        assert spec1.hash == spec2.hash


class TestBenchmarkSpec:
    """Test BenchmarkSpec class."""

    def test_from_dict_minimal(self):
        """Test creating benchmark from minimal dict."""
        data = {"id": "test", "target": 100, "tolerance": 0.2}
        bm = BenchmarkSpec.from_dict(data)

        assert bm.id == "test"
        assert bm.target == 100
        assert bm.tolerance == 0.2
        assert bm.lower_is_better is False

    def test_from_dict_full(self):
        """Test creating benchmark from full dict."""
        data = {
            "id": "latency",
            "name": "Response Latency",
            "description": "Measures response time",
            "target": 50,
            "unit": "ms",
            "tolerance": 0.25,
            "lower_is_better": True,
            "iterations": 10,
        }

        bm = BenchmarkSpec.from_dict(data)

        assert bm.id == "latency"
        assert bm.name == "Response Latency"
        assert bm.lower_is_better is True
        assert bm.iterations == 10

    def test_validate_valid(self):
        """Test validation of valid benchmark."""
        data = {"id": "test", "target": 100, "tolerance": 0.2}
        bm = BenchmarkSpec.from_dict(data)
        errors = bm.validate()

        assert len(errors) == 0

    def test_validate_invalid_tolerance(self):
        """Test validation catches invalid tolerance."""
        data = {"id": "test", "target": 100, "tolerance": 1.5}
        bm = BenchmarkSpec.from_dict(data)
        errors = bm.validate()

        assert len(errors) > 0
        assert any("tolerance" in e.lower() for e in errors)
