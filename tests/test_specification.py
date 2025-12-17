"""Tests for AVIR specification parsing and validation."""

import json
import pytest
from pathlib import Path
import tempfile

from avir.specification import Specification, BenchmarkSpec, Capability, SystemInfo


class TestSpecification:
    """Test Specification class."""

    def test_from_dict_minimal(self):
        """Test creating specification from minimal dict."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Test Spec"},
            "system": {"name": "test-system", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["test_benchmark"]}
            ],
            "benchmarks": [
                {"id": "test_benchmark", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }

        spec = Specification.from_dict(data)

        assert spec.avir_version == "1.0.0"
        assert spec.metadata["name"] == "Test Spec"
        assert len(spec.benchmarks) == 1
        assert spec.benchmarks[0].id == "test_benchmark"

    def test_from_dict_full(self):
        """Test creating specification from full dict."""
        data = {
            "avir_version": "1.0.0",
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
            "capabilities": [
                {"name": "throughput_cap", "description": "Throughput capability", "category": "performance", "benchmarks": ["throughput", "latency"]},
            ],
            "benchmarks": [
                {
                    "id": "throughput",
                    "description": "Throughput Test",
                    "methodology": "measure",
                    "target": 100,
                    "unit": "ops/s",
                    "tolerance": 0.2,
                    "runs": 5,
                },
                {
                    "id": "latency",
                    "description": "Latency Test",
                    "methodology": "measure",
                    "target": 50,
                    "unit": "ms",
                    "tolerance": 0.25,
                    "lower_is_better": True,
                },
            ],
        }

        spec = Specification.from_dict(data)

        assert len(spec.benchmarks) == 2
        assert spec.benchmarks[1].lower_is_better is True
        assert len(spec.capabilities) == 1

    def test_from_file(self):
        """Test loading specification from file."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "File Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
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
            "avir_version": "1.0.0",
            "metadata": {"name": "Valid Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }

        spec = Specification.from_dict(data)
        errors = spec.validate()

        assert len(errors) == 0

    def test_validate_invalid_tolerance(self):
        """Test validation catches invalid tolerance."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Invalid Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 1.5}
            ],
        }

        spec = Specification.from_dict(data)
        errors = spec.validate()

        assert len(errors) > 0
        assert any("tolerance" in e.lower() for e in errors)

    def test_validate_invalid_benchmark_reference(self):
        """Test validation catches capability referencing non-existent benchmark."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Invalid Spec"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["nonexistent"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }

        spec = Specification.from_dict(data)
        errors = spec.validate()

        assert len(errors) > 0
        assert any("nonexistent" in e for e in errors)

    def test_hash_deterministic(self):
        """Test specification hash is deterministic."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Hash Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }

        spec1 = Specification.from_dict(data)
        spec2 = Specification.from_dict(data)

        assert spec1.hash == spec2.hash

    def test_to_dict_roundtrip(self):
        """Test converting to dict and back."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Roundtrip Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["test"]}
            ],
            "benchmarks": [
                {"id": "test", "description": "Test benchmark", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2}
            ],
        }

        spec1 = Specification.from_dict(data)
        exported = spec1.to_dict()
        spec2 = Specification.from_dict(exported)

        assert spec1.hash == spec2.hash

    def test_get_benchmark(self):
        """Test getting benchmark by ID."""
        data = {
            "avir_version": "1.0.0",
            "metadata": {"name": "Test"},
            "system": {"name": "test", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["bm1", "bm2"]}
            ],
            "benchmarks": [
                {"id": "bm1", "description": "Benchmark 1", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2},
                {"id": "bm2", "description": "Benchmark 2", "methodology": "test", "target": 50, "unit": "ops", "tolerance": 0.1},
            ],
        }

        spec = Specification.from_dict(data)

        bm1 = spec.get_benchmark("bm1")
        assert bm1 is not None
        assert bm1.target == 100

        bm2 = spec.get_benchmark("bm2")
        assert bm2 is not None
        assert bm2.target == 50

        missing = spec.get_benchmark("nonexistent")
        assert missing is None


class TestBenchmarkSpec:
    """Test BenchmarkSpec dataclass."""

    def test_default_values(self):
        """Test benchmark default values."""
        bm = BenchmarkSpec(
            id="test",
            description="Test benchmark",
            methodology="test",
            target=100,
            unit="ms",
            tolerance=0.2,
        )

        assert bm.lower_is_better is False
        assert bm.runs == 5
        assert bm.warmup_runs == 0
        assert bm.outlier_policy == "iqr"
        assert bm.requirements == []
        assert bm.setup == []
        assert bm.teardown == []

    def test_custom_values(self):
        """Test benchmark with custom values."""
        bm = BenchmarkSpec(
            id="latency",
            description="Response Latency",
            methodology="measure",
            target=50,
            unit="ms",
            tolerance=0.25,
            lower_is_better=True,
            runs=10,
            warmup_runs=2,
            outlier_policy="zscore",
            requirements=["network"],
            setup=["start_server"],
            teardown=["stop_server"],
        )

        assert bm.lower_is_better is True
        assert bm.runs == 10
        assert bm.warmup_runs == 2
        assert bm.outlier_policy == "zscore"
        assert "network" in bm.requirements


class TestCapability:
    """Test Capability dataclass."""

    def test_capability_creation(self):
        """Test capability creation."""
        cap = Capability(
            name="test_capability",
            description="A test capability",
            category="performance",
            benchmarks=["bm1", "bm2"],
        )

        assert cap.name == "test_capability"
        assert cap.category == "performance"
        assert len(cap.benchmarks) == 2


class TestSystemInfo:
    """Test SystemInfo dataclass."""

    def test_system_info_minimal(self):
        """Test system info with minimal fields."""
        sys = SystemInfo(name="test-system", version="1.0.0")

        assert sys.name == "test-system"
        assert sys.version == "1.0.0"
        assert sys.description is None
        assert sys.repository is None

    def test_system_info_full(self):
        """Test system info with all fields."""
        sys = SystemInfo(
            name="test-system",
            version="1.0.0",
            description="A test system",
            repository="https://github.com/example/test",
        )

        assert sys.description == "A test system"
        assert sys.repository == "https://github.com/example/test"
