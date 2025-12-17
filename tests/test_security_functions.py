"""
Tests for AVIR security and blinding functions.

Tests BLAKE2b salted hashing, canary ID generation and detection,
and context blinding mechanisms.
"""

import pytest
import re
from avir.cross_verification import (
    blake2b_salted_hash,
    generate_canary_id,
    BlindedContext,
)
from avir.specification import Specification


class TestBlake2bSaltedHash:
    """Test BLAKE2b salted hashing for context blinding."""

    def test_produces_hex_output(self):
        """Test that output is valid hexadecimal."""
        result = blake2b_salted_hash("test data")
        assert all(c in '0123456789abcdef' for c in result)

    def test_default_length(self):
        """Test default output length is 32 characters (16 bytes)."""
        result = blake2b_salted_hash("test data")
        assert len(result) == 32  # 16 bytes = 32 hex chars

    def test_custom_length(self):
        """Test custom output length."""
        result = blake2b_salted_hash("test data", length=8)
        assert len(result) == 16  # 8 bytes = 16 hex chars

        result = blake2b_salted_hash("test data", length=32)
        assert len(result) == 64  # 32 bytes = 64 hex chars

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different hashes."""
        salt = b'fixed_salt_16byt'  # Fixed salt for comparison
        hash1 = blake2b_salted_hash("input one", salt=salt)
        hash2 = blake2b_salted_hash("input two", salt=salt)
        assert hash1 != hash2

    def test_same_input_different_salt_different_output(self):
        """Test that same input with different salts produces different hashes."""
        salt1 = b'salt_number_one!'
        salt2 = b'salt_number_two!'
        hash1 = blake2b_salted_hash("same input", salt=salt1)
        hash2 = blake2b_salted_hash("same input", salt=salt2)
        assert hash1 != hash2

    def test_same_input_same_salt_same_output(self):
        """Test determinism: same input and salt produce same hash."""
        salt = b'deterministic_16'
        hash1 = blake2b_salted_hash("test input", salt=salt)
        hash2 = blake2b_salted_hash("test input", salt=salt)
        assert hash1 == hash2

    def test_auto_generated_salt_produces_different_hashes(self):
        """Test that auto-generated salts produce different hashes."""
        # Without explicit salt, should auto-generate random salt
        hash1 = blake2b_salted_hash("same input")
        hash2 = blake2b_salted_hash("same input")
        # With random salts, same input should produce different hashes
        assert hash1 != hash2

    def test_empty_string_input(self):
        """Test hashing empty string."""
        result = blake2b_salted_hash("")
        assert len(result) == 32
        assert all(c in '0123456789abcdef' for c in result)

    def test_unicode_input(self):
        """Test hashing unicode content."""
        result = blake2b_salted_hash("日本語テスト 🔐")
        assert len(result) == 32
        assert all(c in '0123456789abcdef' for c in result)

    def test_large_input(self):
        """Test hashing large input."""
        large_input = "x" * 1_000_000  # 1MB of data
        result = blake2b_salted_hash(large_input)
        assert len(result) == 32


class TestCanaryIdGeneration:
    """Test canary ID generation for leak detection."""

    def test_canary_format(self):
        """Test canary ID follows expected format."""
        canary = generate_canary_id()
        # Should be CANARY-{16 hex chars}
        assert canary.startswith("CANARY-")
        hex_part = canary.split("-")[1]
        assert len(hex_part) == 16
        assert all(c in '0123456789ABCDEF' for c in hex_part)

    def test_canary_uniqueness(self):
        """Test that generated canaries are unique."""
        canaries = [generate_canary_id() for _ in range(100)]
        assert len(set(canaries)) == 100  # All unique

    def test_canary_regex_match(self):
        """Test canary matches detection regex."""
        canary = generate_canary_id()
        pattern = r'^CANARY-[0-9A-F]{16}$'
        assert re.match(pattern, canary)

    def test_canary_is_detectable_in_text(self):
        """Test canary can be found in mixed text."""
        canary = generate_canary_id()
        text = f"Some random text {canary} and more text"
        pattern = r'CANARY-[0-9A-F]{16}'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == canary

    def test_no_false_positives(self):
        """Test that similar patterns don't match."""
        text = "CANARY-123 and CANARY-ABCDEFG and canary-1234567890ABCDEF"
        pattern = r'CANARY-[0-9A-F]{16}'
        matches = re.findall(pattern, text)
        assert len(matches) == 0  # None should match


class TestCanaryLeakDetection:
    """Test canary-based information leak detection."""

    def test_detect_canary_in_output(self):
        """Test detection of leaked canary in provider output."""
        canary = generate_canary_id()

        # Simulate provider output containing leaked canary
        provider_output = f"""
        Analysis complete. Results:
        - Benchmark passed
        - Context ID: {canary}
        - Score: 95%
        """

        pattern = r'CANARY-[0-9A-F]{16}'
        matches = re.findall(pattern, provider_output)

        assert len(matches) == 1
        assert matches[0] == canary

    def test_no_leak_in_clean_output(self):
        """Test no false detection in clean output."""
        clean_output = """
        Analysis complete. Results:
        - Benchmark passed
        - Context ID: abc123
        - Score: 95%
        """

        pattern = r'CANARY-[0-9A-F]{16}'
        matches = re.findall(pattern, clean_output)

        assert len(matches) == 0

    def test_multiple_canary_detection(self):
        """Test detection of multiple leaked canaries."""
        canary1 = generate_canary_id()
        canary2 = generate_canary_id()

        output = f"First: {canary1}, Second: {canary2}"

        pattern = r'CANARY-[0-9A-F]{16}'
        matches = re.findall(pattern, output)

        assert len(matches) == 2
        assert canary1 in matches
        assert canary2 in matches


class TestBlindedContext:
    """Test context blinding with BLAKE2b hashing."""

    @pytest.fixture
    def sample_spec(self):
        """Create sample specification for testing."""
        return Specification.from_dict({
            "avir_version": "1.0.0",
            "metadata": {"name": "Test Spec"},
            "system": {"name": "test-system", "version": "1.0.0"},
            "capabilities": [
                {"name": "test_cap", "description": "Test capability", "category": "performance", "benchmarks": ["benchmark_one", "benchmark_two"]}
            ],
            "benchmarks": [
                {"id": "benchmark_one", "description": "Test 1", "methodology": "test", "target": 100, "unit": "ms", "tolerance": 0.2},
                {"id": "benchmark_two", "description": "Test 2", "methodology": "test", "target": 50, "unit": "ms", "tolerance": 0.1},
            ],
        })

    def test_benchmark_ids_anonymized(self, sample_spec):
        """Test that benchmark IDs are anonymized."""
        blinded = BlindedContext.create(sample_spec)

        original_ids = ["benchmark_one", "benchmark_two"]
        blinded_ids = [b["id"] for b in blinded.benchmark_specs]

        # No original IDs should appear
        for orig_id in original_ids:
            assert orig_id not in blinded_ids

        # Blinded IDs should be 16-character hashes (8 bytes = 16 hex chars)
        for bid in blinded_ids:
            assert len(bid) == 16

    def test_specification_hash_preserved(self, sample_spec):
        """Test that specification hash is preserved for verification."""
        blinded = BlindedContext.create(sample_spec)
        assert blinded.specification_hash == sample_spec.hash

    def test_context_id_generated(self, sample_spec):
        """Test that context ID is generated."""
        blinded = BlindedContext.create(sample_spec)
        assert blinded.context_id is not None
        assert len(blinded.context_id) == 32  # 16 bytes = 32 hex chars

    def test_nonce_generated(self, sample_spec):
        """Test that nonce is generated for uniqueness."""
        blinded = BlindedContext.create(sample_spec)
        assert blinded.nonce is not None

    def test_different_contexts_different_ids(self, sample_spec):
        """Test that each blinding produces different IDs."""
        blinded1 = BlindedContext.create(sample_spec)
        blinded2 = BlindedContext.create(sample_spec)

        # Context IDs should differ (due to random nonce)
        assert blinded1.context_id != blinded2.context_id

    def test_benchmark_targets_preserved(self, sample_spec):
        """Test that benchmark targets and tolerances are preserved."""
        blinded = BlindedContext.create(sample_spec)

        # Targets should be preserved for execution
        targets = [b["target"] for b in blinded.benchmark_specs]
        assert 100 in targets
        assert 50 in targets

        # Tolerances should be preserved
        tolerances = [b["tolerance"] for b in blinded.benchmark_specs]
        assert 0.2 in tolerances
        assert 0.1 in tolerances


class TestHashCollisionResistance:
    """Test collision resistance properties."""

    def test_no_collisions_in_reasonable_sample(self):
        """Test no collisions in large sample of hashes."""
        salt = b'fixed_salt_16byt'
        hashes = set()

        for i in range(10000):
            h = blake2b_salted_hash(f"input_{i}", salt=salt)
            assert h not in hashes, f"Collision detected at input_{i}"
            hashes.add(h)

    def test_similar_inputs_different_hashes(self):
        """Test that similar inputs produce different hashes."""
        salt = b'fixed_salt_16byt'

        # Single character difference
        h1 = blake2b_salted_hash("test_input_a", salt=salt)
        h2 = blake2b_salted_hash("test_input_b", salt=salt)
        assert h1 != h2

        # Single bit difference (conceptually)
        h3 = blake2b_salted_hash("test_input_0", salt=salt)
        h4 = blake2b_salted_hash("test_input_1", salt=salt)
        assert h3 != h4


class TestSecurityProperties:
    """Test security properties of blinding system."""

    def test_original_data_not_recoverable(self):
        """Test that original data cannot be recovered from hash."""
        original = "sensitive_benchmark_name"
        hashed = blake2b_salted_hash(original)

        # Hash should not contain original
        assert original not in hashed
        assert "sensitive" not in hashed
        assert "benchmark" not in hashed

    def test_salt_provides_additional_security(self):
        """Test that salt adds security against rainbow tables."""
        # Same input with different salts
        hashes = set()
        for i in range(100):
            salt = f"salt_{i:04d}_bytes".encode()[:16]
            h = blake2b_salted_hash("known_input", salt=salt)
            hashes.add(h)

        # All should be different due to salt
        assert len(hashes) == 100
