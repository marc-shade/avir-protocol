"""
Tests for AVIR statistical reliability functions.

Tests Fleiss' κ (kappa) and Krippendorff's α (alpha) implementations
for inter-rater reliability measurement in cross-provider verification.
"""

import pytest
import math
from avir.cross_verification import (
    calculate_fleiss_kappa,
    calculate_krippendorff_alpha,
    QuorumConfig,
)


class TestFleissKappa:
    """Test Fleiss' κ calculation for inter-rater reliability.

    Note: The implementation expects ratings in format [items x raters]
    where each value is a category index (0 to categories-1).
    """

    def test_perfect_agreement(self):
        """Test κ = 1.0 for perfect agreement."""
        # All 3 raters agree on category 0 for all 5 items
        # Format: [item][rater] = category_chosen
        ratings = [
            [0, 0, 0],  # Item 1: all raters chose category 0
            [0, 0, 0],  # Item 2: all raters chose category 0
            [0, 0, 0],  # Item 3: all raters chose category 0
            [0, 0, 0],  # Item 4: all raters chose category 0
            [0, 0, 0],  # Item 5: all raters chose category 0
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_no_agreement_beyond_chance(self):
        """Test κ ≈ 0 when agreement equals chance."""
        # Each item has one rater choosing each category
        # 3 raters, 3 categories, complete disagreement per item
        ratings = [
            [0, 1, 2],  # Each rater chooses different category
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        # Complete disagreement should produce low/negative kappa
        assert kappa < 0.2

    def test_moderate_agreement(self):
        """Test κ in moderate range for partial agreement."""
        # Moderate agreement with balanced category usage across items
        # Using 4 raters to avoid ties
        ratings = [
            [0, 0, 0, 1],  # 3 agree on 0, 1 on 1
            [1, 1, 1, 0],  # 3 agree on 1, 1 on 0
            [2, 2, 2, 1],  # 3 agree on 2, 1 on 1
            [0, 0, 1, 1],  # 2 on 0, 2 on 1
            [1, 1, 2, 2],  # 2 on 1, 2 on 2
            [2, 2, 0, 0],  # 2 on 2, 2 on 0
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        # Should be in moderate range (partial agreement with balanced margins)
        assert 0.0 <= kappa <= 0.8

    def test_known_example(self):
        """Test against known Fleiss' κ example pattern."""
        # Example with 4 items, 5 raters
        ratings = [
            [0, 0, 0, 0, 0],  # All 5 raters chose category 0
            [1, 1, 1, 2, 2],  # 3 chose 1, 2 chose 2
            [2, 2, 2, 2, 2],  # All chose category 2
            [0, 0, 1, 1, 2],  # Split between categories
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        # Should compute without error and be in valid range
        assert -1.0 <= kappa <= 1.0

    def test_empty_ratings(self):
        """Test handling of empty ratings."""
        ratings = []
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        assert kappa == 0.0

    def test_single_item(self):
        """Test with single item."""
        ratings = [[0, 0, 0]]  # All 3 raters agree on category 0
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        # Single item with perfect agreement
        assert kappa == pytest.approx(1.0, abs=0.01)


class TestKrippendorffAlpha:
    """Test Krippendorff's α calculation with missing data support."""

    def test_perfect_agreement_no_missing(self):
        """Test α = 1.0 for perfect agreement without missing data."""
        ratings = [
            [0, 0, 0],  # All rate 0
            [1, 1, 1],  # All rate 1
            [2, 2, 2],  # All rate 2
        ]
        alpha = calculate_krippendorff_alpha(ratings, categories=3)
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_handles_missing_data(self):
        """Test that None values (abstains) are handled correctly."""
        ratings = [
            [0, 0, None],  # Third rater abstained
            [1, 1, 1],     # All rated
            [2, None, 2],  # Second rater abstained
        ]
        alpha = calculate_krippendorff_alpha(ratings, categories=3)
        # Should compute without error
        assert -1.0 <= alpha <= 1.0

    def test_all_missing_returns_zero(self):
        """Test that all missing data returns 0."""
        ratings = [
            [None, None, None],
            [None, None, None],
        ]
        alpha = calculate_krippendorff_alpha(ratings, categories=3)
        assert alpha == 0.0

    def test_complete_disagreement(self):
        """Test α near 0 or negative for complete disagreement."""
        ratings = [
            [0, 1, 2],  # Complete disagreement
            [1, 2, 0],  # Complete disagreement
            [2, 0, 1],  # Complete disagreement
        ]
        alpha = calculate_krippendorff_alpha(ratings, categories=3)
        # Should be low (near 0 or negative)
        assert alpha < 0.3

    def test_partial_agreement_with_abstains(self):
        """Test realistic scenario with some abstains."""
        ratings = [
            [0, 0, 0],      # All agree
            [1, 1, None],   # Two agree, one abstains
            [0, 0, 0],      # All agree
            [None, 1, 1],   # Two agree, one abstains
            [2, 2, 2],      # All agree
        ]
        alpha = calculate_krippendorff_alpha(ratings, categories=3)
        # High agreement should result in high alpha despite abstains
        assert alpha > 0.7

    def test_empty_ratings(self):
        """Test handling of empty ratings."""
        ratings = []
        alpha = calculate_krippendorff_alpha(ratings, categories=3)
        assert alpha == 0.0


class TestQuorumConfig:
    """Test QuorumConfig validation."""

    def test_default_config(self):
        """Test default quorum configuration."""
        config = QuorumConfig()
        assert config.minimum_providers == 2
        assert config.minimum_agreement == pytest.approx(2/3, abs=0.01)
        assert config.maximum_abstain_ratio == pytest.approx(1/3, abs=0.01)
        assert config.require_cross_org is True

    def test_custom_config(self):
        """Test custom quorum configuration."""
        config = QuorumConfig(
            minimum_providers=3,
            minimum_agreement=0.8,
            maximum_abstain_ratio=0.2,
            require_cross_org=False,
        )
        assert config.minimum_providers == 3
        assert config.minimum_agreement == 0.8
        assert config.maximum_abstain_ratio == 0.2
        assert config.require_cross_org is False

    def test_config_validation_bounds(self):
        """Test that config values are within valid bounds."""
        config = QuorumConfig()
        assert 0.0 <= config.minimum_agreement <= 1.0
        assert 0.0 <= config.maximum_abstain_ratio <= 1.0
        assert config.minimum_providers >= 1


class TestInterRaterReliabilityInterpretation:
    """Test interpretation thresholds for reliability coefficients."""

    def test_kappa_interpretation_thresholds(self):
        """
        Test Fleiss' κ interpretation thresholds.

        Standard interpretation (Landis & Koch, 1977):
        - < 0.00: Poor
        - 0.00-0.20: Slight
        - 0.21-0.40: Fair
        - 0.41-0.60: Moderate
        - 0.61-0.80: Substantial
        - 0.81-1.00: Almost perfect
        """
        # Near-perfect agreement: all 4 raters choose category 0
        perfect = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        kappa_perfect = calculate_fleiss_kappa(perfect, categories=5)
        assert kappa_perfect > 0.8, "Perfect agreement should have κ > 0.8"

    def test_alpha_reliability_threshold(self):
        """
        Test Krippendorff's α reliability threshold.

        Krippendorff recommends:
        - α ≥ 0.800 for reliable data
        - α ≥ 0.667 for tentative conclusions
        """
        # High reliability ratings
        reliable = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
        alpha = calculate_krippendorff_alpha(reliable, categories=2)
        assert alpha >= 0.8, "Consistent ratings should have α ≥ 0.8"


class TestStatisticalEdgeCases:
    """Test edge cases in statistical calculations."""

    def test_two_raters_two_categories(self):
        """Test minimum meaningful scenario."""
        ratings = [
            [0, 0],  # Both raters chose category 0
            [1, 1],  # Both raters chose category 1
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=2)
        assert -1.0 <= kappa <= 1.0

    def test_large_number_of_raters(self):
        """Test with many raters."""
        # 10 raters with high agreement, balanced category usage across items
        ratings = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # All 10 agree on 0
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All 10 agree on 1
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # All 10 agree on 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9 agree on 0, 1 differs
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # 9 agree on 1, 1 differs
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 0],  # 9 agree on 2, 1 differs
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        assert kappa > 0.8  # High agreement

    def test_single_category_used(self):
        """Test when all raters use single category."""
        ratings = [
            [0, 0, 0],  # All chose category 0
            [0, 0, 0],
            [0, 0, 0],
        ]
        kappa = calculate_fleiss_kappa(ratings, categories=3)
        # Perfect agreement on single category
        assert kappa == pytest.approx(1.0, abs=0.01)
