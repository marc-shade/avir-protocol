"""
AVIR Verdict System

Defines verdict types and criteria for verification outcomes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class VerdictLevel(Enum):
    """Overall verification verdict levels."""
    VERIFIED = "VERIFIED"       # All benchmarks pass
    PARTIAL = "PARTIAL"         # 60-99% pass
    FAILED = "FAILED"           # <60% pass
    INVALID = "INVALID"         # Spec or execution errors
    INCONCLUSIVE = "INCONCLUSIVE"  # Insufficient data


@dataclass
class Verdict:
    """
    Verification verdict with details.

    Represents the final outcome of AVIR verification.
    """
    level: VerdictLevel
    passed: int
    failed: int
    total: int
    pass_rate: float
    details: str
    failed_benchmarks: List[str]
    warnings: List[str]

    @classmethod
    def from_results(
        cls,
        passed: int,
        failed: int,
        total: int,
        failed_benchmarks: List[str] = None,
        warnings: List[str] = None,
    ) -> "Verdict":
        """
        Create verdict from result counts.

        Args:
            passed: Number of passing benchmarks
            failed: Number of failing benchmarks
            total: Total benchmark count
            failed_benchmarks: IDs of failed benchmarks
            warnings: Any warning messages

        Returns:
            Verdict instance
        """
        failed_benchmarks = failed_benchmarks or []
        warnings = warnings or []

        if total == 0:
            return cls(
                level=VerdictLevel.INVALID,
                passed=0,
                failed=0,
                total=0,
                pass_rate=0.0,
                details="No benchmarks executed",
                failed_benchmarks=[],
                warnings=["No benchmarks were available for verification"],
            )

        pass_rate = passed / total

        if pass_rate == 1.0:
            level = VerdictLevel.VERIFIED
            details = f"All {total} benchmarks passed within tolerance"
        elif pass_rate >= 0.6:
            level = VerdictLevel.PARTIAL
            details = f"{passed}/{total} benchmarks passed ({failed} failed)"
        else:
            level = VerdictLevel.FAILED
            details = f"Only {passed}/{total} benchmarks passed"

        return cls(
            level=level,
            passed=passed,
            failed=failed,
            total=total,
            pass_rate=pass_rate,
            details=details,
            failed_benchmarks=failed_benchmarks,
            warnings=warnings,
        )

    @property
    def is_success(self) -> bool:
        """Check if verdict indicates successful verification."""
        return self.level == VerdictLevel.VERIFIED

    @property
    def is_partial(self) -> bool:
        """Check if verdict indicates partial success."""
        return self.level == VerdictLevel.PARTIAL

    @property
    def is_failure(self) -> bool:
        """Check if verdict indicates failure."""
        return self.level in (VerdictLevel.FAILED, VerdictLevel.INVALID)

    def to_dict(self) -> dict:
        return {
            'verdict': self.level.value,
            'passed': self.passed,
            'failed': self.failed,
            'total': self.total,
            'pass_rate': round(self.pass_rate, 4),
            'details': self.details,
            'failed_benchmarks': self.failed_benchmarks,
            'warnings': self.warnings,
        }

    def __str__(self) -> str:
        return f"{self.level.value}: {self.details}"


def determine_overall_verdict(
    benchmark_verdicts: List[dict],
    min_pass_rate: float = 0.6,
) -> Verdict:
    """
    Determine overall verdict from benchmark verdicts.

    Args:
        benchmark_verdicts: List of benchmark verdict dictionaries
        min_pass_rate: Minimum pass rate for PARTIAL (default 0.6)

    Returns:
        Overall Verdict
    """
    if not benchmark_verdicts:
        return Verdict.from_results(0, 0, 0)

    passed = 0
    failed = 0
    failed_benchmarks = []
    warnings = []

    for v in benchmark_verdicts:
        if v.get('verdict') == 'PASS':
            passed += 1
        else:
            failed += 1
            failed_benchmarks.append(v.get('benchmark_id', 'unknown'))

        # Check for high variance warnings
        stats = v.get('statistics', {})
        if stats.get('std_dev', 0) > stats.get('mean', 1) * 0.3:
            warnings.append(
                f"High variance in {v.get('benchmark_id')}: "
                f"std_dev={stats.get('std_dev', 0):.2f}"
            )

    total = passed + failed

    return Verdict.from_results(
        passed=passed,
        failed=failed,
        total=total,
        failed_benchmarks=failed_benchmarks,
        warnings=warnings,
    )


def compare_verdicts(v1: Verdict, v2: Verdict) -> dict:
    """
    Compare two verdicts for consistency.

    Useful for multi-provider verification.

    Args:
        v1: First verdict
        v2: Second verdict

    Returns:
        Comparison results
    """
    agreement = v1.level == v2.level

    return {
        'agreement': agreement,
        'verdict_1': v1.level.value,
        'verdict_2': v2.level.value,
        'pass_rate_diff': abs(v1.pass_rate - v2.pass_rate),
        'consensus': v1.level.value if agreement else 'INCONCLUSIVE',
    }
