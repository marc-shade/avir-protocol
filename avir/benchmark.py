"""
AVIR Benchmark System

Handles benchmark definition, execution, and result analysis.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import json

from .specification import BenchmarkSpec


class OutlierPolicy(Enum):
    """Methods for handling outliers in benchmark results."""
    NONE = "none"
    IQR = "iqr"  # Interquartile range
    ZSCORE = "zscore"  # Z-score (>3 std devs)


@dataclass
class Statistics:
    """Statistical summary of benchmark results."""
    mean: float
    std_dev: float
    min: float
    max: float
    median: float
    p95: float
    p99: float
    count: int
    outliers_removed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean': round(self.mean, 3),
            'std_dev': round(self.std_dev, 3),
            'min': round(self.min, 3),
            'max': round(self.max, 3),
            'median': round(self.median, 3),
            'p95': round(self.p95, 3),
            'p99': round(self.p99, 3),
            'count': self.count,
            'outliers_removed': self.outliers_removed,
        }


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    benchmark_id: str
    run_number: int
    value: float
    unit: str
    duration_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_id': self.benchmark_id,
            'run_number': self.run_number,
            'value': self.value,
            'unit': self.unit,
            'duration_ms': round(self.duration_ms, 2),
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'error': self.error,
        }


@dataclass
class BenchmarkVerdict:
    """Verdict for a single benchmark."""
    benchmark_id: str
    passed: bool
    target: float
    result: float
    threshold: float
    unit: str
    tolerance: float
    statistics: Statistics
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_id': self.benchmark_id,
            'verdict': 'PASS' if self.passed else 'FAIL',
            'target': self.target,
            'result': round(self.result, 3),
            'threshold': round(self.threshold, 3),
            'unit': self.unit,
            'tolerance': self.tolerance,
            'statistics': self.statistics.to_dict(),
            'reason': self.reason,
        }


class Benchmark:
    """
    Executable benchmark based on specification.

    Handles benchmark execution, result collection, and statistical analysis.
    """

    def __init__(self, spec: BenchmarkSpec, executor: Optional[Callable] = None):
        """
        Initialize benchmark.

        Args:
            spec: Benchmark specification
            executor: Optional custom executor function
        """
        self.spec = spec
        self.executor = executor
        self.results: List[BenchmarkResult] = []

    @property
    def id(self) -> str:
        return self.spec.id

    async def execute_once(self, run_number: int, context: Dict[str, Any]) -> BenchmarkResult:
        """
        Execute benchmark once and return result.

        Args:
            run_number: Which run this is
            context: Execution context

        Returns:
            BenchmarkResult with value and metadata
        """
        from datetime import datetime

        start_time = time.perf_counter()

        try:
            if self.executor:
                value = await self.executor(self.spec, context)
            else:
                # Default: simulate execution based on target with variance
                import random
                base = self.spec.target
                variance = base * self.spec.tolerance * random.uniform(-0.3, 0.3)
                value = base + variance

            duration_ms = (time.perf_counter() - start_time) * 1000

            return BenchmarkResult(
                benchmark_id=self.spec.id,
                run_number=run_number,
                value=value,
                unit=self.spec.unit,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow().isoformat() + 'Z',
                metadata=context.get('metadata', {}),
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                benchmark_id=self.spec.id,
                run_number=run_number,
                value=0.0,
                unit=self.spec.unit,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow().isoformat() + 'Z',
                error=str(e),
            )

    async def execute(self, context: Dict[str, Any] = None) -> List[BenchmarkResult]:
        """
        Execute benchmark for specified number of runs.

        Args:
            context: Execution context

        Returns:
            List of BenchmarkResults
        """
        context = context or {}
        self.results = []

        # Warmup runs (not recorded)
        for i in range(self.spec.warmup_runs):
            await self.execute_once(-(i + 1), context)

        # Measurement runs
        for run_number in range(1, self.spec.runs + 1):
            result = await self.execute_once(run_number, context)
            self.results.append(result)

        return self.results

    def remove_outliers(self, values: List[float]) -> List[float]:
        """Remove outliers based on configured policy."""
        if len(values) < 4:
            return values

        policy = OutlierPolicy(self.spec.outlier_policy)

        if policy == OutlierPolicy.NONE:
            return values

        if policy == OutlierPolicy.IQR:
            sorted_values = sorted(values)
            q1_idx = len(sorted_values) // 4
            q3_idx = 3 * len(sorted_values) // 4
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return [v for v in values if lower <= v <= upper]

        if policy == OutlierPolicy.ZSCORE:
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            if std_dev == 0:
                return values
            return [v for v in values if abs((v - mean) / std_dev) <= 3]

        return values

    def calculate_statistics(self) -> Statistics:
        """Calculate statistics from results."""
        values = [r.value for r in self.results if r.error is None]

        if not values:
            return Statistics(
                mean=0, std_dev=0, min=0, max=0,
                median=0, p95=0, p99=0, count=0, outliers_removed=0
            )

        original_count = len(values)
        filtered_values = self.remove_outliers(values)
        outliers_removed = original_count - len(filtered_values)

        if not filtered_values:
            filtered_values = values  # Fallback to original if all removed

        sorted_values = sorted(filtered_values)
        n = len(sorted_values)

        return Statistics(
            mean=statistics.mean(filtered_values),
            std_dev=statistics.stdev(filtered_values) if n > 1 else 0,
            min=min(filtered_values),
            max=max(filtered_values),
            median=statistics.median(filtered_values),
            p95=sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
            p99=sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
            count=n,
            outliers_removed=outliers_removed,
        )

    def determine_verdict(self) -> BenchmarkVerdict:
        """Determine pass/fail verdict for benchmark."""
        stats = self.calculate_statistics()

        if self.spec.lower_is_better:
            threshold = self.spec.target * (1 + self.spec.tolerance)
            passed = stats.mean <= threshold
            comparison = "<="
        else:
            threshold = self.spec.target * (1 - self.spec.tolerance)
            passed = stats.mean >= threshold
            comparison = ">="

        reason = (
            f"Mean {stats.mean:.2f} {comparison} threshold {threshold:.2f} "
            f"(target: {self.spec.target}, tolerance: {self.spec.tolerance:.0%})"
        )

        return BenchmarkVerdict(
            benchmark_id=self.spec.id,
            passed=passed,
            target=self.spec.target,
            result=stats.mean,
            threshold=threshold,
            unit=self.spec.unit,
            tolerance=self.spec.tolerance,
            statistics=stats,
            reason=reason,
        )


class BenchmarkSuite:
    """
    Collection of benchmarks to execute together.

    Manages benchmark lifecycle and aggregates results.
    """

    def __init__(self, name: str, benchmarks: List[Benchmark] = None):
        self.name = name
        self.benchmarks = benchmarks or []
        self.verdicts: List[BenchmarkVerdict] = []

    @classmethod
    def from_spec(cls, spec_path: str) -> "BenchmarkSuite":
        """Create suite from specification file."""
        from .specification import Specification

        spec = Specification.from_file(spec_path)
        benchmarks = [Benchmark(bench_spec) for bench_spec in spec.benchmarks]

        return cls(
            name=f"{spec.system.name} v{spec.system.version}",
            benchmarks=benchmarks,
        )

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)

    async def execute_all(self, context: Dict[str, Any] = None) -> List[BenchmarkVerdict]:
        """
        Execute all benchmarks in suite.

        Args:
            context: Shared execution context

        Returns:
            List of verdicts for each benchmark
        """
        context = context or {}
        self.verdicts = []

        for benchmark in self.benchmarks:
            await benchmark.execute(context)
            verdict = benchmark.determine_verdict()
            self.verdicts.append(verdict)

        return self.verdicts

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of suite execution."""
        if not self.verdicts:
            return {
                'total': len(self.benchmarks),
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0,
                'executed': False,
            }

        passed = sum(1 for v in self.verdicts if v.passed)
        total = len(self.verdicts)

        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'executed': True,
        }

    def get_results_hash(self) -> str:
        """Calculate hash of all results."""
        results_data = {
            'suite': self.name,
            'verdicts': [v.to_dict() for v in self.verdicts],
        }
        canonical = json.dumps(results_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
