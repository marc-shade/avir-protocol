"""
AVIR Specification Document Parser and Validator

Handles parsing and validation of AVIR specification documents
in YAML and JSON formats.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class SystemInfo:
    """System metadata from specification."""
    name: str
    version: str
    description: Optional[str] = None
    repository: Optional[str] = None


@dataclass
class Capability:
    """System capability declaration."""
    name: str
    description: str
    category: str  # memory, reasoning, coordination, performance, custom
    benchmarks: List[str]


@dataclass
class BenchmarkSpec:
    """Benchmark specification from document."""
    id: str
    description: str
    methodology: str
    target: float
    unit: str
    tolerance: float
    lower_is_better: bool = False
    runs: int = 5
    warmup_runs: int = 0
    outlier_policy: str = "iqr"
    requirements: List[str] = field(default_factory=list)
    setup: List[str] = field(default_factory=list)
    teardown: List[str] = field(default_factory=list)


@dataclass
class Specification:
    """
    AVIR Specification Document.

    Contains all information needed to verify a system's capabilities.
    """
    avir_version: str
    system: SystemInfo
    capabilities: List[Capability]
    benchmarks: List[BenchmarkSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)

    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    _hash: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_file(cls, path: str | Path) -> "Specification":
        """Load specification from YAML or JSON file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Specification file not found: {path}")

        content = path.read_text()

        if path.suffix in ('.yaml', '.yml'):
            data = yaml.safe_load(content)
        elif path.suffix == '.json':
            data = json.loads(content)
        else:
            # Try YAML first, then JSON
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError:
                data = json.loads(content)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Specification":
        """Create specification from dictionary."""
        # Validate required fields
        required = ['avir_version', 'system', 'capabilities', 'benchmarks']
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Parse system info
        sys_data = data['system']
        system = SystemInfo(
            name=sys_data['name'],
            version=sys_data['version'],
            description=sys_data.get('description'),
            repository=sys_data.get('repository'),
        )

        # Parse capabilities
        capabilities = []
        for cap_data in data['capabilities']:
            cap = Capability(
                name=cap_data['name'],
                description=cap_data['description'],
                category=cap_data['category'],
                benchmarks=cap_data['benchmarks'],
            )
            capabilities.append(cap)

        # Parse benchmarks
        benchmarks = []
        for bench_data in data['benchmarks']:
            bench = BenchmarkSpec(
                id=bench_data['id'],
                description=bench_data['description'],
                methodology=bench_data['methodology'],
                target=bench_data['target'],
                unit=bench_data['unit'],
                tolerance=bench_data['tolerance'],
                lower_is_better=bench_data.get('lower_is_better', False),
                runs=bench_data.get('runs', 5),
                warmup_runs=bench_data.get('warmup_runs', 0),
                outlier_policy=bench_data.get('outlier_policy', 'iqr'),
                requirements=bench_data.get('requirements', []),
                setup=bench_data.get('setup', []),
                teardown=bench_data.get('teardown', []),
            )
            benchmarks.append(bench)

        return cls(
            avir_version=data['avir_version'],
            system=system,
            capabilities=capabilities,
            benchmarks=benchmarks,
            metadata=data.get('metadata', {}),
            _raw=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        return {
            'avir_version': self.avir_version,
            'system': {
                'name': self.system.name,
                'version': self.system.version,
                'description': self.system.description,
                'repository': self.system.repository,
            },
            'capabilities': [
                {
                    'name': cap.name,
                    'description': cap.description,
                    'category': cap.category,
                    'benchmarks': cap.benchmarks,
                }
                for cap in self.capabilities
            ],
            'benchmarks': [
                {
                    'id': bench.id,
                    'description': bench.description,
                    'methodology': bench.methodology,
                    'target': bench.target,
                    'unit': bench.unit,
                    'tolerance': bench.tolerance,
                    'lower_is_better': bench.lower_is_better,
                    'runs': bench.runs,
                    'warmup_runs': bench.warmup_runs,
                    'outlier_policy': bench.outlier_policy,
                    'requirements': bench.requirements,
                    'setup': bench.setup,
                    'teardown': bench.teardown,
                }
                for bench in self.benchmarks
            ],
            'metadata': self.metadata,
        }

    @property
    def hash(self) -> str:
        """Calculate SHA-256 hash of canonical specification."""
        if self._hash is None:
            canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
            self._hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self._hash

    def get_benchmark(self, benchmark_id: str) -> Optional[BenchmarkSpec]:
        """Get benchmark by ID."""
        for bench in self.benchmarks:
            if bench.id == benchmark_id:
                return bench
        return None

    def validate(self) -> List[str]:
        """
        Validate specification and return list of errors.

        Returns empty list if valid.
        """
        errors = []

        # Check version format
        if not self.avir_version:
            errors.append("avir_version is required")

        # Check system info
        if not self.system.name:
            errors.append("system.name is required")
        if not self.system.version:
            errors.append("system.version is required")

        # Check capabilities reference valid benchmarks
        benchmark_ids = {b.id for b in self.benchmarks}
        for cap in self.capabilities:
            for bench_id in cap.benchmarks:
                if bench_id not in benchmark_ids:
                    errors.append(f"Capability '{cap.name}' references unknown benchmark '{bench_id}'")

        # Check benchmark values
        for bench in self.benchmarks:
            if bench.tolerance < 0 or bench.tolerance > 1:
                errors.append(f"Benchmark '{bench.id}' has invalid tolerance: {bench.tolerance}")
            if bench.runs < 1:
                errors.append(f"Benchmark '{bench.id}' has invalid runs: {bench.runs}")
            if bench.category := getattr(bench, 'category', None):
                valid_categories = {'memory', 'reasoning', 'coordination', 'performance', 'custom'}
                if bench.category not in valid_categories:
                    errors.append(f"Benchmark '{bench.id}' has invalid category: {bench.category}")

        return errors

    def is_valid(self) -> bool:
        """Check if specification is valid."""
        return len(self.validate()) == 0

    def __str__(self) -> str:
        return f"Specification({self.system.name} v{self.system.version}, {len(self.benchmarks)} benchmarks)"
