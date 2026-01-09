# RFC-038: Continuous Review Tooling Implementation

## Metadata

- **RFC ID**: RFC-038
- **Title**: Continuous Review Tooling Implementation
- **Status**: Draft
- **Created**: 2026-01-04
- **Related Issues**: #45, #169, #170
- **Related RFCs**: RFC-031 (Code Complexity Analysis)

## Summary

This RFC provides detailed implementation guidance for the remaining continuous review tools
identified in issue #45:

1. **Dependabot** (#169) - Automated dependency updates
2. **Module Coupling Analysis** (#170) - pydeps for dependency visualization
3. **Pre-release Checklist** - Automated release validation
4. **Future Enhancements** - Memory profiling, performance benchmarking

## Goals

- Automate dependency management with Dependabot
- Visualize and monitor module dependencies with pydeps
- Create enforceable pre-release checklists
- Document implementation patterns for future tooling

## Non-Goals

- Replacing existing security tools (Snyk, pip-audit)
- Full performance benchmarking suite (Phase 3)
- Memory profiling implementation (Phase 3)

---

## 1. Dependabot Configuration (#169)

### 1.1 Overview

Dependabot automatically creates PRs to update dependencies, keeping the project secure and current.

### 1.2 Configuration File

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # =============================================================================
  # Python Dependencies (pip)
  # =============================================================================

  - package-ecosystem: "pip"
    directory: "/"

    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
      timezone: "Europe/Amsterdam"
    open-pull-requests-limit: 5
    reviewers:

      - "chipi"
    labels:

      - "dependencies"
      - "automated"
    commit-message:

      prefix: "deps"
      include: "scope"

```text

    # Group related dependencies to reduce PR noise
    groups:
      # Development tools (can update together)
      dev-tools:
        patterns:

```

```

      docs:
        patterns:

```
      timezone: "Europe/Amsterdam"
    open-pull-requests-limit: 3
    labels:

```

    schedule:
      interval: "monthly"
    labels:

```yaml

### 1.3 Configuration Decisions

| Decision | Rationale |
| ---------- | ----------- |
| Weekly schedule (Monday 6am) | Start of week, gives time to review |
| 5 PR limit for pip | Avoids overwhelming with updates |
| Grouped updates | Reduces PR noise, related packages together |
| Patch-only for ML | ML libraries have frequent breaking changes |
| Monthly for Docker | Base images change less frequently |

### 1.4 Expected Behavior

```yaml

### 1.5 Maintenance

- Review and merge Dependabot PRs weekly
- Monitor for breaking changes in ML dependencies
- Adjust groups based on update patterns

---

## 2. Module Coupling Analysis (#170)

### 2.1 Overview

Use pydeps to visualize module dependencies, detect circular imports, and track architecture health.

### 2.2 Installation

Add to `pyproject.toml`:

```toml

[project.optional-dependencies]
dev = [
    # ... existing dev dependencies ...
    "pydeps>=1.12.0",
]

```

### 2.3 Makefile Targets

Add to `Makefile`:

```makefile

# =============================================================================
# Module Dependency Analysis
# =============================================================================

.PHONY: deps-graph deps-graph-full deps-check-cycles

## Generate module dependency graph (simplified)

deps-graph:
	@echo "=== Module Dependency Graph (Simplified) ==="
	@mkdir -p reports
	@$(PYTHON) -m pydeps src/podcast_scraper \
		--cluster \
		--max-bacon=2 \
		--exclude-exact podcast_scraper.__main__ \
		-o reports/deps-simple.svg
	@echo "Generated: reports/deps-simple.svg"

## Generate full module dependency graph

deps-graph-full:
	@echo "=== Module Dependency Graph (Full) ==="
	@mkdir -p reports
	@$(PYTHON) -m pydeps src/podcast_scraper \
		--cluster \
		--show-deps \
		-o reports/deps-full.svg
	@echo "Generated: reports/deps-full.svg"

## Check for circular imports

deps-check-cycles:
	@echo "=== Checking for Circular Imports ==="
	@$(PYTHON) -m pydeps src/podcast_scraper --show-cycles --no-show || true

## Full dependency analysis

deps-analyze: deps-check-cycles deps-graph
	@echo "=== Dependency Analysis Complete ==="
	@echo "Check reports/deps-simple.svg for visualization"

```

### 2.4 Analysis Script

Create `scripts/analyze_dependencies.py`:

```python

#!/usr/bin/env python3
"""
Analyze module dependencies and detect architectural issues.

Usage:
    python scripts/analyze_dependencies.py [--check] [--report]

Options:
    --check     Run checks and exit with error if issues found
    --report    Generate detailed JSON report
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def run_pydeps_cycles() -> tuple[bool, str]:

```text

    """Check for circular dependencies."""
    result = subprocess.run(
        ["python", "-m", "pydeps", "src/podcast_scraper", "--show-cycles", "--no-show"],
        capture_output=True,
        text=True,
    )
    has_cycles = "cycle" in result.stdout.lower() or "cycle" in result.stderr.lower()
    return has_cycles, result.stdout + result.stderr

```python

def generate_dependency_data() -> dict:

```text

    """Generate dependency data as JSON."""
    result = subprocess.run(
        [
            "python", "-m", "pydeps", "src/podcast_scraper",
            "--show-deps", "--no-show", "--no-output"
        ],
        capture_output=True,
        text=True,
    )

```python

def analyze_imports(src_dir: Path = Path("src/podcast_scraper")) -> dict:

```python

    """Analyze import patterns in source files."""
    import_counts = {}

```python

        try:
            with open(py_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("import ") or line.startswith("from "):
                        imports.append(line)
        except Exception:
            continue

```

    return import_counts

```python

def check_thresholds(import_data: dict) -> list[str]:

```text

    """Check against architectural thresholds."""
    issues = []

```python

def main():

```text

    parser = argparse.ArgumentParser(description="Analyze module dependencies")
    parser.add_argument("--check", action="store_true", help="Check mode - exit with error if issues")
    parser.add_argument("--report", action="store_true", help="Generate JSON report")
    args = parser.parse_args()

```python

    # Analyze imports
    print("\n📊 Analyzing import patterns...")
    import_data = analyze_imports()
    issues = check_thresholds(import_data)

```

            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {report_path}")

```python

if __name__ == "__main__":

```text

    main()

```
### 2.5 CI Integration

Add to nightly workflow (`.github/workflows/nightly.yml`):

```yaml

  dependency-analysis:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5

        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |

          pip install pydeps
          pip install -e ".[dev]"

      - name: Check for circular imports
        run: |

          python -m pydeps src/podcast_scraper --show-cycles --no-show
        continue-on-error: true

      - name: Generate dependency graph
        run: |

          mkdir -p reports
          python -m pydeps src/podcast_scraper --cluster --max-bacon=2 -o reports/deps.svg

      - name: Run dependency analysis
        run: |

          python scripts/analyze_dependencies.py --report

      - name: Upload artifacts
        uses: actions/upload-artifact@v4

```text

        with:
          name: dependency-analysis
          path: reports/

```yaml

### 2.6 Key Metrics

| Metric | Description | Threshold | Action |
| -------- | ------------- | ----------- | -------- |
| Circular imports | Cycles in import graph | 0 | Must fix immediately |
| Max imports per module | Fan-out | ≤15 | Refactor if exceeded |
| Max dependency depth | Longest chain | ≤5 | Review architecture |

---

## 3. Pre-Release Checklist

### 3.1 Overview

Automated validation before releases to ensure quality gates are met.

### 3.2 Checklist Script

Create `scripts/pre_release_check.py`:

```python

#!/usr/bin/env python3
"""
Pre-release validation checklist.

Usage:
    python scripts/pre_release_check.py [--version X.Y.Z]
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str

def run_command(cmd: list[str], check: bool = False) -> tuple[int, str, str]:

```text

    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

```python

def check_tests() -> CheckResult:

```text

    """Verify all tests pass."""
    code, stdout, stderr = run_command(["make", "test"])
    return CheckResult(
        name="Tests",
        passed=code == 0,
        message="All tests passed" if code == 0 else f"Tests failed: {stderr[:200]}"
    )

```python

def check_lint() -> CheckResult:

```text

    """Verify linting passes."""
    code, _, stderr = run_command(["make", "lint"])
    return CheckResult(
        name="Linting",
        passed=code == 0,
        message="Linting passed" if code == 0 else f"Linting failed: {stderr[:200]}"
    )

```python

def check_type_check() -> CheckResult:

```text

    """Verify type checking passes."""
    code, _, stderr = run_command(["make", "type-check"])
    return CheckResult(
        name="Type Checking",
        passed=code == 0,
        message="Type check passed" if code == 0 else f"Type errors: {stderr[:200]}"
    )

```python

def check_docs_build() -> CheckResult:

```text

    """Verify documentation builds."""
    code, _, stderr = run_command(["make", "docs"])
    return CheckResult(
        name="Documentation",
        passed=code == 0,
        message="Docs build successfully" if code == 0 else f"Docs failed: {stderr[:200]}"
    )

```python

def check_security() -> CheckResult:

```text

    """Verify no critical security issues."""
    code, _, stderr = run_command(["make", "security"])
    return CheckResult(
        name="Security",
        passed=code == 0,
        message="No critical vulnerabilities" if code == 0 else f"Security issues: {stderr[:200]}"
    )

```python

def check_changelog(version: str | None) -> CheckResult:

```text

    """Verify changelog is updated."""
    changelog = Path("CHANGELOG.md")
    if not changelog.exists():
        return CheckResult(
            name="Changelog",
            passed=False,
            message="CHANGELOG.md not found"
        )

```python

def check_version_consistency(version: str | None) -> CheckResult:

```text

    """Verify version is consistent across files."""
    if not version:
        return CheckResult(
            name="Version Consistency",
            passed=True,
            message="Skipped (no version specified)"
        )

```
                name="Version Consistency",
                passed=False,
                message=f"Version {version} not in pyproject.toml"
            )

```python

def main():

```text

    parser = argparse.ArgumentParser(description="Pre-release validation")
    parser.add_argument("--version", help="Expected version number")
    parser.add_argument("--quick", action="store_true", help="Skip slow checks")
    args = parser.parse_args()

```
```python

if __name__ == "__main__":

```text

    main()

```
### 3.3 Makefile Target

```makefile

## Pre-release validation

pre-release:
	@echo "=== Pre-Release Validation ==="
	@$(PYTHON) scripts/pre_release_check.py

## Pre-release validation (quick - skip tests)

pre-release-quick:
	@echo "=== Pre-Release Validation (Quick) ==="
	@$(PYTHON) scripts/pre_release_check.py --quick

```yaml

---

## 4. Future Enhancements (Phase 3)

### 4.1 Memory Profiling

For future implementation:

```python

# Install: pip install memory_profiler

from memory_profiler import profile

@profile
def process_large_transcript():
    # Memory-intensive operation
    pass

```
## 4.2 Performance Benchmarking

For future implementation:

```python

# Install: pip install pytest-benchmark

def test_transcription_performance(benchmark):
    result = benchmark(transcribe_audio, audio_file)
    assert result is not None

```yaml

---

## 5. Implementation Summary

### 5.1 Files to Create/Modify

| File | Action | Description |
| ------ | -------- | ------------- |
| `.github/dependabot.yml` | Create | Dependabot configuration |
| `scripts/analyze_dependencies.py` | Create | Dependency analysis script |
| `scripts/pre_release_check.py` | Create | Pre-release validation |
| `Makefile` | Modify | Add deps-graph, pre-release targets |
| `pyproject.toml` | Modify | Add pydeps to dev dependencies |
| `.github/workflows/nightly.yml` | Modify | Add dependency analysis job |

### 5.2 Effort Estimate

| Task | Time | Issue |
| ------ | ------ | ------- |
| Dependabot configuration | 30 min | #169 |
| pydeps setup + Makefile | 1 hour | #170 |
| Dependency analysis script | 1 hour | #170 |
| CI integration | 30 min | #170 |
| Pre-release checklist | 1.5 hours | #45 |
| Documentation updates | 30 min | #45 |
| **Total** | **~5 hours** | |

### 5.3 Rollout Plan

1. **Week 1**: Dependabot (#169)
   - Create `.github/dependabot.yml`
   - Monitor first batch of PRs
   - Adjust groups if needed

2. **Week 2**: Module Coupling (#170)
   - Add pydeps + Makefile targets
   - Create analysis script
   - Add to nightly CI

3. **Week 3**: Pre-release Checklist
   - Create validation script
   - Add Makefile targets
   - Document in release process

---

## 6. Success Criteria

- [ ] Dependabot creates weekly PRs for dependency updates
- [ ] `make deps-graph` generates module dependency visualization
- [ ] `make deps-check-cycles` detects circular imports
- [ ] Nightly CI includes dependency analysis
- [ ] `make pre-release` validates release readiness
- [ ] All tools documented in ci/WORKFLOWS.md

## Related Documentation

- Issue #45: Continuous review parent issue
- Issue #169: Dependabot setup
- Issue #170: Module coupling analysis
- RFC-031: Code complexity tooling (related)
- docs/ci/index.md: CI/CD documentation
