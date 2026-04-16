#!/usr/bin/env python3
"""Enforce the 3-tier ML/AI testing policy across the test corpus.

Checks:
  1. No pytest.importorskip() in unit tests
     Unit tests must only depend on [dev] extras. importorskip silently
     skips tests when an optional extra is missing, hiding failures in CI.

  1b. No FastAPI (or ``import fastapi``) imports in unit tests
     FastAPI lives in the ``[server]`` extra; PR / dev-venv unit jobs install
     ``.[dev]`` only. Route tests belong under ``tests/integration/server/``.

  2. No @pytest.mark.ml_models in integration tests
     Real ML models belong in tests/e2e/ only. Integration tests must
     mock all ML/AI boundaries.

  3. No *_AVAILABLE skip guards in unit tests
     Patterns like SUMMARIZER_AVAILABLE / SPACY_AVAILABLE that gate
     entire test classes are dead code when unit tests properly mock.

  4. No empty test files (zero test methods)
     Abandoned files with no test_ methods add noise and confusion.

See: docs/architecture/TESTING_STRATEGY.md (3-tier ML/AI boundary policy)

Usage:
    python scripts/tools/check_test_policy.py [--fix-hint]

Exit codes:
    0: All checks pass
    1: One or more violations found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent.parent.parent / "tests"
UNIT_DIR = TESTS_ROOT / "unit"
INTEGRATION_DIR = TESTS_ROOT / "integration"

AVAILABLE_GUARD_RE = re.compile(
    r"^\s*(SUMMARIZER_AVAILABLE|SPACY_AVAILABLE|WHISPER_AVAILABLE|"
    r"TRANSFORMERS_AVAILABLE|FASTAPI_AVAILABLE|HTTPX_AVAILABLE)\s*=",
    re.MULTILINE,
)

IMPORTORSKIP_RE = re.compile(r"pytest\.importorskip\s*\(")

# FastAPI / TestClient require ``pip install -e '.[server]'`` — not ``[dev]`` alone.
UNIT_FASTAPI_IMPORT_RE = re.compile(
    r"^\s*(from fastapi\b|import fastapi\b)",
    re.MULTILINE,
)

ML_MODELS_MARKER_RE = re.compile(r"@pytest\.mark\.ml_models")

TEST_METHOD_RE = re.compile(r"^\s+def (test_\w+)\(", re.MULTILINE)

# --- Allowlists ---
# Empty test file stubs: add paths here ONLY for files that are actively
# being developed in a tracked issue/PR. Prefer not creating the file
# until tests are ready.
ALLOWED_EMPTY_FILES: set[str] = set()

# Integration files allowed to use @pytest.mark.ml_models.
# Should be empty — all real-ML tests belong in tests/e2e/.
ALLOWED_INTEGRATION_ML: set[str] = set()


class Violation:
    def __init__(self, path: Path, line: int, rule: str, detail: str):
        self.path = path
        self.line = line
        self.rule = rule
        self.detail = detail

    def __str__(self) -> str:
        rel = self.path.relative_to(TESTS_ROOT.parent)
        return f"  {rel}:{self.line}: [{self.rule}] {self.detail}"


def find_test_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("test_*.py"))


def check_unit_importorskip(files: list[Path]) -> list[Violation]:
    violations = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in IMPORTORSKIP_RE.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            violations.append(
                Violation(
                    f,
                    lineno,
                    "U1-importorskip",
                    "pytest.importorskip() in unit test — unit tests must only "
                    "use [dev] extras; move to integration/ or mock the dependency",
                )
            )
    return violations


def check_unit_fastapi_imports(files: list[Path]) -> list[Violation]:
    violations = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in UNIT_FASTAPI_IMPORT_RE.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            violations.append(
                Violation(
                    f,
                    lineno,
                    "U3-fastapi-in-unit",
                    "FastAPI import in unit test — move to tests/integration/server/ "
                    "(unit CI uses .[dev] only; FastAPI is in .[server])",
                )
            )
    return violations


def check_unit_available_guards(files: list[Path]) -> list[Violation]:
    violations = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in AVAILABLE_GUARD_RE.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            var_name = m.group(1)
            violations.append(
                Violation(
                    f,
                    lineno,
                    "U2-available-guard",
                    f"{var_name} skip guard in unit test — unit tests should "
                    f"mock ML dependencies, not skip when absent",
                )
            )
    return violations


def check_integration_ml_models(files: list[Path]) -> list[Violation]:
    violations = []
    for f in files:
        rel = str(f.relative_to(TESTS_ROOT.parent))
        if rel in ALLOWED_INTEGRATION_ML:
            continue
        text = f.read_text(encoding="utf-8")
        for m in ML_MODELS_MARKER_RE.finditer(text):
            lineno = text[: m.start()].count("\n") + 1
            violations.append(
                Violation(
                    f,
                    lineno,
                    "I1-ml-models-marker",
                    "@pytest.mark.ml_models in integration test — real ML "
                    "models belong in tests/e2e/ only",
                )
            )
    return violations


def check_empty_test_files(files: list[Path]) -> list[Violation]:
    violations = []
    for f in files:
        if f.name == "conftest.py":
            continue
        rel = str(f.relative_to(TESTS_ROOT.parent))
        if rel in ALLOWED_EMPTY_FILES:
            continue
        text = f.read_text(encoding="utf-8")
        if not TEST_METHOD_RE.search(text):
            violations.append(
                Violation(
                    f,
                    1,
                    "G1-empty-test-file",
                    "test file has zero test methods — delete or add tests",
                )
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix-hint",
        action="store_true",
        help="Show fix suggestions for each violation",
    )
    args = parser.parse_args()

    print("Checking test policy compliance...")
    print("=" * 70)

    unit_files = find_test_files(UNIT_DIR)
    integration_files = find_test_files(INTEGRATION_DIR)
    all_test_files = find_test_files(TESTS_ROOT)

    violations: list[Violation] = []

    # Rule U1: No importorskip in unit tests
    violations.extend(check_unit_importorskip(unit_files))

    # Rule U3: No FastAPI imports in unit tests ([server] extra)
    violations.extend(check_unit_fastapi_imports(unit_files))

    # Rule U2: No *_AVAILABLE skip guards in unit tests
    violations.extend(check_unit_available_guards(unit_files))

    # Rule I1: No ml_models marker in integration tests
    violations.extend(check_integration_ml_models(integration_files))

    # Rule G1: No empty test files anywhere
    violations.extend(check_empty_test_files(all_test_files))

    # Summary
    rule_counts: dict[str, int] = {}
    for v in violations:
        rule_counts[v.rule] = rule_counts.get(v.rule, 0) + 1

    if violations:
        print(f"\nFound {len(violations)} violation(s):\n")
        for v in sorted(violations, key=lambda x: (x.rule, str(x.path), x.line)):
            print(v)

        if args.fix_hint:
            print("\nFix hints:")
            if "U1-importorskip" in rule_counts:
                print(
                    "  U1: Move test to tests/integration/ or mock the "
                    "dependency with unittest.mock.patch"
                )
            if "U3-fastapi-in-unit" in rule_counts:
                print(
                    "  U3: Move FastAPI / TestClient tests to "
                    "tests/integration/server/ (see UNIT_TESTING_GUIDE.md)"
                )
            if "U2-available-guard" in rule_counts:
                print(
                    "  U2: Remove the try/except guard and mock the ML "
                    "import instead (the module uses lazy imports)"
                )
            if "I1-ml-models-marker" in rule_counts:
                print(
                    "  I1: Move test to tests/e2e/ or mock the ML model "
                    "(integration tests must not use real ML)"
                )
            if "G1-empty-test-file" in rule_counts:
                print("  G1: Delete the file or add test methods")

        print(f"\nSummary: {', '.join(f'{v} {k}' for k, v in sorted(rule_counts.items()))}")
        print("\nSee: docs/architecture/TESTING_STRATEGY.md (3-tier ML/AI policy)")
        return 1
    else:
        stats = (
            f"{len(unit_files)} unit, "
            f"{len(integration_files)} integration, "
            f"{len(all_test_files)} total"
        )
        allowlisted = len(ALLOWED_EMPTY_FILES) + len(ALLOWED_INTEGRATION_ML)
        print(f"\nAll checks passed ({stats} test files scanned, " f"{allowlisted} allowlisted)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
