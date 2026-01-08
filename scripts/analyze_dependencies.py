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
    """Check for circular dependencies."""
    result = subprocess.run(
        ["python", "-m", "pydeps", "src/podcast_scraper", "--show-cycles", "--no-show"],
        capture_output=True,
        text=True,
    )
    has_cycles = "cycle" in result.stdout.lower() or "cycle" in result.stderr.lower()
    return has_cycles, result.stdout + result.stderr


def generate_dependency_data() -> dict:
    """Generate dependency data as JSON."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "pydeps",
            "src/podcast_scraper",
            "--show-deps",
            "--no-show",
            "--no-output",
        ],
        capture_output=True,
        text=True,
    )
    # Parse pydeps output (simplified)
    lines = result.stdout.strip().split("\n") if result.stdout else []
    return {
        "module_count": len(lines),
        "raw_output": result.stdout,
    }


def analyze_imports(src_dir: Path = Path("src/podcast_scraper")) -> dict:
    """Analyze import patterns in source files."""
    import_counts = {}

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        module_name = str(py_file.relative_to(src_dir.parent)).replace("/", ".").replace(".py", "")
        imports = []

        try:
            with open(py_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("import ") or line.startswith("from "):
                        imports.append(line)
        except Exception:
            continue

        import_counts[module_name] = {
            "import_count": len(imports),
            "imports": imports[:10],  # First 10 for brevity
        }

    return import_counts


def check_thresholds(import_data: dict) -> list[str]:
    """Check against architectural thresholds."""
    issues = []

    MAX_IMPORTS = 15  # Maximum imports per module

    for module, data in import_data.items():
        if data["import_count"] > MAX_IMPORTS:
            issues.append(
                f"⚠️  {module}: {data['import_count']} imports " f"(threshold: {MAX_IMPORTS})"
            )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Analyze module dependencies")
    parser.add_argument("--check", action="store_true", help="Exit with error if issues found")
    parser.add_argument("--report", action="store_true", help="Generate detailed JSON report")
    args = parser.parse_args()

    print("=== Module Dependency Analysis ===\n")

    # Check for circular imports
    print("Checking for circular imports...")
    has_cycles, cycle_output = run_pydeps_cycles()
    if has_cycles:
        print("⚠️  Circular imports detected:")
        print(cycle_output)
    else:
        print("✓ No circular imports detected")

    # Analyze import patterns
    print("\nAnalyzing import patterns...")
    import_data = analyze_imports()
    print(f"✓ Analyzed {len(import_data)} modules")

    # Check thresholds
    print("\nChecking architectural thresholds...")
    issues = check_thresholds(import_data)
    if issues:
        print("⚠️  Modules exceeding import threshold:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ All modules within thresholds")

    # Generate report if requested
    if args.report:
        print("\nGenerating detailed report...")
        report = {
            "circular_imports": has_cycles,
            "circular_imports_output": cycle_output,
            "modules_analyzed": len(import_data),
            "import_data": import_data,
            "threshold_issues": issues,
        }
        report_path = Path("reports/deps-analysis.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to {report_path}")

    # Exit with error if check mode and issues found
    if args.check and (has_cycles or issues):
        print("\n❌ Dependency analysis found issues")
        sys.exit(1)

    print("\n✓ Dependency analysis complete")


if __name__ == "__main__":
    main()
