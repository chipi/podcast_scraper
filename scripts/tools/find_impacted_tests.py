#!/usr/bin/env python3
"""Find impacted tests for changed source files.

This script maps source files to pytest markers, allowing discovery of
tests that should be run when specific files are changed.

Usage:
    python scripts/tools/find_impacted_tests.py \
        --files src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py \
        --test-type unit  # optional: unit, integration, e2e, or all
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set

# Map source file patterns to module markers
MODULE_MAPPING = {
    # Config
    "config.py": "module_config",
    "config_constants.py": "module_config",
    # CLI
    "cli.py": "module_cli",
    # Service
    "service.py": "module_service",
    # Workflow
    "workflow/": "module_workflow",
    # RSS
    "rss/downloader.py": "module_downloader",
    "rss/parser.py": "module_rss_parser",
    # Providers
    "providers/ml/": "module_ml_providers",
    "providers/openai/": "module_openai_providers",
    # Summarization
    "summarization/": "module_summarization",
    # Transcription
    "transcription/": "module_transcription",
    # Speaker detection
    "speaker_detectors/": "module_speaker_detection",
    # Preprocessing
    "preprocessing/": "module_preprocessing",
    # Cache
    "cache/": "module_cache",
    # Evaluation
    "evaluation/": "module_evaluation",
    # Utils
    "utils/": "module_utils",
    # Models
    "models.py": "module_models",
    # Exceptions
    "exceptions.py": "module_exceptions",
}


def extract_module_marker(file_path: str) -> str | None:
    """Extract module marker from source file path.

    Args:
        file_path: Path to source file (e.g., "src/podcast_scraper/config.py")

    Returns:
        Module marker (e.g., "module_config") or None if not found
    """
    # Normalize path
    path = Path(file_path)
    # Remove src/podcast_scraper/ prefix if present
    if "podcast_scraper" in path.parts:
        idx = path.parts.index("podcast_scraper")
        relative_path = Path(*path.parts[idx + 1 :])
    else:
        relative_path = path

    # Check exact matches first (most specific)
    file_str = str(relative_path)
    for pattern, marker in MODULE_MAPPING.items():
        if pattern.endswith("/"):
            # Directory pattern
            if file_str.startswith(pattern):
                return marker
        else:
            # File pattern
            if file_str == pattern or file_str.endswith(f"/{pattern}"):
                return marker

    return None


def build_marker_expression(
    markers: Set[str], test_type: str | None = None, fast_only: bool = False
) -> str:
    """Build pytest marker expression.

    Args:
        markers: Set of module markers
        test_type: Filter by test type (unit, integration, e2e, or None for all)
        fast_only: Only include critical_path tests

    Returns:
        Pytest marker expression string
    """
    if not markers:
        return ""

    # Build base expression with OR for multiple markers
    if len(markers) == 1:
        base_expr = list(markers)[0]
    else:
        base_expr = " or ".join(sorted(markers))

    # Add test type filter
    if test_type:
        if test_type == "unit":
            base_expr = f"({base_expr}) and unit"
        elif test_type == "integration":
            base_expr = f"({base_expr}) and integration"
        elif test_type == "e2e":
            base_expr = f"({base_expr}) and e2e"

    # Add fast_only filter
    if fast_only:
        base_expr = f"({base_expr}) and critical_path"

    return base_expr


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find impacted tests for changed source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Source files to analyze (e.g., src/podcast_scraper/config.py)",
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "e2e", "all"],
        default="all",
        help="Filter by test type (default: all)",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Only include critical_path tests",
    )
    parser.add_argument(
        "--output-format",
        choices=["makefile", "pytest", "list", "expression"],
        default="pytest",
        help="Output format (default: pytest)",
    )

    args = parser.parse_args()

    # Extract markers from files
    markers: Set[str] = set()
    unmapped_files: List[str] = []

    for file_path in args.files:
        marker = extract_module_marker(file_path)
        if marker:
            markers.add(marker)
            print(f"  {file_path} → {marker}", file=sys.stderr)
        else:
            unmapped_files.append(file_path)
            print(f"  ⚠️  {file_path} → (no mapping found)", file=sys.stderr)

    if unmapped_files:
        print(
            f"\n⚠️  Warning: {len(unmapped_files)} file(s) could not be mapped to modules:",
            file=sys.stderr,
        )
        for f in unmapped_files:
            print(f"    {f}", file=sys.stderr)

    if not markers:
        print("❌ Error: No modules found for any files", file=sys.stderr)
        return 1

    # Build marker expression
    test_type = None if args.test_type == "all" else args.test_type
    marker_expr = build_marker_expression(markers, test_type, args.fast_only)

    # Output based on format
    if args.output_format == "makefile":
        # Output for Makefile (shell variable)
        print(f"TEST_MARKERS={marker_expr}")
    elif args.output_format == "pytest":
        # Output pytest command
        print(f"pytest -m '{marker_expr}'")
    elif args.output_format == "expression":
        # Output just the marker expression (for Makefile)
        print(marker_expr)
    else:  # list
        # Output list of markers
        print("\n".join(sorted(markers)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
