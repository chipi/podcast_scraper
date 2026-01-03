#!/usr/bin/env python3
"""Audit E2E tests to categorize them for Phase 3 refactoring.

Categories:
- KEEP: True user workflows (CLI commands, library API, service API)
- MOVE_TO_INTEGRATION: Component interactions with real dependencies
- MOVE_TO_UNIT: Function-level tests with mocked dependencies
- REMOVE: Duplicate coverage, redundant tests
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

# Test file patterns and their likely categories
FILE_CATEGORIES = {
    # True E2E tests (keep)
    "test_basic_e2e.py": "KEEP",
    "test_cli_e2e.py": "KEEP",
    "test_library_api_e2e.py": "KEEP",
    "test_service_api_e2e.py": "KEEP",
    "test_full_pipeline_e2e.py": "KEEP",
    "test_multi_episode_e2e.py": "KEEP",
    "test_pipeline_concurrent_e2e.py": "KEEP",
    "test_pipeline_error_recovery_e2e.py": "KEEP",
    "test_openai_provider_integration_e2e.py": "KEEP",
    "test_data_quality_e2e.py": "KEEP",
    "test_http_behaviors_e2e.py": "KEEP",
    "test_error_handling_e2e.py": "KEEP",
    "test_edge_cases_e2e.py": "KEEP",
    "test_whisper_e2e.py": "KEEP",
    "test_ml_models_e2e.py": "KEEP",
    "test_e2e_server.py": "KEEP",
    "test_openai_mock.py": "KEEP",
    "test_fixture_mapping.py": "KEEP",
    "test_network_guard.py": "KEEP",
    "test_e2e_infrastructure.py": "KEEP",
    "test_env_variables.py": "KEEP",
    # Likely integration tests (move)
    "test_workflow_e2e.py": "MOVE_TO_INTEGRATION",  # Has @pytest.mark.integration
    # Likely unit tests (move)
    "test_cli.py": "MOVE_TO_UNIT",  # Function-level CLI tests
    "test_service.py": "MOVE_TO_UNIT",  # Function-level service tests
    "test_podcast_scraper.py": "MOVE_TO_UNIT",  # Function-level tests
    # Likely duplicates or infrastructure (review)
    "test_eval_scripts.py": "REVIEW",
}


def analyze_test_file(file_path: Path) -> Dict:
    """Analyze a test file to determine its category."""
    content = file_path.read_text(encoding="utf-8")

    # Check for markers
    has_e2e_marker = "@pytest.mark.e2e" in content
    has_integration_marker = "@pytest.mark.integration" in content
    has_unit_marker = "@pytest.mark.unit" in content

    # Check for mocking
    has_mock_import = "from unittest.mock import" in content or "import unittest.mock" in content
    has_patch = "@patch" in content or "patch(" in content
    has_mock = "Mock(" in content or "MagicMock(" in content

    # Check for real dependencies
    has_e2e_server = "e2e_server" in content
    has_real_http = "real HTTP" in content.lower() or "no mocking" in content.lower()

    # Check for CLI/library/service API usage
    uses_cli = "cli.main(" in content or "CLI" in content
    uses_library_api = "run_pipeline(" in content or "Library API" in content
    uses_service_api = "service.run(" in content or "Service API" in content

    # Count test functions
    test_count = len(re.findall(r"def test_\w+", content))

    return {
        "file": file_path.name,
        "has_e2e_marker": has_e2e_marker,
        "has_integration_marker": has_integration_marker,
        "has_unit_marker": has_unit_marker,
        "has_mock_import": has_mock_import,
        "has_patch": has_patch,
        "has_mock": has_mock,
        "has_e2e_server": has_e2e_server,
        "has_real_http": has_real_http,
        "uses_cli": uses_cli,
        "uses_library_api": uses_library_api,
        "uses_service_api": uses_service_api,
        "test_count": test_count,
    }


def categorize_test(analysis: Dict, file_category: str = None) -> str:
    """Categorize a test based on its analysis."""
    # Use file-level category if provided
    if file_category and file_category != "REVIEW":
        return file_category

    # If marked as integration, move to integration
    if analysis["has_integration_marker"]:
        return "MOVE_TO_INTEGRATION"

    # If has mocks but no e2e_server, likely unit test
    if (analysis["has_mock"] or analysis["has_patch"]) and not analysis["has_e2e_server"]:
        return "MOVE_TO_UNIT"

    # If uses CLI/library/service API with e2e_server, keep as E2E
    if analysis["has_e2e_server"] and (
        analysis["uses_cli"] or analysis["uses_library_api"] or analysis["uses_service_api"]
    ):
        return "KEEP"

    # If marked as e2e and uses real HTTP, keep
    if analysis["has_e2e_marker"] and analysis["has_real_http"]:
        return "KEEP"

    # Default to review
    return "REVIEW"


def main():
    """Main audit function."""
    e2e_dir = Path(__file__).parent.parent / "tests" / "e2e"

    results = defaultdict(list)
    total_tests = 0

    for test_file in sorted(e2e_dir.glob("test_*.py")):
        if test_file.name == "__init__.py":
            continue

        analysis = analyze_test_file(test_file)
        file_category = FILE_CATEGORIES.get(test_file.name, None)
        category = categorize_test(analysis, file_category)

        results[category].append(
            {
                "file": test_file.name,
                "category": category,
                "test_count": analysis["test_count"],
                "analysis": analysis,
            }
        )

        total_tests += analysis["test_count"]

    # Print summary
    print("=" * 80)
    print("E2E TEST AUDIT SUMMARY")
    print("=" * 80)
    print(f"\nTotal test files: {len([f for cat in results.values() for f in cat])}")
    print(f"Total test functions: {total_tests}")
    print()

    for category in ["KEEP", "MOVE_TO_INTEGRATION", "MOVE_TO_UNIT", "REVIEW", "REMOVE"]:
        if category not in results:
            continue

        files = results[category]
        test_count = sum(f["test_count"] for f in files)
        print(f"\n{category}: {len(files)} files, {test_count} tests")
        print("-" * 80)

        for file_info in sorted(files, key=lambda x: x["file"]):
            print(f"  {file_info['file']}: {file_info['test_count']} tests")
            analysis = file_info["analysis"]
            markers = []
            if analysis["has_e2e_marker"]:
                markers.append("e2e")
            if analysis["has_integration_marker"]:
                markers.append("integration")
            if analysis["has_mock"]:
                markers.append("mocks")
            if analysis["has_e2e_server"]:
                markers.append("e2e_server")
            if markers:
                print(f"    → {', '.join(markers)}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    move_to_integration = sum(f["test_count"] for f in results.get("MOVE_TO_INTEGRATION", []))
    move_to_unit = sum(f["test_count"] for f in results.get("MOVE_TO_UNIT", []))
    keep = sum(f["test_count"] for f in results.get("KEEP", []))
    review = sum(f["test_count"] for f in results.get("REVIEW", []))

    print(f"\n1. KEEP as E2E: {keep} tests")
    print(f"2. MOVE to Integration: {move_to_integration} tests")
    print(f"3. MOVE to Unit: {move_to_unit} tests")
    print(f"4. REVIEW: {review} tests")

    print("\nExpected after Phase 3:")
    e2e_pct = keep / total_tests * 100 if total_tests > 0 else 0
    print(f"  - E2E: {keep} tests (~{e2e_pct:.1f}%)")
    integration_total = 236 + move_to_integration
    unit_total = 1116 + move_to_unit
    total_after = total_tests + move_to_integration + move_to_unit
    integration_pct = (integration_total / total_after * 100) if total_after > 0 else 0
    unit_pct = (unit_total / total_after * 100) if total_after > 0 else 0
    print(f"  - Integration: {integration_total} tests (~{integration_pct:.1f}%)")
    print(f"  - Unit: {unit_total} tests (~{unit_pct:.1f}%)")


if __name__ == "__main__":
    main()
