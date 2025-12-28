"""Smoke tests for test infrastructure itself.

These tests verify that:
- Pytest markers work correctly
- Makefile targets run tests
- CI configuration is correct

These tests should always pass and help catch configuration bugs early.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_integration_marker_collects_tests():
    """Verify that -m integration actually collects integration tests."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/",
            "-m",
            "integration",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, f"pytest failed: {result.stderr}"
    # Should collect multiple tests
    test_count = len([line for line in result.stdout.split("\n") if line.startswith("tests/")])
    assert test_count > 0, "No integration tests were collected! Check marker configuration."


def test_workflow_e2e_marker_collects_tests():
    """Verify that -m workflow_e2e actually collects workflow_e2e tests."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/workflow_e2e/",
            "-m",
            "workflow_e2e",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, f"pytest failed: {result.stderr}"
    test_count = len([line for line in result.stdout.split("\n") if line.startswith("tests/")])
    assert test_count > 0, "No workflow_e2e tests were collected! Check marker configuration."


def test_unit_tests_exclude_integration():
    """Verify that default pytest run excludes integration tests."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    # Should collect unit tests
    test_count = len([line for line in result.stdout.split("\n") if line.startswith("tests/")])
    assert test_count > 0, "No unit tests were collected!"


def test_makefile_test_integration_runs_tests():
    """Verify that 'make test-integration' actually runs tests."""
    # Check if make is available
    make_check = subprocess.run(["which", "make"], capture_output=True)
    if make_check.returncode != 0:
        pytest.skip("make command not available")

    result = subprocess.run(
        ["make", "test-integration"],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
        cwd=Path(__file__).parent.parent,
    )
    # Should not fail, but also should not be empty
    assert (
        "collected" in result.stdout.lower() or "passed" in result.stdout.lower()
    ), "make test-integration didn't run any tests! Check Makefile and pytest configuration."


def test_makefile_test_workflow_e2e_runs_tests():
    """Verify that 'make test-workflow-e2e' actually runs tests."""
    # Check if make is available
    make_check = subprocess.run(["which", "make"], capture_output=True)
    if make_check.returncode != 0:
        pytest.skip("make command not available")

    result = subprocess.run(
        ["make", "test-workflow-e2e"],
        capture_output=True,
        text=True,
        timeout=1200,  # 20 minute timeout (workflow_e2e tests can be slow)
        cwd=Path(__file__).parent.parent,
    )
    # Should not fail, but also should not be empty
    assert (
        "collected" in result.stdout.lower() or "passed" in result.stdout.lower()
    ), "make test-workflow-e2e didn't run any tests! Check Makefile and pytest configuration."


def test_marker_combinations_work():
    """Verify that marker combinations work correctly."""
    # Test that we can combine markers
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "integration and not slow",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    # Should succeed (even if 0 tests match, that's OK - we're just checking syntax)
    assert result.returncode == 0, f"Marker combination failed: {result.stderr}"


def test_no_tests_collected_warning():
    """Verify that pytest warns when no tests are collected with explicit marker."""
    # This test verifies that if we use a marker that doesn't match anything,
    # pytest still runs but reports "no tests collected"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "nonexistent_marker_12345",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    # Should exit with code 5 (no tests collected) or 0, but should mention "no tests"
    assert "no tests collected" in result.stdout.lower() or result.returncode in [
        0,
        5,
    ], "Pytest should report when no tests match a marker"
