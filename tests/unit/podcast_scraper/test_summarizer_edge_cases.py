#!/usr/bin/env python3
"""Additional tests for summarization edge cases and error conditions."""

import os
import sys
import types
import unittest
from pathlib import Path

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper.providers.ml import summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.module_summarization]


# TestModelLoadingFailures and TestMemoryCleanup moved to
# tests/integration/test_summarizer_security_integration.py
# because they require filesystem I/O (tempfile.mkdtemp() in setUp/tearDown)


if __name__ == "__main__":
    unittest.main()
