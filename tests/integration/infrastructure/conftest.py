"""Pytest configuration for infrastructure integration tests.

This module provides the e2e_server fixture for infrastructure tests that were
moved from tests/e2e/ to tests/integration/infrastructure/ as part of Phase 3.

Note: This conftest is in a subdirectory, so it won't interfere with other
integration tests. Pytest will load both this conftest and the parent conftest.
"""

import sys
from pathlib import Path

import pytest

# Add tests directory to path to import e2e_server fixture
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import e2e_server fixture from e2e fixtures
# This makes the fixture available to infrastructure tests
# Use absolute import to avoid conflicts
try:
    from tests.e2e.fixtures.e2e_http_server import e2e_server  # noqa: F401, E402
except ImportError:
    # Fallback: try relative import
    try:
        from ...e2e.fixtures.e2e_http_server import e2e_server  # noqa: F401, E402
    except ImportError:
        # If both fail, create a dummy fixture
        @pytest.fixture(scope="session")
        def e2e_server():
            """Dummy E2E server fixture (import failed)."""
            pytest.skip("E2E server fixture not available")
