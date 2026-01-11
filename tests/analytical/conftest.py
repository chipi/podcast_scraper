"""Pytest configuration for analytical tests.

Analytical tests reuse E2E test infrastructure (E2E server, fixtures) but are
separate from regular E2E tests. They import fixtures from e2e/conftest.py.
"""

# Import E2E fixtures so analytical tests can use them
# This allows analytical tests to use e2e_server and other E2E infrastructure
import sys
from pathlib import Path

# Add e2e directory to path to import conftest
e2e_dir = Path(__file__).parent.parent / "e2e"
if str(e2e_dir) not in sys.path:
    sys.path.insert(0, str(e2e_dir))

# Import all fixtures from e2e conftest
from tests.e2e.conftest import *  # noqa: F401, F403
