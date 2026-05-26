"""Pytest fixtures shared across the integration test suite.

Auto-generates the multi-run corpus fixture for the v2.6.1 hotfix tests
(``test_multi_run_corpus_fixture.py``) so contributors don't need a manual
generator step. The fixture is small + deterministic; regenerating it
takes ~50 ms.

Generation runs at conftest module-import time (before pytest collects
tests) so the test module's ``pytest.mark.skipif`` markers — evaluated at
collection — see the fixture present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "multi-run-corpus"
_GENERATOR_PATH = _REPO_ROOT / "scripts" / "tools" / "build_multi_run_fixture.py"


def _ensure_multi_run_corpus_fixture() -> None:
    """Generate the multi-run corpus fixture if missing."""
    if _FIXTURE_DIR.exists() and (_FIXTURE_DIR / "corpus_manifest.json").exists():
        return

    spec = importlib.util.spec_from_file_location("_build_multi_run_fixture", _GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generator at {_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_fixture(
        _FIXTURE_DIR,
        n_feeds=3,
        probe_episodes=1,
        middle_episodes=5,
        latest_episodes=5,
        overlap=3,
    )


# Module-level: runs at conftest import (before collection).
_ensure_multi_run_corpus_fixture()
