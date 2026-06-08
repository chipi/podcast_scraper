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

import pytest

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


@pytest.fixture(scope="session", autouse=True)
def _prewarm_minilm_embedding():
    """Pre-load MiniLM into the per-process embedding cache at session start.

    The offline search tests (test_aux_tier / test_two_tier_indexer / test_hybrid_search)
    embed via ``embedding_loader.encode`` -> ``get_embedding_model`` (process-cached, keyed
    by model/device/cache_dir/allow_download). Loading it once here -- at session start, when
    the on-disk HF cache is clean (right after the CI ensure step, before any test mutates HF
    state) -- means those tests hit the in-memory instance and never re-read the disk cache.
    This sidesteps a CI-only failure where a fresh offline load raised LocalEntryNotFoundError
    despite the model being on disk and loadable seconds earlier (in the separate ensure-step
    process). Best-effort: skip silently if MiniLM isn't cached (e.g. a subset run without it);
    the individual tests still gate on their own provisioning.
    """
    try:
        from podcast_scraper import config_constants as _cc
        from podcast_scraper.providers.ml.embedding_loader import get_embedding_model

        # Same args the search path uses (encode -> get_embedding_model defaults), so the
        # cache key matches and encode() gets a hit instead of a fresh disk load.
        get_embedding_model(_cc.DEFAULT_EMBEDDING_MODEL)
    except Exception:
        pass
    yield
