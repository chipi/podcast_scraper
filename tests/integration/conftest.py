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
import time as _real_time
from pathlib import Path
from unittest.mock import patch

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


def requires(*modules: str) -> "pytest.MarkDecorator":
    """Skip ONE test when an optional ML/search module is not importable.

    Integration tests may skip on a missing extra — unlike unit tests, where the same guard is
    banned (U1), because a unit test must not depend on a non-``[dev]`` extra in the first place.

    Applied per TEST, deliberately, never at module level. Every file that needs this also holds
    tests that pass without the extra — ``test_e2e_server.py`` is 2 ML-dependent tests beside 26
    that are not — so a module-level ``importorskip`` would have silenced about a hundred working
    integration tests to quiet a dozen failures. Skipping more than the thing that is actually
    unavailable is how a suite stops being evidence.

    Inert in CI: twelve jobs install ``.[dev,ml,llm,search]``, so the modules are present and every
    guarded test runs. It only bites on a machine that cannot install the ML stack at all — macOS
    x86_64, where torch/torchcodec publish no wheels.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"needs the [ml]/[search] extra — not importable: {', '.join(missing)}",
    )


class _NoBackoffSleep:
    """Stand-in for the ``time`` module used by ``retry_with_metrics``: ``sleep`` is a no-op;
    everything else (``time()``, ``monotonic()``, …) delegates to the real module."""

    def sleep(self, *_args, **_kwargs):  # retry-backoff no-op
        return None

    def __getattr__(self, name):
        return getattr(_real_time, name)


@pytest.fixture(autouse=True)
def _instant_retry_backoff():
    """No-op retry-backoff sleeps across the whole integration suite so tests don't
    wall-clock-wait real exponential backoff.

    Provider error / resilience tests mock the upstream API to fail, then run
    ``retry_with_metrics()``'s real ``time.sleep()`` schedule — e.g. Gemini's *production*
    config is 6 retries @ up to 60s backoff, so one ``test_summarize_api_error`` used to sit
    for ~123s (``docs/wip/nightly-test-time-analysis.md``). That wait tests nothing here: the
    retry SCHEDULE (delays/jitter/cap/exhaustion) is asserted with a mocked clock in
    ``tests/unit/.../test_provider_metrics.py``. Here we only need the retry PATH to run
    (N attempts → correct terminal error), which it still does — instantly.

    Module-scoped on the retry util's own ``time`` reference, so global ``time.sleep`` and
    every other module are untouched. One fixture → ALL providers + integration resilience
    tests, no per-directory gaps.
    """
    from podcast_scraper.utils import provider_metrics

    with patch.object(provider_metrics, "time", _NoBackoffSleep()):
        yield
