"""Unit tests for the Search v3 API perf capturer (S0(d)).

Loads scripts/dev/capture_search_api.py directly (not via the package) so the
tests don't need a running api or network. Covers the pure math functions
(percentile, summarize) + the queries-by-intent loader.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "dev" / "capture_search_api.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capture_search_api_module", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_percentile_median_and_extremes() -> None:
    mod = _load_module()
    vals = [10, 20, 30, 40, 50]
    assert mod._percentile(vals, 50) == pytest.approx(30.0)
    assert mod._percentile(vals, 0) == pytest.approx(10.0)
    assert mod._percentile(vals, 100) == pytest.approx(50.0)


def test_percentile_empty_is_zero() -> None:
    mod = _load_module()
    assert mod._percentile([], 50) == 0.0


def test_percentile_single_value() -> None:
    mod = _load_module()
    assert mod._percentile([42], 95) == pytest.approx(42.0)


def test_summarize_shape() -> None:
    mod = _load_module()
    s = mod._summarize("scenario-x", [10, 20, 30, 40, 50], iterations=1, ok=5, sigsegv_free=None)
    assert s.name == "scenario-x"
    assert s.request_count == 5
    assert s.ok_count == 5
    assert s.p50_ms == pytest.approx(30.0)
    assert s.max_ms == 50.0
    assert s.sigsegv_free is None


def test_summarize_empty_latencies() -> None:
    mod = _load_module()
    s = mod._summarize("empty", [], iterations=0, ok=0, sigsegv_free=True)
    assert s.request_count == 0
    assert s.p50_ms == 0.0
    assert s.mean_ms == 0.0
    assert s.sigsegv_free is True


def test_queries_by_intent_groups_correctly(tmp_path: Path) -> None:
    mod = _load_module()
    payload = {
        "queries": [
            {"id": "a", "q": "one", "intent_expected": "entity_lookup"},
            {"id": "b", "q": "two", "intent_expected": "entity_lookup"},
            {"id": "c", "q": "three", "intent_expected": "semantic"},
            {"id": "d", "q": "four", "intent_expected": None},  # falls into "unknown"
        ]
    }
    fp = tmp_path / "q.json"
    fp.write_text(json.dumps(payload))
    got = mod._queries_by_intent(fp)
    assert set(got.keys()) == {"entity_lookup", "semantic", "unknown"}
    assert got["entity_lookup"] == ["one", "two"]
    assert got["semantic"] == ["three"]
    assert got["unknown"] == ["four"]


def test_queries_by_intent_uses_real_search_queries_fixture() -> None:
    """The shipped search-queries.json must load into 5 intent buckets."""
    mod = _load_module()
    fixture = (
        REPO_ROOT / "tests" / "fixtures" / "viewer-validation-corpus" / "v3" / "search-queries.json"
    )
    got = mod._queries_by_intent(fixture)
    expected = {
        "entity_lookup",
        "raw_evidence",
        "temporal_tracking",
        "cross_show_synthesis",
        "semantic",
    }
    assert set(got.keys()) >= expected, f"missing intent buckets: {expected - set(got.keys())}"
    for intent in expected:
        assert len(got[intent]) >= 5, f"intent {intent} has fewer than 5 queries"
