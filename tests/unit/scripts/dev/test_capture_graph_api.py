"""Unit tests for the graph API perf capturer's aggregation math.

Loads scripts/dev/capture_graph_api.py directly (no running api / network).
Mirrors test_capture_search_api.py — the two capturers carry their own
percentile/summarize copies, so both need guarding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "dev" / "capture_graph_api.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capture_graph_api_module", SCRIPT_PATH)
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
    s = mod._summarize("scenario-x", [10, 20, 30, 40, 50], iterations=1, ok=5, sigsegv_free=True)
    assert s.name == "scenario-x"
    assert s.request_count == 5
    assert s.ok_count == 5
    assert s.p50_ms == pytest.approx(30.0)
    assert s.max_ms == 50.0
    assert s.sigsegv_free is True


def test_summarize_empty_latencies() -> None:
    mod = _load_module()
    s = mod._summarize("empty", [], iterations=0, ok=0, sigsegv_free=None)
    assert s.request_count == 0
    assert s.p50_ms == 0.0
    assert s.mean_ms == 0.0
