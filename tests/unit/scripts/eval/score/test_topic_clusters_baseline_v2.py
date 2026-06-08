"""Unit tests for `scripts/eval/score/topic_clusters_baseline_v2.py` (#903 audit follow-up)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_PATH = ROOT / "scripts" / "eval" / "score" / "topic_clusters_baseline_v2.py"
_spec = importlib.util.spec_from_file_location("topic_clusters_baseline_v2_under_test", _PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _cluster(tc_id: str, feeds: list[str], labels: list[str]) -> dict:
    return {
        "tc_id": tc_id,
        "label": labels[0],
        "member_count": len(labels),
        "unique_topic_count": len(labels),
        "feeds": feeds,
        "feed_count": len(feeds),
        "podcasts": [f.replace("feed-", "") for f in feeds],
        "podcast_count": len(feeds),
        "topic_ids": [f"topic:{lbl.lower().replace(' ', '-')}" for lbl in labels],
        "labels": labels,
    }


def test_frame_negative_test_passes_on_empty_clusters() -> None:
    result = _mod._frame_negative_test([])
    assert result["pass"] is True
    assert result["violations"] == []
    assert result["exercised"] is False


def test_frame_negative_test_passes_when_no_cross_feed_frame() -> None:
    """Current v2 corpus state: frame labels only in p04 → vacuous pass."""
    clusters = [
        _cluster("tc:frame", ["feed-p04"], ["frame composition", "frame and light"]),
        _cluster("tc:other", ["feed-p02", "feed-p05"], ["systems thinking"]),
    ]
    result = _mod._frame_negative_test(clusters)
    assert result["pass"] is True
    assert result["exercised"] is False  # vacuous — no non-p04 frame source


def test_frame_negative_test_fires_on_real_violation() -> None:
    """Cluster bundles a p04 frame label with a non-p04 frame label → FAIL."""
    clusters = [
        _cluster(
            "tc:frame",
            ["feed-p02", "feed-p04"],
            ["frame composition", "frame this issue"],
        ),
    ]
    result = _mod._frame_negative_test(clusters)
    assert result["pass"] is False
    assert len(result["violations"]) == 1
    assert result["exercised"] is True


def test_frame_token_word_boundary_not_substring() -> None:
    """`framework` / `framing` / `timeframe` must NOT trip the test."""
    clusters = [
        _cluster(
            "tc:framework",
            ["feed-p02", "feed-p04"],
            ["framework design", "framework adoption"],
        ),
    ]
    result = _mod._frame_negative_test(clusters)
    assert result["pass"] is True
    assert result["violations"] == []


def test_aggregate_emits_required_shape() -> None:
    clusters = [
        _cluster("tc:a", ["feed-p01", "feed-p02"], ["alpha"]),
        _cluster("tc:b", ["feed-p03"], ["beta"]),
    ]
    payload = {"clusters": clusters, "row_count": 42}
    agg = _mod.aggregate(payload)
    assert agg["tc_parent_count"] == 2
    assert agg["tc_cross_feed_count"] == 1
    assert agg["topic_row_count"] == 42
    assert agg["ac_pass"]["tc_parent_count"] is True
    assert agg["ac_pass"]["tc_cross_feed_count"] is True
    assert agg["ac_pass"]["frame_negative_test"] is True
    assert agg["tc_parents_per_podcast"]["p01"] == 1
    assert agg["tc_parents_per_podcast"]["p02"] == 1
    assert agg["tc_parents_per_podcast"]["p03"] == 1
