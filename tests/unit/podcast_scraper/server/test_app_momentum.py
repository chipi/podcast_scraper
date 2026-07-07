"""Unit tests for the RFC-103 Phase 3 momentum capability (``app_momentum``).

Covers the EWMA primitive, read-time trending anchored to a pinned reference week, cluster/storyline
aggregation, content⊕engagement blend, per-user scope, and the corpus engagement floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_momentum import (
    _weeks_ending,
    momentum,
    MomentumConfig,
    resolve_as_of_week,
    trending,
)

pytestmark = [pytest.mark.unit]

_CFG = MomentumConfig()
_NOW = "2026-07-01T00:00:00Z"


# --------------------------------------------------------------------------- #
# EWMA primitive
# --------------------------------------------------------------------------- #
def test_momentum_flat_is_about_one() -> None:
    v, _ = momentum([1] * 30, _CFG)
    assert 0.9 <= v <= 1.1  # steady → velocity ≈ 1.0


def test_momentum_recent_spike_is_rising() -> None:
    v, vol = momentum([0] * 28 + [5, 8], _CFG)
    assert v > 1.5 and vol > 0  # a recent burst reads as rising


def test_momentum_old_burst_has_cooled() -> None:
    v, _ = momentum([8, 8] + [0] * 28, _CFG)
    assert v < 1.0  # silent for weeks → fast EWMA decays below slow → cooling


# --------------------------------------------------------------------------- #
# trending — content
# --------------------------------------------------------------------------- #
def _write_content(root: Path, topics: list[dict], persons: list[dict] | None = None) -> None:
    env = {
        "enricher_id": "temporal_velocity",
        "status": "ok",
        "data": {"content_series": {"topics": topics, "persons": persons or []}},
    }
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "temporal_velocity.json").write_text(json.dumps(env), encoding="utf-8")


def _clusters(root: Path, rel: str, gpid: str, members: list[str]) -> None:
    payload = {
        "clusters": [
            {"graph_compound_parent_id": gpid, "members": [{"topic_id": m} for m in members]}
        ]
    }
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trending_topics_ranks_rising_first_and_flags_heating(tmp_path: Path) -> None:
    weeks = _weeks_ending(resolve_as_of_week(_NOW))
    r0, r1 = weeks[-2], weeks[-1]
    _write_content(
        tmp_path,
        topics=[
            {"topic_id": "topic:rising", "weekly_counts": {r0: 4, r1: 6}},
            {"topic_id": "topic:flat", "weekly_counts": {w: 1 for w in weeks[::4]}},
        ],
    )
    out = trending(tmp_path, None, kind="topic", now=_NOW, limit=10)
    ids = [t.entity_id for t in out]
    assert ids[0] == "topic:rising"
    top = out[0]
    assert top.heating_up and top.velocity >= _CFG.velocity_threshold
    # flat topic present but not heating up
    flat = next(t for t in out if t.entity_id == "topic:flat")
    assert not flat.heating_up


def test_trending_storyline_aggregates_member_topics(tmp_path: Path) -> None:
    weeks = _weeks_ending(resolve_as_of_week(_NOW))
    r0, r1 = weeks[-2], weeks[-1]
    _write_content(
        tmp_path,
        topics=[
            {"topic_id": "topic:a", "weekly_counts": {r0: 2, r1: 3}},
            {"topic_id": "topic:b", "weekly_counts": {r1: 4}},
        ],
    )
    _clusters(tmp_path, "enrichments/topic_theme_clusters.json", "thc:s", ["topic:a", "topic:b"])
    out = trending(tmp_path, None, kind="storyline", now=_NOW, limit=5)
    assert [t.entity_id for t in out] == ["thc:s"]
    # aggregated series = a+b: recent weeks carry 2 and 3+4=7 → clearly rising
    assert out[0].heating_up and out[0].total == 9


def test_as_of_week_anchoring_changes_result(tmp_path: Path) -> None:
    weeks_now = _weeks_ending(resolve_as_of_week(_NOW))
    _write_content(tmp_path, topics=[{"topic_id": "topic:x", "weekly_counts": {weeks_now[-1]: 9}}])
    # As of NOW, the spike is in the last week → rising.
    hot = trending(tmp_path, None, kind="topic", now=_NOW, limit=5)[0]
    assert hot.heating_up
    # As of a year later, that same spike is ~52 weeks stale → cooled (not heating).
    later = trending(tmp_path, None, kind="topic", now="2027-07-01T00:00:00Z", limit=5)
    assert later == [] or not later[0].heating_up


# --------------------------------------------------------------------------- #
# trending — engagement + blend + scope
# --------------------------------------------------------------------------- #
def _seed_opens(data_dir: Path, uid: str, slug: str, feed: str, week_iso_dates: list[str]) -> None:
    from datetime import datetime, timezone

    for d in week_iso_dates:
        ts = int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())
        app_user_state.append_listen_event(data_dir, uid, slug, feed, ts)


def test_trending_episodes_from_engagement_and_scope(tmp_path: Path) -> None:
    root, data = tmp_path / "corpus", tmp_path / "appdata"
    root.mkdir()
    _write_content(root, topics=[])  # no content for episodes
    _seed_opens(data, "u1", "ep-hot", "p05", ["2026-06-20T00:00:00", "2026-06-27T00:00:00"])
    _seed_opens(data, "u2", "ep-hot", "p05", ["2026-06-27T00:00:00"])
    # corpus scope: both users' opens; min_events floor (5) drops it → configure floor 0 here.
    cfg = MomentumConfig(min_events_corpus=0)
    corpus = trending(root, data, kind="episode", now=_NOW, scope="corpus", limit=5, config=cfg)
    assert corpus and corpus[0].entity_id == "ep-hot" and corpus[0].total == 3
    # scope=mine: only u1's two opens (no floor for mine).
    mine = trending(root, data, kind="episode", now=_NOW, scope="mine", user_id="u1", limit=5)
    assert mine[0].total == 2


def test_corpus_engagement_floor_hides_thin_signals(tmp_path: Path) -> None:
    root, data = tmp_path / "corpus", tmp_path / "appdata"
    root.mkdir()
    _write_content(root, topics=[])
    _seed_opens(data, "u1", "ep-thin", "p05", ["2026-06-27T00:00:00"])  # 1 event < floor 5
    corpus = trending(root, data, kind="episode", now=_NOW, scope="corpus", limit=5)
    assert corpus == []  # below min_events_corpus → hidden corpus-wide
