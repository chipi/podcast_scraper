"""Unit tests for the RFC-103 Phase 3 momentum capability (``app_momentum``).

Covers the EWMA primitive, read-time trending anchored to a pinned reference week, cluster/storyline
aggregation, content⊕engagement blend, per-user scope, and the corpus engagement floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_momentum import (
    _weeks_ending,
    momentum,
    MomentumConfig,
    resolve_as_of_week,
    trending,
    window_momentum,
    WINDOW_MONTHS,
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


def _write_kg_person_episode(
    root: Path, *, stem: str, episode_id: str, persons: list[tuple[str, str, str]]
) -> None:
    """A minimal KG episode (metadata + kg.json) so ``_person_roles`` can read speaker roles."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    meta = {
        "feed": {"feed_id": "f1", "title": "Show"},
        "episode": {
            "episode_id": episode_id,
            "title": episode_id,
            "published_date": "2024-01-01T00:00:00",
        },
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    nodes = [
        {"id": pid, "type": "Person", "properties": {"name": n, "role": r}} for pid, n, r in persons
    ]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )


def _stamp(root: Path, mtime: float) -> None:
    import os

    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    os.utime(stamp, (mtime, mtime))


def test_trending_people_carry_headline_role(tmp_path: Path) -> None:
    """Person entities carry their strongest KG role (host>guest>mentioned); topics never do."""
    perf_cache.clear()
    weeks = _weeks_ending(resolve_as_of_week(_NOW))
    r0, r1 = weeks[-2], weeks[-1]
    _write_content(
        tmp_path,
        topics=[{"topic_id": "topic:ai", "weekly_counts": {r0: 3, r1: 5}}],
        persons=[
            {
                "person_id": "person:jane",
                "person_label": "Jane Doe",
                "weekly_counts": {r0: 4, r1: 6},
            },
            {"person_id": "person:bob", "person_label": "Bob", "weekly_counts": {r0: 3, r1: 5}},
            {"person_id": "person:zoe", "person_label": "Zoe", "weekly_counts": {r0: 2, r1: 4}},
        ],
    )
    # Jane guests in one episode and hosts another → host wins; Bob is only ever mentioned; Zoe
    # has no KG node at all → no role.
    _write_kg_person_episode(
        tmp_path,
        stem="0001",
        episode_id="e1",
        persons=[("person:jane", "Jane Doe", "guest"), ("person:bob", "Bob", "mentioned")],
    )
    _write_kg_person_episode(
        tmp_path,
        stem="0002",
        episode_id="e2",
        persons=[("person:jane", "Jane Doe", "host")],
    )
    _stamp(tmp_path, 1_000_000.0)

    people = {t.entity_id: t for t in trending(tmp_path, None, kind="person", now=_NOW, limit=10)}
    assert people["person:jane"].role == "host"
    assert people["person:bob"].role == "mentioned"
    assert people["person:zoe"].role is None

    # Role is a person-only concept — topics never carry it.
    topics = trending(tmp_path, None, kind="topic", now=_NOW, limit=5)
    assert topics and all(t.role is None for t in topics)
    perf_cache.clear()


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


# --------------------------------------------------------------------------- #
# trending — shows (RFC-103 §show: publishing cadence from the catalog)
# --------------------------------------------------------------------------- #
def _write_episode(root: Path, stem: str, feed_id: str, title: str, published: str) -> None:
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": feed_id, "title": title},
        "episode": {"episode_id": stem, "title": stem, "published_date": published},
        "summary": {"title": "s", "bullets": ["a"]},
    }
    (meta / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def test_trending_shows_from_publishing_cadence(tmp_path: Path) -> None:
    """A show that shipped episodes in recent weeks trends (content = publishing cadence),
    labelled by its title — not the feed id."""
    for i, wk in enumerate(("2026-06-08", "2026-06-15", "2026-06-22", "2026-06-29")):
        _write_episode(tmp_path, f"e{i}", "myfeed", "My Show", wk)
    rows = trending(tmp_path, None, kind="show", now=_NOW)
    shows = {r.entity_id: r for r in rows}
    assert "myfeed" in shows
    assert shows["myfeed"].label == "My Show"  # real title, not the raw feed id
    assert shows["myfeed"].total == 4
    assert shows["myfeed"].velocity > 1.0  # recent, regular publishing → rising


# --------------------------------------------------------------------------- #
# RFC-103 Revision 2 — monthly window momentum, min_total inclusion, ranking
# --------------------------------------------------------------------------- #
def test_window_momentum_flat_is_about_one() -> None:
    # A steady monthly series reads flat regardless of window length.
    v, vol, total = window_momentum([2] * 12, 3, _CFG)
    assert 0.9 <= v <= 1.1 and total == 6 and vol == 6.0


def test_window_momentum_recent_rise_is_rising() -> None:
    # 9 months at 1/mo, last 3 months at 6/mo → recent rate 6× the prior baseline.
    v, _, total = window_momentum([1] * 9 + [6, 6, 6], 3, _CFG)
    assert v > 1.5 and total == 18


def test_window_momentum_cooled_is_below_one() -> None:
    # Busy early, quiet lately → recent rate far below the prior baseline.
    v, _, total = window_momentum([6] * 9 + [1, 0, 0], 3, _CFG)
    assert v < 1.0 and total == 1


def test_window_momentum_new_entity_is_capped() -> None:
    # Leading zeros are trimmed, so an entity that exists ONLY inside the window has no prior
    # baseline → it reads as rising at the configured cap (a genuinely new topic).
    v, _, total = window_momentum([0] * 9 + [3, 3, 3], 3, _CFG)
    assert v == _CFG.new_entity_velocity and total == 9


def test_window_length_changes_the_window_total() -> None:
    monthly = [1] * 11 + [10]
    _, _, t1 = window_momentum(monthly, WINDOW_MONTHS["1m"], _CFG)
    _, _, t3 = window_momentum(monthly, WINDOW_MONTHS["3m"], _CFG)
    assert t1 == 10 and t3 == 12  # 1m = last month; 3m = last three


def test_min_total_is_a_list_inclusion_floor(tmp_path: Path) -> None:
    # The RFC's degenerate case: a one-off mention is an anecdote, not a trend. min_total (=3)
    # EXCLUDES it from the list — it is not merely left un-badged.
    weeks = _weeks_ending(resolve_as_of_week(_NOW))
    _write_content(
        tmp_path,
        topics=[
            {"topic_id": "topic:real", "weekly_counts": {weeks[-1]: 2, weeks[-2]: 2, weeks[-3]: 2}},
            {"topic_id": "topic:oneoff", "weekly_counts": {weeks[-1]: 1}},
        ],
    )
    ids = [t.entity_id for t in trending(tmp_path, None, kind="topic", now=_NOW, limit=10)]
    assert "topic:real" in ids and "topic:oneoff" not in ids


def test_ranking_is_monotonic_in_volume_at_equal_velocity(tmp_path: Path) -> None:
    # velocity × volume: with the same trajectory shape (same velocity), the better-covered topic
    # ranks first — the co-factor that stops a thin spike from topping a well-covered rise.
    weeks = _weeks_ending(resolve_as_of_week(_NOW))
    w = weeks[-3:]
    _write_content(
        tmp_path,
        topics=[
            {"topic_id": "topic:small", "weekly_counts": {w[0]: 1, w[1]: 1, w[2]: 1}},
            {"topic_id": "topic:big", "weekly_counts": {w[0]: 3, w[1]: 3, w[2]: 3}},
        ],
    )
    ids = [t.entity_id for t in trending(tmp_path, None, kind="topic", now=_NOW, limit=10)]
    assert ids.index("topic:big") < ids.index("topic:small")


def test_anchor_defaults_to_corpus_latest_month_not_wall_clock(tmp_path: Path, monkeypatch) -> None:
    # With no `now` / APP_TRENDING_NOW, the anchor is the corpus's latest CONTENT month, so a corpus
    # whose newest episode is weeks old still trends (the wall-clock-anchored bug returned nothing).
    monkeypatch.delenv("APP_TRENDING_NOW", raising=False)
    # Build a series ending well in the past relative to any real "now".
    weeks = _weeks_ending("2025-W20")  # anchor the fixture ~a year before today
    w = weeks[-3:]
    _write_content(
        tmp_path,
        topics=[{"topic_id": "topic:past", "weekly_counts": {w[0]: 3, w[1]: 4, w[2]: 5}}],
    )
    out = trending(tmp_path, None, kind="topic", limit=5)  # no now → corpus-anchored
    assert [t.entity_id for t in out] == ["topic:past"]
    assert out[0].total >= _CFG.min_total  # it counted, rather than falling outside a now-window
