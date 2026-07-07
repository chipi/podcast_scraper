"""Unit tests for the RFC-103 Phase 2 engagement aggregator.

Seeds per-user engagement via the real writers (so on-disk formats are exercised) and asserts the
aggregator buckets every kind's events into per-entity weekly counts — corpus-wide and per-user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from podcast_scraper.server import app_ranking_telemetry, app_user_state
from podcast_scraper.server.app_engagement_series import engagement_series

pytestmark = [pytest.mark.unit]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


def _wk(iso: str) -> str:
    c = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).isocalendar()
    return f"{c.year:04d}-W{c.week:02d}"


def _by_entity(data: dict) -> dict[tuple[str, str], dict]:
    return {(e["kind"], e["entity_id"]): e for e in data["entities"]}


def test_engagement_series_covers_every_kind(tmp_path: Path) -> None:
    d = _wk("2026-01-19")  # a single ISO week for all the events below
    t = "2026-01-19T00:00:00"
    app_user_state.append_listen_event(tmp_path, "u1", "ep-index-investing", "p05", _ts(t))
    app_ranking_telemetry.record_click(
        tmp_path, "u1", slug="ep-index-investing", position=0, variant="v", ts=_ts(t)
    )
    # an impression must NOT count as engagement (only clicks do)
    app_ranking_telemetry.record_impressions(
        tmp_path, "u1", shown=["ep-index-investing"], variant="v", ts=_ts(t)
    )
    app_user_state.add_subscription(tmp_path, "u1", {"feed_id": "p05", "added_at": _ts(t)})
    app_user_state.add_favorite(
        tmp_path, "u1", {"kind": "insight", "ref": "ep#i1", "added_at": _ts(t)}
    )
    app_user_state.add_favorite(
        tmp_path, "u1", {"kind": "topic", "ref": "topic:ai", "added_at": _ts(t)}
    )
    app_user_state.record_interest_follow(tmp_path, "u1", "thc:managing-risk", _ts(t))
    app_user_state.record_interest_follow(tmp_path, "u1", "person:jane", _ts(t))

    e = _by_entity(engagement_series(tmp_path))
    # episode: 1 open + 1 click = 2 (impression ignored)
    assert e[("episode", "ep-index-investing")]["weekly_counts"] == {d: 2}
    # show: 1 open (feed_id on listen) + 1 subscribe = 2
    assert e[("show", "p05")]["total"] == 2
    assert e[("insight", "ep#i1")]["total"] == 1
    assert e[("topic", "topic:ai")]["total"] == 1
    assert e[("storyline", "thc:managing-risk")]["total"] == 1
    assert e[("person", "person:jane")]["total"] == 1


def test_engagement_series_aggregates_corpus_wide_and_scopes_to_user(tmp_path: Path) -> None:
    for uid in ("u1", "u2"):
        app_user_state.append_listen_event(tmp_path, uid, "ep-a", "p01", _ts("2026-02-02T00:00:00"))
    # corpus-wide: both users' opens sum
    corpus = _by_entity(engagement_series(tmp_path))
    assert corpus[("episode", "ep-a")]["total"] == 2
    # scope=mine (single user): just that user's opens
    mine = _by_entity(engagement_series(tmp_path, user_id="u1"))
    assert mine[("episode", "ep-a")]["total"] == 1


def test_engagement_series_weekly_axis_is_contiguous(tmp_path: Path) -> None:
    app_user_state.append_listen_event(tmp_path, "u1", "ep-a", "p01", _ts("2026-01-05T00:00:00"))
    app_user_state.append_listen_event(tmp_path, "u1", "ep-a", "p01", _ts("2026-02-16T00:00:00"))
    data = engagement_series(tmp_path)
    axis = data["window_weeks"]
    assert axis == sorted(axis)  # contiguous, oldest→newest
    assert _wk("2026-01-05") in axis and _wk("2026-02-16") in axis
    assert len(axis) >= 6  # ~6 weeks between the two events (a real gap, zero-filled at read)


def test_engagement_series_empty_without_users(tmp_path: Path) -> None:
    assert engagement_series(tmp_path) == {"window_weeks": [], "entities": []}
