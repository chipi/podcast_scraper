"""The guided start's show order (operator 2026-10-08): active first, then loved."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_guided_shows import (
    guided_show_signals,
    MAX_PER_CATEGORY,
    rank_guided_shows,
    ShowSignals,
)

NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def _row(feed: str, published: str, *, category: str = "", rel: str = "", slug: str = "") -> Any:
    return SimpleNamespace(
        feed_id=feed,
        publish_date=published,
        feed_category=category,
        metadata_relative_path=rel or f"metadata/{feed}-{published}.json",
        slug=slug,
    )


def _signals(**followers: int) -> ShowSignals:
    return ShowSignals(Counter(followers), Counter(), Counter(), Counter())


def _rank(rows: list[Any], signals: ShowSignals | None = None, **kw: Any) -> list[str]:
    args: dict[str, Any] = {
        "signals": signals or _signals(),
        "now": NOW,
        "followed": set(),
        "matching_relpaths": set(),
        "limit": 8,
    }
    args.update(kw)
    return rank_guided_shows(rows, **args)


def test_an_active_show_outranks_a_loved_one_that_stopped_publishing() -> None:
    rows = [_row("quiet", "2025-01-01"), _row("active", "2026-07-10")]
    assert _rank(rows, _signals(quiet=50)) == ["active", "quiet"]


def test_among_active_shows_the_most_loved_leads() -> None:
    rows = [_row("a", "2026-07-10"), _row("b", "2026-07-01"), _row("c", "2026-07-15")]
    assert _rank(rows, _signals(b=3, a=1)) == ["b", "a", "c"]


def test_with_no_love_anywhere_the_newest_leads() -> None:
    rows = [_row("old", "2026-07-01"), _row("new", "2026-07-18")]
    assert _rank(rows) == ["new", "old"]


def test_a_future_dated_episode_does_not_make_a_show_active() -> None:
    rows = [_row("future", "2027-01-01"), _row("now", "2026-07-10")]
    assert _rank(rows) == ["now", "future"]


def test_shows_already_followed_are_left_out() -> None:
    rows = [_row("a", "2026-07-10"), _row("b", "2026-07-10")]
    assert _rank(rows, followed={"a"}) == ["b"]


def test_a_show_carrying_a_chosen_interest_moves_up() -> None:
    rows = [
        _row("plain", "2026-07-18"),
        _row("match", "2026-07-01", rel="metadata/m1.json"),
    ]
    assert _rank(rows, matching_relpaths={"metadata/m1.json"}) == ["match", "plain"]


def test_no_category_takes_more_than_its_share_while_others_wait() -> None:
    rows = [_row(f"tech{i}", "2026-07-1%d" % i, category="Tech") for i in range(4)]
    rows.append(_row("art", "2026-07-01", category="Arts"))
    out = _rank(rows)
    assert out[: MAX_PER_CATEGORY + 1].count("art") == 1
    assert len(out) == 5  # the held-back shows still come after, not dropped


def test_limit_applies() -> None:
    rows = [_row(f"s{i}", "2026-07-10") for i in range(10)]
    assert len(_rank(rows, limit=3)) == 3


def test_signals_count_each_listener_once_per_show(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("podcast_scraper.server.app_guided_shows.slug_for_row", lambda r: r.slug)
    rows = [_row("A", "2026-07-10", slug="ep-a1"), _row("A", "2026-07-11", slug="ep-a2")]
    for uid in ("u1", "u2"):
        app_user_state.add_subscription(tmp_path, uid, {"feed_id": "A"})
        app_user_state.set_playback(tmp_path, uid, "ep-a1", 30.0, updated_at=1)
        app_user_state.set_playback(tmp_path, uid, "ep-a2", 30.0, updated_at=2)
    app_user_state.add_favorite(tmp_path, "u1", {"kind": "show", "ref": "A"})
    app_user_state.add_favorite(tmp_path, "u1", {"kind": "episode", "ref": "ep-a1"})

    s = guided_show_signals(tmp_path, rows)
    assert (s.followers["A"], s.listeners["A"]) == (2, 2)
    assert (s.show_favorites["A"], s.episode_favorites["A"]) == (1, 1)
