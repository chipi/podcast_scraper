"""Unit tests for the user's heard∪captured episode set (P3 #1120, RFC-101 §1)."""

from __future__ import annotations

from podcast_scraper.server.app_user_corpus import derive_episode_set


def test_heard_requires_threshold_of_known_duration() -> None:
    playback = [
        {"slug": "ep-30pct", "position_seconds": 300},  # 300/1000 = 30% → heard
        {"slug": "ep-29pct", "position_seconds": 290},  # below 30% → not heard
        {"slug": "ep-nodur", "position_seconds": 999},  # unknown duration → not heard alone
    ]
    durations = {"ep-30pct": 1000.0, "ep-29pct": 1000.0}  # ep-nodur absent
    got = derive_episode_set(playback, [], durations)
    assert got == {"ep-30pct"}


def test_captured_always_qualifies_even_without_playback() -> None:
    got = derive_episode_set([], ["ep-hl", "ep-fav", ""], {})
    assert got == {"ep-hl", "ep-fav"}  # blanks dropped


def test_union_of_heard_and_captured() -> None:
    playback = [{"slug": "ep-heard", "position_seconds": 600}]
    durations = {"ep-heard": 1000.0}
    got = derive_episode_set(playback, ["ep-cap"], durations)
    assert got == {"ep-heard", "ep-cap"}


def test_custom_threshold() -> None:
    playback = [{"slug": "ep", "position_seconds": 100}]  # 10%
    durations = {"ep": 1000.0}
    assert derive_episode_set(playback, [], durations, threshold=0.05) == {"ep"}
    assert derive_episode_set(playback, [], durations, threshold=0.5) == set()


def test_malformed_playback_rows_are_skipped() -> None:
    playback: list[dict] = [
        {"position_seconds": 500},
        {"slug": "", "position_seconds": 5},
        {"slug": "ok"},
    ]
    durations = {"ok": 10.0}
    # 'ok' has no position_seconds → defaults 0 → not heard; no crash on missing slug/position
    assert derive_episode_set(playback, [], durations) == set()
