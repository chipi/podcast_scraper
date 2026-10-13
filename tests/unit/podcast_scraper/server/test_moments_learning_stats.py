"""Short listens and learning in the stats and the recap (operator 2026-10-12).

Stats counted listening by show and episode only. A short listen saved no position and accrued no
listening time, so it left no trace; a learning day was a bare index with no time; a path that
ended was deleted. These pin what is now recorded and what the numbers built from it say.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from podcast_scraper.server import app_recap, app_stats, app_user_state

DAY = 86_400
#: Noon UTC, 2026-10-12: far from a day boundary, so the listener's day is unambiguous.
NOW = int(datetime(2026, 10, 12, 12, tzinfo=timezone.utc).timestamp())


def _path(subject_id: str, kind: str, label: str, steps: int, started_at: int) -> dict:
    return {
        "subject_id": subject_id,
        "subject_kind": kind,
        "subject_label": label,
        "started_at": started_at,
        "steps": [{"episode_slug": f"e{i}"} for i in range(steps)],
        "done": [],
    }


def _clip(tmp: Path, slug: str, at: int, *, seconds: float = 30.0, subject=None, insight="i1"):
    rec = {"kind": "clip", "slug": slug, "insight_id": insight, "seconds": seconds}
    if subject:
        rec.update(subject_id=subject[0], subject_kind=subject[1], subject_label=subject[2])
    app_user_state.append_moment_heard(tmp, "u", rec, at)


def test_a_learning_day_records_when_and_a_finished_path_outlives_leaving(tmp_path: Path) -> None:
    app_user_state.add_learning_path(
        tmp_path, "u", _path("thc:risk", "storyline", "Managing risk", 3, NOW - 2 * DAY), cap=3
    )
    for i, at in enumerate((NOW - 2 * DAY, NOW - DAY, NOW)):
        app_user_state.mark_learning_step_done(tmp_path, "u", "thc:risk", i, now=at)
    # Done again later: the first time stands.
    app_user_state.mark_learning_step_done(tmp_path, "u", "thc:risk", 0, now=NOW + 5)

    path = app_user_state.get_learning_path(tmp_path, "u", "thc:risk")
    assert path is not None
    assert path["done_at"] == {"0": NOW - 2 * DAY, "1": NOW - DAY, "2": NOW}
    assert path["finished_at"] == NOW

    stats = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert stats["paths_finished"] == 1
    assert stats["learning_days_total"] == 3
    assert stats["learning_streak"] == 3

    # Leaving a finished path does not record it a second time, as left.
    assert app_user_state.remove_learning_path(tmp_path, "u", "thc:risk", now=NOW + 10)
    stats = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert (stats["paths_finished"], stats["paths_left"]) == (1, 0)
    assert stats["learning_days_total"] == 3, "its days stay counted after the path is gone"
    assert stats["learning_now"] == []


def test_a_path_left_early_or_swapped_out_is_kept_as_left(tmp_path: Path) -> None:
    for sid in ("tc:a", "tc:b"):
        app_user_state.add_learning_path(
            tmp_path, "u", _path(sid, "theme", sid, 7, NOW - DAY), cap=2
        )
    app_user_state.mark_learning_step_done(tmp_path, "u", "tc:a", 0, now=NOW - DAY)
    app_user_state.remove_learning_path(tmp_path, "u", "tc:a", now=NOW)
    # At the cap, starting another replaces one: that one ended too.
    app_user_state.add_learning_path(
        tmp_path, "u", _path("tc:c", "theme", "c", 7, NOW), cap=1, replace="tc:b"
    )
    history = app_user_state.get_learning_history(tmp_path, "u")
    assert [(h["subject_id"], h["outcome"], h["done"]) for h in history] == [
        ("tc:a", "left", 1),
        ("tc:b", "left", 0),
    ]
    stats = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert stats["paths_left"] == 2
    assert [p["subject_id"] for p in stats["learning_now"]] == ["tc:c"]
    assert stats["learning_now"][0]["day"] == 1


def test_moments_are_counted_and_a_redelivered_clip_once(tmp_path: Path) -> None:
    _clip(tmp_path, "ep1", NOW - 60, seconds=40)
    _clip(tmp_path, "ep1", NOW - 60, seconds=40)  # the same event, delivered twice
    _clip(tmp_path, "ep2", NOW - 30, seconds=20, insight="i2")
    stats = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert stats["moments_heard_total"] == 2
    assert stats["moments_seconds_total"] == 60.0


def test_the_recap_puts_moments_and_learning_beside_the_listening(tmp_path: Path) -> None:
    risk = ("topic:risk", "topic", "risk management")
    _clip(tmp_path, "ep1", NOW - 3600, seconds=45, subject=risk)
    _clip(tmp_path, "ep2", NOW - 3500, seconds=15, subject=risk, insight="i2")
    _clip(tmp_path, "ep3", NOW - 20 * DAY, seconds=99, subject=risk, insight="i3")  # last month
    # ep1 then played in full; ep2 kept listening from the clip.
    app_user_state.append_listen_event(tmp_path, "u", "ep1", "f1", NOW - 1800)
    app_user_state.append_moment_heard(tmp_path, "u", {"kind": "keep", "slug": "ep2"}, NOW - 3400)
    app_user_state.add_learning_path(
        tmp_path, "u", _path("thc:risk", "storyline", "Managing risk", 7, NOW - DAY), cap=3
    )
    app_user_state.mark_learning_step_done(tmp_path, "u", "thc:risk", 0, now=NOW - DAY)

    r = app_recap.build_recap(tmp_path, "u", "week", NOW)
    assert (r["moments_heard"], r["moments_seconds"]) == (2, 60.0)
    assert r["listening_seconds"] == 0, "a short listen is not listening time"
    assert r["learning_days"] == 1
    assert (r["moment_episodes"], r["moment_to_full"]) == (2, 2)
    assert r["distinct_shows"] == 1
    by_id = {s["id"]: s for s in r["subjects"]}
    assert by_id["topic:risk"]["moments"] == 2 and by_id["topic:risk"]["kind"] == "topic"
    assert by_id["thc:risk"]["learning_days"] == 1 and by_id["thc:risk"]["kind"] == "storyline"
    assert by_id["topic:risk"]["is_new"] is True

    month = app_recap.build_recap(tmp_path, "u", "month", NOW)
    assert month["moments_heard"] == 3


def test_a_moment_not_followed_by_the_episode_does_not_count_as_led_to_it(tmp_path: Path) -> None:
    app_user_state.append_listen_event(tmp_path, "u", "ep1", "f1", NOW - 7200)  # BEFORE the clip
    _clip(tmp_path, "ep1", NOW - 3600)
    r = app_recap.build_recap(tmp_path, "u", "week", NOW)
    assert (r["moment_episodes"], r["moment_to_full"]) == (1, 0)


def test_captures_say_where_they_were_saved(tmp_path: Path) -> None:
    for i, source in enumerate((None, "moment", "learn", "learn")):
        row = {"id": f"h{i}", "episode_slug": "ep1", "kind": "span", "created_at": NOW}
        if source:
            row["source"] = source
        app_user_state.add_highlight_if_absent(tmp_path, "u", row)
    stats = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert (
        stats["capture_from_episode"],
        stats["capture_from_moment"],
        stats["capture_from_learn"],
    ) == (1, 1, 2)


def test_clearing_listening_history_clears_the_moments_too(tmp_path: Path) -> None:
    _clip(tmp_path, "ep1", NOW)
    assert "moments_heard.jsonl" in app_user_state.clear_listening_history(tmp_path, "u")
    assert app_user_state.list_moments_heard(tmp_path, "u") == []


def test_the_day_a_path_is_on_counts_days_opened_early(tmp_path: Path) -> None:
    """ "Day 1 of 7 · 2 done" read wrong once the second day was opened ahead (2026-10-12)."""
    app_user_state.add_learning_path(
        tmp_path, "u", _path("tc:rel", "theme", "reliability", 7, NOW), cap=3
    )
    app_user_state.mark_learning_step_done(tmp_path, "u", "tc:rel", 0, now=NOW)
    # On the test's clock: on the wall clock, from a day after NOW the second day is open by itself
    # and undone, so nothing more opens (this failed on 2026-10-13, the day after it was written).
    assert app_user_state.open_next_learning_day(tmp_path, "u", "tc:rel", now=NOW) is not None
    stats = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert stats["learning_now"][0]["day"] == 2
