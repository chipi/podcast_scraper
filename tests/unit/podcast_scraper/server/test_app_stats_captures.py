"""Capture + review-loop stats (operator 2026-09-18).

Stats were listening-only, which measured consumption and said nothing about the half of the
product that is the user's own writing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from podcast_scraper.server import app_stats, app_user_state

NOW = 1_800_000_000
DAY = 86_400


def _write(root: Path, uid: str, name: str, payload: object) -> None:
    d = root / "users" / uid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _hl(hid: str, slug: str, kind: str, created: int) -> dict:
    return {"id": hid, "episode_slug": slug, "kind": kind, "created_at": created}


def test_counts_what_was_kept_and_what_was_done_with_it(tmp_path: Path) -> None:
    uid = "u_1"
    _write(
        tmp_path,
        uid,
        "highlights",
        [
            _hl("a", "ep-1", "span", NOW - 2 * DAY),
            _hl("b", "ep-1", "moment", NOW - 3 * DAY),
            _hl("c", "ep-2", "insight", NOW - 30 * DAY),
            _hl("d", "ep-3", "span", NOW - 40 * DAY),
        ],
    )
    _write(
        tmp_path,
        uid,
        "notes",
        [{"id": "n1", "target": "highlight", "target_id": "a", "text": "x"}],
    )
    # The ladder: two captures answered (5 reviews between them), one muted.
    _write(
        tmp_path,
        uid,
        "resurfacing",
        {
            "a": {"count": 3, "last_surfaced": NOW - DAY},
            "b": {"count": 2, "last_surfaced": NOW - DAY},
            "d": {"retired": True},
        },
    )

    s = app_stats.compute_user_stats(tmp_path, uid, now=NOW)

    assert s["captures"] == 4
    assert s["capture_quotes"] == 2 and s["capture_moments"] == 1 and s["capture_insights"] == 1
    assert s["capture_episodes"] == 3, "distinct episodes captured from"
    assert s["notes"] == 1
    # Only the two inside the 7-day window.
    assert s["captures_last_7_days"] == 2
    # The review loop — the sum nothing had ever added up.
    assert s["reviews_total"] == 5, "total reviews answered across every capture"
    assert s["captures_reviewed"] == 2, "distinct captures answered at least once"
    assert s["captures_muted"] == 1


def test_a_muted_capture_is_not_counted_as_reviewed(tmp_path: Path) -> None:
    """Muting is not answering. They are different decisions and must not merge into one number."""
    uid = "u_2"
    _write(tmp_path, uid, "highlights", [_hl("a", "ep-1", "span", NOW - DAY)])
    _write(tmp_path, uid, "resurfacing", {"a": {"retired": True}})
    s = app_stats.compute_user_stats(tmp_path, uid, now=NOW)
    assert s["captures_muted"] == 1
    assert s["captures_reviewed"] == 0
    assert s["reviews_total"] == 0


def test_degrades_to_zeroes_rather_than_raising(tmp_path: Path) -> None:
    """Every file here is hand-editable and may be absent, corrupt, or the wrong shape.

    A stats panel showing zeroes is a bad day; a stats panel 500ing takes the whole Profile with it.
    """
    uid = "u_3"
    _write(
        tmp_path,
        uid,
        "highlights",
        [
            _hl("ok", "ep-1", "span", NOW - DAY),
            # A row the STORE itself rejects: `get_highlights` drops a non-numeric `created_at`, so
            # it never reaches this counter. That is the right answer rather than a special case —
            # stats then agree with the Saved list, which is the same store call.
            {"id": "bad", "episode_slug": "ep", "created_at": "soon"},
        ],
    )
    _write(
        tmp_path, uid, "resurfacing", {"a": "corrupt", "b": {"count": "many"}, "c": {"count": -4}}
    )
    s = app_stats.compute_user_stats(tmp_path, uid, now=NOW)
    assert s["captures"] == 1, "a capture the store rejects must not be counted here either"
    assert s["reviews_total"] == 0, "a non-numeric or negative count must not leak into the sum"
    assert s["captures_reviewed"] == 0
    assert s["captures_muted"] == 0


def test_no_captures_reports_zeroes_not_absence(tmp_path: Path) -> None:
    s = app_stats.compute_user_stats(tmp_path, "nobody", now=NOW)
    for key in ("captures", "notes", "reviews_total", "captures_reviewed", "captures_muted"):
        assert s[key] == 0, f"{key} was missing rather than zero"


def test_the_review_numbers_come_from_the_ladder_the_app_actually_writes(tmp_path: Path) -> None:
    """Guards the seam: `mark_surfaced` is what produces `count`, and this reads the same file.

    Written through the real store rather than a hand-made fixture, so a change to the ladder's
    on-disk shape breaks this instead of silently zeroing the panel.
    """
    uid = "u_4"
    _write(tmp_path, uid, "highlights", [_hl("a", "ep-1", "span", NOW - DAY)])
    app_user_state.mark_surfaced(tmp_path, uid, "a", int(time.time()))
    app_user_state.mark_surfaced(tmp_path, uid, "a", int(time.time()))
    s = app_stats.compute_user_stats(tmp_path, uid, now=NOW)
    assert s["reviews_total"] == 2
    assert s["captures_reviewed"] == 1
