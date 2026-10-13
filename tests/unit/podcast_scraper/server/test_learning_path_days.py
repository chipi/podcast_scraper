"""Days opened early carry forward (operator 2026-10-12, review N6).

A listener may finish several days in one sitting and carry on the next day from where they got
to: opening three days early on day 1 means day 5 is open on day 2 — the head start is kept, not
taken back when the next calendar day arrives.
"""

from __future__ import annotations

from podcast_scraper.server.app_learning_path import open_days, unlocked

_DAY_S = 24 * 60 * 60
_START = 1_800_000_000


def _path(ahead: int) -> dict:
    return {"started_at": _START, "steps": [{} for _ in range(7)], "ahead": ahead, "done": []}


def test_days_opened_early_carry_forward_to_the_next_day() -> None:
    day_1 = _START + 60
    day_2 = _START + _DAY_S + 60
    path = _path(ahead=3)  # day 1 plus three opened early
    assert open_days(path, day_1) == 4
    assert open_days(path, day_2) == 5
    assert unlocked(path, 4, day_2)  # day 5, on day 2
    assert not unlocked(path, 5, day_2)


def test_never_more_days_open_than_the_path_has() -> None:
    assert open_days(_path(ahead=10), _START + 3 * _DAY_S) == 7
