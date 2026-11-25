"""Re-checking a written article for a correction (CRFC-006 Part B, B.2; W2's public half): the
fixture cases B.5 names (same page, whitespace change, edit, 404, paywall appears) on the
normaliser, the compare and the schedule."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from podcast_scraper.news import recheck

pytestmark = pytest.mark.unit

URL = "https://news.test/world/2026/nov/24/a"
_P1 = "The council voted on Tuesday to close the bridge for repairs from next month. " * 3
_P2 = "Engineers said the steel showed fatigue cracks that could not wait for spring. " * 3
_P3 = "Residents will be offered a free shuttle across the river while the work runs. " * 3
SERVED = "\n\n".join([_P1, _P2, _P3])
T0 = datetime(2026, 11, 24, 6, 0, tzinfo=timezone.utc)


def test_the_same_page_is_the_same() -> None:
    got = recheck.compare(SERVED, SERVED, url=URL)
    assert got.outcome == recheck.SAME and got.served_hash == got.candidate_hash


def test_a_whitespace_or_quote_change_is_the_same() -> None:
    rewrapped = SERVED.replace(". ", ".  \n").replace("the bridge", "the ‘bridge’", 1)
    rewrapped = rewrapped.replace("the ‘bridge’", "the 'bridge'")
    assert (
        recheck.compare(SERVED.replace("the bridge", "the 'bridge'", 1), rewrapped, url=URL).outcome
        == recheck.SAME
    )
    assert recheck.normalised_hash(SERVED + "\n\n\n") == recheck.normalised_hash(SERVED)


def test_page_furniture_is_not_a_change() -> None:
    with_offer = SERVED + "\n\nSubscribe to our newsletter for daily updates."
    assert recheck.compare(SERVED, with_offer, url=URL).outcome == recheck.SAME


def test_an_edit_is_a_change() -> None:
    corrected = SERVED.replace("next month", "next week")
    got = recheck.compare(SERVED, corrected, url=URL)
    assert got.outcome == recheck.CHANGED and got.served_hash != got.candidate_hash


def test_a_paywall_in_place_of_the_article_is_held() -> None:
    paywalled = _P1[:160] + "\n\nSubscribe to read the full story. Already a subscriber? Log in."
    got = recheck.compare(SERVED, paywalled, url=URL)
    assert got.outcome == recheck.HELD_SHORT and got.length_share < recheck.DEFAULT_HOLD_SHARE


def test_a_full_article_turned_teaser_is_held_even_above_the_share() -> None:
    served = "\n\n".join([_P1, _P2, _P3[:150]])  # just over the 600-character teaser threshold
    teaser = "\n\n".join([_P1, _P2[:300]])  # under it, but above the hold share
    assert len(served) >= 600 > len(teaser)
    got = recheck.compare(served, teaser, url=URL)
    assert got.length_share >= recheck.DEFAULT_HOLD_SHARE
    assert got.outcome == recheck.HELD_NOT_FULL


def test_the_schedule_runs_on_its_days_and_ends() -> None:
    state = recheck.start(T0)
    assert state.next_due == T0 + timedelta(days=1)
    dues: list[datetime | None] = []
    for _ in recheck.DEFAULT_DAYS:
        dues.append(state.next_due)
        state, action = recheck.advance(state, recheck.SAME)
        assert action == recheck.RECORD
    assert dues == [T0 + timedelta(days=d) for d in recheck.DEFAULT_DAYS]
    assert state.done and state.next_due is None and state.checks_done == 4
    with pytest.raises(ValueError):
        recheck.advance(state, recheck.SAME)


def test_a_window_of_0_never_checks() -> None:
    assert recheck.start(T0, days=()).done


def test_an_edit_writes_a_version_and_a_hold_holds() -> None:
    state, action = recheck.advance(recheck.start(T0), recheck.CHANGED)
    assert action == recheck.WRITE_VERSION and not state.done
    _, action = recheck.advance(state, recheck.HELD_SHORT)
    assert action == recheck.HOLD


def test_gone_twice_in_a_row_withdraws_and_once_does_not() -> None:
    state, action = recheck.advance(recheck.start(T0), recheck.GONE)
    assert action == recheck.RECORD and state.gone_in_a_row == 1
    back, _ = recheck.advance(state, recheck.SAME)
    assert back.gone_in_a_row == 0  # answered again: the count restarts
    state, action = recheck.advance(state, recheck.GONE)
    assert action == recheck.WITHDRAW and state.done


def test_a_failed_check_is_retried_on_the_next_day() -> None:
    state, action = recheck.advance(recheck.start(T0), recheck.FAILED)
    assert action == recheck.RECORD and state.next_due == T0 + timedelta(days=2)
