"""A lone host seat is the stated host it does NOT keep addressing (#2304 Phase C, Hard Fork)."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import _seat_by_cohost_address

pytestmark = pytest.mark.unit

HOSTS = ["Kevin Roose", "Casey Newton"]


def _talk(kevin: int, casey: int) -> str:
    return " ".join(["Well, Kevin, that is the story."] * kevin + ["Right, Casey, go on."] * casey)


def test_a_seat_that_keeps_addressing_kevin_is_casey() -> None:
    assert _seat_by_cohost_address(_talk(20, 5), HOSTS) == "Casey Newton"
    assert _seat_by_cohost_address(_talk(0, 4), HOSTS) == "Kevin Roose"


@pytest.mark.parametrize(
    "kevin, casey",
    [
        (1, 0),  # one address: guests address hosts too
        (2, 0),  # still below the floor
        (5, 4),  # addresses both alike: says nothing
        (6, 3),  # the other is addressed more than a third as often
        (0, 0),
    ],
)
def test_weak_or_mixed_address_names_nobody(kevin: int, casey: int) -> None:
    assert _seat_by_cohost_address(_talk(kevin, casey), HOSTS) is None


def test_only_with_exactly_two_stated_hosts() -> None:
    assert _seat_by_cohost_address(_talk(20, 0), ["Kevin Roose"]) is None
    assert _seat_by_cohost_address(_talk(20, 0), [*HOSTS, "Third Host"]) is None
