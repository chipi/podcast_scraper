"""An UNPLACED speaker (named by a source, matched to no voice) passes the same publish gate as a
placed one. Measured 2026-10-02: "The China-Global South Project" was published as an unplaced host
on 10 episodes, read from the feed description; the placed path would have refused it.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import pytest

from podcast_scraper.workflow.metadata_generation import _unplaced_speakers

pytestmark = pytest.mark.unit


def _unplaced(known_hosts, guests=()):
    return _unplaced_speakers(
        [],
        diagnostics={"tried": {"known_hosts": list(known_hosts)}},
        detected_hosts=None,
        detected_guests=list(guests),
        feed_title="River Trade Weekly",
    )


def test_an_organisation_stated_as_host_is_not_published() -> None:
    assert _unplaced(["The Harbour Research Project"]) == []


def test_a_role_word_stated_as_a_guest_is_not_published() -> None:
    assert _unplaced([], guests=["Host"]) == []


def test_a_real_stated_host_is_still_published_unplaced() -> None:
    out = _unplaced(["Tobias Wren"])
    assert [(s.name, s.role, s.placed) for s in out] == [("Tobias Wren", "host", False)]


def test_a_stated_name_with_the_show_in_front_is_published_as_the_person() -> None:
    out = _unplaced(["River Trade Weekly's Tobias Wren"])
    assert [s.name for s in out] == ["Tobias Wren"]
