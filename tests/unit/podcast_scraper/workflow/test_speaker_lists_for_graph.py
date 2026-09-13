"""Merge rules for the host/guest lists handed to the graph (#2062).

The diarization roster listened to the episode; ``detected_hosts`` / ``detected_guests`` only read
the show notes, before any audio was processed. Host and guest are SPEAKING roles, so when a roster
exists it is the only source consulted; the hint is a fallback for episodes that have no roster,
not a supplement to one.

This file first pinned the opposite rule — an additive merge that kept hint-only names so a guest
the roster could not place would not be dropped. Production refuted it: 39.5% of host nodes and 40%
of guest nodes belong to someone who never spoke in that episode (a co-host who sat the episode out,
the show's own name as a person, or an ASR variant of someone who did speak, which puts one human in
the graph twice). The tests below now pin the corrected rule and the reason it changed.
"""

from __future__ import annotations

import pytest


from podcast_scraper.workflow.metadata_generation import (
    SpeakerInfo,
    _speaker_lists_for_graph,
)

pytestmark = pytest.mark.unit


def _sp(name: str, role: str, sid: str | None = None) -> SpeakerInfo:
    return SpeakerInfo(id=sid or role, name=name, role=role)


class TestTheRosterWins:
    def test_roster_guest_survives_an_empty_hint(self) -> None:
        # The prod shape: a network feed whose pre-diarization hint knows nobody.
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Patrick O'Shaughnessy", "host"), _sp("Brian Chesky", "guest")], [], []
        )
        assert hosts == ["Patrick O'Shaughnessy"]
        assert guests == ["Brian Chesky"]

    def test_roster_guest_beats_a_hint_that_calls_them_a_host(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Brian Chesky", "guest")], ["Brian Chesky"], []
        )
        assert guests == ["Brian Chesky"]
        assert hosts == []

    def test_roster_host_beats_a_hint_that_calls_them_a_guest(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Patrick O'Shaughnessy", "host")], [], ["Patrick O'Shaughnessy"]
        )
        assert hosts == ["Patrick O'Shaughnessy"]
        assert guests == []


class TestTheHintIsOnlyUsedWhenThereIsNoRoster:
    def test_a_name_the_roster_never_heard_is_not_published_as_a_guest(self) -> None:
        # THE 40% CASE. The hint names someone the roster never heard. They did not speak, so they
        # are not a guest of this episode. Extraction still records them as `mentioned` if the
        # transcript mentions them — which is the truthful role.
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Patrick O'Shaughnessy", "host")], [], ["Someone On Tape"]
        )
        assert guests == []
        assert hosts == ["Patrick O'Shaughnessy"]

    def test_a_co_host_who_sat_the_episode_out_is_not_a_host_of_it(self) -> None:
        # "Sarah Guo" on an episode where Elad Gil interviews Glenn Fogel — a real prod case.
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Elad Gil", "host"), _sp("Glenn Fogel", "guest")], ["Sarah Guo"], []
        )
        assert hosts == ["Elad Gil"]
        assert guests == ["Glenn Fogel"]
        assert "Sarah Guo" not in hosts + guests

    def test_no_roster_at_all_falls_back_entirely_to_the_hint(self) -> None:
        hosts, guests = _speaker_lists_for_graph(None, ["A Host"], ["A Guest"])
        assert (hosts, guests) == (["A Host"], ["A Guest"])

    def test_everything_empty_yields_empty(self) -> None:
        assert _speaker_lists_for_graph(None, None, None) == ([], [])


class TestHygiene:
    def test_duplicates_collapse_case_insensitively_first_spelling_wins(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [_sp("Brian Chesky", "guest")], [], ["brian chesky", "BRIAN CHESKY"]
        )
        assert guests == ["Brian Chesky"]

    def test_blank_names_are_dropped(self) -> None:
        hosts, guests = _speaker_lists_for_graph([_sp("  ", "guest")], ["  "], [""])
        assert (hosts, guests) == ([], [])

    def test_order_is_preserved(self) -> None:
        hosts, guests = _speaker_lists_for_graph(
            [
                _sp("G1", "guest", "guest_1"),
                _sp("G2", "guest", "guest_2"),
                _sp("G3", "guest", "guest_3"),
            ],
            [],
            [],
        )
        assert guests == ["G1", "G2", "G3"]

    def test_hint_order_is_preserved_when_there_is_no_roster(self) -> None:
        hosts, guests = _speaker_lists_for_graph(None, ["H1", "H2"], ["G1", "G2"])
        assert (hosts, guests) == (["H1", "H2"], ["G1", "G2"])

    def test_a_roster_voice_with_no_usable_role_is_treated_as_host(self) -> None:
        # Matches _build_speakers_from_diarized_segments' own fallback, so the two agree.
        hosts, guests = _speaker_lists_for_graph([_sp("Nobody Knows", "")], [], [])
        assert hosts == ["Nobody Knows"]
