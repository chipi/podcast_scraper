"""Merge rules for the host/guest lists handed to the graph (#2062).

The diarization roster listened to the episode; ``detected_hosts`` / ``detected_guests`` only read
the show notes, before any audio was processed. Host and guest are SPEAKING roles, so when a roster
exists it is the only source consulted; the hint is a fallback for episodes that have no roster,
not a supplement to one.

This file first pinned the opposite rule — an additive merge that kept hint-only names so a guest
the roster could not place would not be dropped. Production refuted it: on episodes where
diarization named every voice it heard, 19.4% of host nodes and 14.3% of guest nodes belong to
someone who did not speak — a co-host who sat the episode out, or the show's own name as a person.
(Counting every episode gives a bigger number, but a partial roster is silent about its anonymous
voices rather than denying them, so that is an upper bound.) The tests below pin the corrected
rule and the reason it changed.
"""

from __future__ import annotations

import pytest

from podcast_scraper.workflow.metadata_generation import (
    _speaker_lists_for_graph,
    SpeakerInfo,
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


class TestEveryCallerPassesTheFeedTitle:
    """`_speaker_lists_for_graph` can only refuse a show name when it KNOWS the title (advisor).

    `names_the_show(candidate, feed_title)` returns False on an empty title by design — "no title
    means no opinion". So a caller that omits `feed_title` silently disables the guard it is
    calling the function for. The enrich-edges path in `search/cli_handlers.py` did exactly that.

    The review called it Low, reasoning a show name rarely appears as a literal `<Name>:` marker.
    Measured, the gap is wider than that: WITHOUT the title every show name is kept.

        name                      no title    with title
        Africa Tech Summit        KEPT        refused
        Machine Learning Street   KEPT        refused
        Latent.Space              KEPT        refused
        Kevin Roose (real host)   KEPT        KEPT

    Two code paths giving two different answers to "is this a person" is the drift this arc keeps
    paying for, so the caller now passes it.
    """

    @staticmethod
    def _roster(name: str, role: str = "host"):
        from types import SimpleNamespace

        return [SimpleNamespace(name=name, role=role)]

    def test_without_a_title_the_show_name_survives(self) -> None:
        hosts, _g = _speaker_lists_for_graph(self._roster("Africa Tech Summit"), [], [])
        assert "Africa Tech Summit" in hosts, "documents WHY the title is required, not optional"

    def test_with_the_title_the_show_name_is_refused(self) -> None:
        hosts, _g = _speaker_lists_for_graph(
            self._roster("Africa Tech Summit"), [], [], feed_title="Africa Tech Summit Podcast"
        )
        assert "Africa Tech Summit" not in hosts

    def test_a_real_host_is_kept_either_way(self) -> None:
        for kwargs in ({}, {"feed_title": "Hard Fork"}):
            hosts, _g = _speaker_lists_for_graph(self._roster("Kevin Roose"), [], [], **kwargs)
            assert "Kevin Roose" in hosts

    def test_the_enrich_edges_caller_passes_it(self) -> None:
        # Guards the CALL SITE. A unit test of the function cannot catch an omission at a caller,
        # and this omission is invisible: the function returns a plausible answer either way.
        import inspect

        from podcast_scraper.search import cli_handlers

        src = inspect.getsource(cli_handlers)
        idx = src.index("_speaker_lists_for_graph(")
        assert (
            "feed_title" in src[idx : idx + 500]
        ), "enrich-edges must pass feed_title or the show-name guard is inert on that path"
