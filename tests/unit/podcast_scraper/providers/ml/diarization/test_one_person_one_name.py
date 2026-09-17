"""One person, one name, one role — across every voice diarization split them over (#2075).

The naming paths name each voice on its own, so a person over-split by diarization came out
under two spellings and two roles (validation run: `Elad Gil` host / `Elad` guest on No Priors,
`Misha Glenny` host / `Misha Glennie` guest on In Our Time). Every surface is written from the
roster's names, so a split person became two people in the transcript, the quotes, the record and
the graph.

Synthetic fixtures only; the name shapes mirror the measured cases.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _one_name_per_person,
    _same_person_on_one_episode,
    _tidy_published_name,
    resolve_speaker_roster,
    SpeakerRole,
)

pytestmark = pytest.mark.unit


def _r(name: str, role: str, source: str = "llm") -> SpeakerRole:
    return SpeakerRole(name=name, role=role, named=True, source=source)


class TestSamePersonOnOneEpisode:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Elad", "Elad Gil"),  # mononym = the other's given name
            ("Misha Glenny", "Misha Glennie"),  # ASR surname spelling
            ("Michael Barbaro", "Michael Babaro"),
            ("Bernard Leong", "Bernard Leung"),
            ("Dr. Adam Rodman", "Adam Rodman"),  # the cross-source rule still applies
            ("amanda aronchik", "Amanda Aronchik"),
        ],
    )
    def test_the_measured_splits_are_one_person(self, a: str, b: str) -> None:
        assert _same_person_on_one_episode(a, b)
        assert _same_person_on_one_episode(b, a)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Robert Pape", "Karen Pape"),  # shared surname, different given name
            ("Karen Pape", "Karen Page"),  # short surnames one letter apart: two families
            ("Tracy Alloway", "Joe Weisenthal"),
            ("Kevin Roose", "Casey Newton"),
            ("Anna Smith", "Anna Jones"),  # shared given name, unrelated surname
            ("Elad", "Sarah"),
        ],
    )
    def test_different_people_stay_apart(self, a: str, b: str) -> None:
        assert not _same_person_on_one_episode(a, b)


class TestTidyPublishedName:
    @pytest.mark.parametrize(
        ("raw", "tidy"),
        [
            ("Amanda  Aronchik", "Amanda Aronchik"),
            ("Aaron Levie)", "Aaron Levie"),
            ("(Jen Kha", "Jen Kha"),
            ("Martin Luther King Jr.", "Martin Luther King Jr."),  # a real trailing dot is kept
            ("Conan O'Brien", "Conan O'Brien"),
        ],
    )
    def test_only_whitespace_and_wrapping_punctuation_change(self, raw: str, tidy: str) -> None:
        assert _tidy_published_name(raw) == tidy


class TestOneNamePerPerson:
    def test_a_stated_spelling_wins_over_the_asr_one(self) -> None:
        out = _one_name_per_person(
            {"S0": _r("Misha Glennie", "guest"), "S1": _r("Misha Glenny", "guest")},
            {"S0": 900.0, "S1": 20.0},
            ["Misha Glenny"],
            [],
        )
        assert {r.name for r in out.values()} == {"Misha Glenny"}

    def test_a_spelling_the_episode_description_writes_wins(self) -> None:
        """Hard Fork: detection returned no names, the description says "the historian Jill Lepore
        joins", and the longest-talking voice carried the ASR's `Jill Laporte`."""
        out = _one_name_per_person(
            {
                "S2": _r("Jill Laporte", "guest"),
                "S1": _r("Jill Lepore", "guest"),
                "S4": _r("Jill Lapour", "guest"),
            },
            {"S2": 1631.0, "S1": 1169.0, "S4": 31.0},
            [],
            ["Casey Newton", "Kevin Roose"],
            episode_text="Then, the historian Jill Lepore joins to discuss her new book.",
        )
        assert {r.name for r in out.values()} == {"Jill Lepore"}

    def test_a_name_inside_a_longer_word_is_not_a_statement(self) -> None:
        out = _one_name_per_person(
            {"S1": _r("Jill Lepore", "guest"), "S2": _r("Jill Laporte", "guest")},
            {"S1": 10.0, "S2": 900.0},
            [],
            [],
            episode_text="Jill Leporeschi joins us.",
        )
        assert {r.name for r in out.values()} == {"Jill Laporte"}, "fell back to the longer voice"

    def test_without_a_stated_spelling_the_fullest_name_wins(self) -> None:
        out = _one_name_per_person(
            {"S0": _r("Elad", "guest"), "S1": _r("Elad Gil", "guest")},
            {"S0": 900.0, "S1": 20.0},
            [],
            [],
        )
        assert out["S0"].name == out["S1"].name == "Elad Gil"

    def test_conflicting_roles_resolve_to_host_only_when_the_feed_states_it(self) -> None:
        split = {"S0": _r("Elad Gil", "host"), "S1": _r("Elad", "guest")}
        stated_host = _one_name_per_person(split, {}, [], ["Elad Gil", "Sarah Guo"])
        assert {r.role for r in stated_host.values()} == {"host"}

        not_stated = _one_name_per_person(split, {}, [], ["Sarah Guo"])
        assert {r.role for r in not_stated.values()} == {
            "guest"
        }, "host is a claim the FEED must make; a longer voice or a host-sounding turn is not one"

    def test_the_longer_voice_does_not_decide_the_role(self) -> None:
        """Measured: the longer-voice rule made Olaf Storbeck a host of Unhedged."""
        out = _one_name_per_person(
            {"S0": _r("Olaf Storbeck", "host"), "S1": _r("Olaf Storbeck", "guest")},
            {"S0": 1200.0, "S1": 60.0},
            ["Olaf Storbeck"],
            ["Robert Armstrong"],
        )
        assert {r.role for r in out.values()} == {"guest"}

    def test_an_agreeing_role_is_kept_even_when_unstated(self) -> None:
        out = _one_name_per_person(
            {"S0": _r("Amanda Aronchik", "host"), "S1": _r("Amanda  Aronchik", "host")}, {}, [], []
        )
        assert [(r.name, r.role) for r in out.values()] == [("Amanda Aronchik", "host")] * 2

    def test_different_people_are_untouched(self) -> None:
        roster = {"S0": _r("Tracy Alloway", "host"), "S1": _r("Joe Weisenthal", "host")}
        assert _one_name_per_person(roster, {}, [], ["Tracy Alloway"]) == roster

    def test_unnamed_voices_are_never_renamed(self) -> None:
        raw = SpeakerRole(name="SPEAKER_02", role="unknown", named=False, source="raw")
        out = _one_name_per_person({"S0": _r("Elad Gil", "host"), "SPEAKER_02": raw}, {}, [], [])
        assert out["SPEAKER_02"] == raw

    def test_no_name_is_invented(self) -> None:
        """The kept spelling is always one a voice already carried."""
        roster = {"S0": _r("Elad", "guest"), "S1": _r("E. Gil", "guest")}
        out = _one_name_per_person(roster, {}, ["Elad Gil"], [])
        assert {r.name for r in out.values()} <= {"Elad", "E. Gil"}


def _diarization(voices: dict[str, float]) -> DiarizationResult:
    segs, t = [], 30.0
    for v, dur in voices.items():
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=v))
        t += dur
    return DiarizationResult(segments=segs, num_speakers=len(voices))


class TestTheResolvedRosterCarriesOneName:
    """Through ``resolve_speaker_roster``: the unification runs on every naming path's output.

    Both scenarios were checked to FAIL with the unification removed — the roster's other paths
    (stated-name recovery, used-name dedupe) do not catch them.
    """

    _VOICES = {"SPEAKER_00": 600.0, "SPEAKER_01": 900.0, "SPEAKER_02": 300.0}

    def test_two_spellings_of_one_guest_publish_as_one_name(self) -> None:
        roster = resolve_speaker_roster(
            _diarization(self._VOICES),
            None,
            known_hosts=["Sarah Guo"],
            llm_voice_names={
                "SPEAKER_00": "Sarah Guo",
                "SPEAKER_01": "Elad",
                "SPEAKER_02": "Elad Gil",
            },
            llm_voice_roles={"SPEAKER_00": "host", "SPEAKER_01": "guest", "SPEAKER_02": "host"},
        )
        by = {v: (r.name, r.role) for v, r in roster.by_voice.items()}
        assert by["SPEAKER_01"] == by["SPEAKER_02"] == ("Elad Gil", "guest"), by
        assert by["SPEAKER_00"] == ("Sarah Guo", "host"), by

    def test_one_person_is_never_both_host_and_guest(self) -> None:
        """Planet Money's shape: the feed states no hosts, one person on two voices, two roles."""
        roster = resolve_speaker_roster(
            _diarization(self._VOICES),
            None,
            llm_voice_names={
                "SPEAKER_00": "Amanda Aronchik",
                "SPEAKER_01": "Amanda  Aronchik",
                "SPEAKER_02": "Jeff Guo",
            },
            llm_voice_roles={"SPEAKER_00": "host", "SPEAKER_01": "guest", "SPEAKER_02": "host"},
        )
        by = {v: (r.name, r.role) for v, r in roster.by_voice.items()}
        assert by["SPEAKER_00"] == by["SPEAKER_01"], by
        assert by["SPEAKER_00"][0] == "Amanda Aronchik", by
