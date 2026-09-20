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
            ("Misha Glenny", "Misha Glennie"),  # ASR surname spelling
            ("Michael Barbaro", "Michael Babaro"),
            ("Bernard Leong", "Bernard Leung"),
            # MEASURED ON ODD LOTS. The publisher transcript writes `Tracey Alloway`, the feed
            # states `Tracy Alloway`, and the two landed on different voices as host AND guest —
            # one human, two KG Person nodes, contradictory roles. The matcher tolerated a
            # near-spelling in the SURNAME from the start but required the given name to match
            # exactly, so this slipped through on the one axis it did not cover.
            ("Tracy Alloway", "Tracey Alloway"),
            # SHORT SURNAMES ARE RESPELLINGS TOO. These were refused on the theory that names
            # this short are "different families" — true across a corpus, wrong inside ONE
            # episode, where `Sarah Chen` and `Sarah Chan` are the diarizer splitting one guest
            # and the ASR spelling her surname two ways. Length was never the question; the old
            # rule asked it anyway and got the answer wrong.
            ("Sarah Chen", "Sarah Chan"),
            ("Alice Pape", "Alice Page"),
            ("Karen Pape", "Karen Page"),
            # A TITLE IS NOT PART OF A NAME, so it is stripped before anything is compared.
            ("Dr. Sarah Chen", "Sarah Chan"),
            ("Professor Fenwick", "Fenwick"),
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
            ("Robert Pape", "Karen Pape"),  # shared surname, different GIVEN name
            ("Tracy Alloway", "Joe Weisenthal"),
            ("Kevin Roose", "Casey Newton"),
            ("Anna Smith", "Anna Jones"),  # shared given name, unrelated surname
            ("Elad", "Elad Gil"),  # a one-word name is NOT merged (advisor review, #2075)
            ("Alex", "Alex Maasi"),  # Planet Money: a site worker and the host, two people
            # The given-name tolerance must not become a licence to merge strangers. Short tokens
            # are excluded on BOTH sides for the reason that always applied to surnames: names
            # this short are one letter apart by coincidence, not by respelling.
            ("Jon Chen", "Jan Chen"),
            ("Joe Weisenthal", "Jane Weisenthal"),
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
            # A NAME STARTS AND ENDS WITH A LETTER OR A DIGIT. The trailing dot of an
            # abbreviated suffix is punctuation, not part of the name, and leaving it made
            # `... Jr.` a different person id from `... Jr` on any surface that dropped it.
            # Operator decision 2026-09-20; digits are admitted for regnal numbers (`Louis 14`).
            ("Martin Luther King Jr.", "Martin Luther King Jr"),
            ("Peter Attia, MD", "Peter Attia"),
            ("Peter Attia,", "Peter Attia"),
            (",Peter Attia", "Peter Attia"),
            ("Louis 14", "Louis 14"),
            ("(Benedict XVI)", "Benedict XVI"),
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
            {"S0": _r("Misha Glennie", "guest"), "S1": _r("Misha Glennnie", "guest")},
            {"S0": 900.0, "S1": 20.0},
            [],
            [],
        )
        assert len({r.name for r in out.values()}) == 1

    def test_conflicting_roles_resolve_to_host_only_when_the_feed_states_it(self) -> None:
        split = {"S0": _r("Elad Gilman", "host"), "S1": _r("Elad Gilman", "guest")}
        stated_host = _one_name_per_person(split, {}, [], ["Elad Gilman", "Sarah Guo"])
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
        out = _one_name_per_person({"S0": _r("Elad Gilman", "host"), "SPEAKER_02": raw}, {}, [], [])
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
                "SPEAKER_01": "Elad Gilman",
                "SPEAKER_02": "Elad Gilmann",
            },
            llm_voice_roles={"SPEAKER_00": "host", "SPEAKER_01": "guest", "SPEAKER_02": "host"},
        )
        by = {v: (r.name, r.role) for v, r in roster.by_voice.items()}
        assert by["SPEAKER_01"] == by["SPEAKER_02"], by
        assert by["SPEAKER_01"][1] == "guest", by
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


@pytest.mark.unit
class TestWholeNameSimilarityWithinOneEpisode:
    """Two speakers in ONE episode whose full names are nearly identical are one person.

    The scope is what makes this safe. Across a corpus, `Sarah Chen` and `Sarah Chan` could easily
    be two people; as two speakers of a SINGLE episode they are a diarizer split with an ASR
    respelling. The structural rules cannot see every shape that takes — a transposition, a middle
    name, a differing token count — so similarity is OR'd in as an additional signal.

    THE THRESHOLD IS MEASURED, not chosen. Over the pairs this repo has OBSERVED on the corpus the
    populations separate with room to spare: same-person bottoms out at 0.846, different-people
    tops out at 0.667. `_ONE_EPISODE_NAME_SIMILARITY` = 0.91 sits inside that gap AND above every
    adversarial pair anyone has constructed (max 0.900), so it does not depend on those being
    impossible — it refuses them regardless.
    """

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Kevin Roose", "Kevin Roosee"),
            ("Jonathan Weisenthal", "Jonathan Weisental"),
            ("Mark Galeotti", "Mark Galeoti"),
            ("Tracey Alloway", "Tracy Alloway"),
        ],
    )
    def test_a_near_identical_full_name_is_one_person(self, a: str, b: str) -> None:
        assert _same_person_on_one_episode(a, b)
        assert _same_person_on_one_episode(b, a)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Joe Weisenthal", "Jane Weisenthal"),
            ("Jon Chen", "Jan Chen"),
            ("Anna Smith", "Anna Jones"),
        ],
    )
    def test_a_different_name_is_not_a_respelling(self, a: str, b: str) -> None:
        """The varying token has to be a RESPELLING, not another name.

        `joe`/`jane` (0.571), `jon`/`jan` (0.667) and `smith`/`jones` (0.200) are all below
        `_TOKEN_RESPELLING_SIMILARITY`, and all three are pairs of real, distinct names — which is
        the distinction that matters, not how many characters they happen to have.
        """
        assert not _same_person_on_one_episode(a, b)

    def test_the_structural_rules_still_carry_what_similarity_cannot_see(self) -> None:
        """A title prefix costs four characters of pure difference (0.846) and a surname
        respelling can fall under the bar (0.880) — both below 0.91, both still merged."""
        assert _same_person_on_one_episode("Dr. Adam Rodman", "Adam Rodman")
        assert _same_person_on_one_episode("Misha Glenny", "Misha Glennie")


@pytest.mark.unit
class TestABareFirstNameJoinsItsFullName:
    """`Elad` and `Elad Gil` on two voices of one episode are one person, split by the diarizer.

    Leaving them apart mints two KG Person nodes for him. But name shape alone cannot prove it:
    Planet Money has host `Alex Maasi` AND a construction-site worker who says "I'm Alex" —
    structurally identical, two real people, and merging them made the host a guest (advisor
    review, #2075). So the pure name predicate still refuses this shape, and the merge happens
    here, where TALK TIME is available to tell the two apart.

    A person the diarizer split keeps a substantial share of the conversation on both voices; a
    passer-by interviewed for one answer does not.
    """

    def _roles(self, names, talk, known_hosts=()):
        by = {
            v: SpeakerRole(name=n, role=r, named=True, source="test") for v, (n, r) in names.items()
        }
        out = _one_name_per_person(by, talk, stated=[], known_hosts=list(known_hosts))
        return {v: (out[v].name, out[v].role) for v in out}

    def test_a_substantial_bare_name_joins_the_full_name(self) -> None:
        got = self._roles(
            {"SPEAKER_00": ("Elad Gil", "host"), "SPEAKER_01": ("Elad", "guest")},
            {"SPEAKER_00": 900.0, "SPEAKER_01": 420.0},
            known_hosts=["Elad Gil"],
        )
        assert got["SPEAKER_00"] == got["SPEAKER_01"] == ("Elad Gil", "host"), got

    def test_a_cameo_walk_on_is_not_the_host(self) -> None:
        """THE MEASURED COUNTER-EXAMPLE. A site worker who says "I'm Alex" for twelve seconds is
        not Alex Maasi, and merging them made the host a guest."""
        got = self._roles(
            {"SPEAKER_00": ("Alex Maasi", "host"), "SPEAKER_01": ("Alex", "guest")},
            {"SPEAKER_00": 900.0, "SPEAKER_01": 12.0},
        )
        assert got["SPEAKER_00"] == ("Alex Maasi", "host"), got
        assert got["SPEAKER_01"][0] == "Alex", got

    def test_an_ambiguous_given_name_is_never_guessed(self) -> None:
        """Two full names share the given name, so nothing says which the bare voice belongs to.
        Picking one would be exactly the #876 failure."""
        got = self._roles(
            {
                "SPEAKER_00": ("Alex Maasi", "host"),
                "SPEAKER_01": ("Alex Blumberg", "guest"),
                "SPEAKER_02": ("Alex", "guest"),
            },
            {"SPEAKER_00": 900.0, "SPEAKER_01": 600.0, "SPEAKER_02": 300.0},
        )
        assert got["SPEAKER_02"][0] == "Alex", got
        assert got["SPEAKER_00"][0] == "Alex Maasi"
        assert got["SPEAKER_01"][0] == "Alex Blumberg"


@pytest.mark.unit
class TestTwoVouchedSpellingsAreTwoPeople:
    """Near-identical names in one episode are usually one person — but not when the episode
    introduces BOTH of them.

    Sharing a surname makes two speakers MORE likely to be related, not less, so the dangerous
    class is exactly the one that co-occurs: spouses, siblings, a parent and child. And where the
    family name comes first, two unrelated guests collide outright.

    Measured against the matcher without this gate:

        Dan Smith / Dana Smith          Maria Silva / Mario Silva
        Sergei Ivanov / Sergei Ivanova  Li Qiang / Li Keqiang
        Kim Jong-un / Kim Jong-il       Lee Jae-myung / Lee Jae-yong

    all merged — each publishing one real person under another real person's name (#876). Sinica,
    Korea Deconstructed and The China-Global South Podcast are live feeds, so this is not
    hypothetical.

    The discriminator is the EPISODE, not the strings: a diarizer split leaves at most one
    spelling vouched for, because the other is the ASR's transcription of it.
    """

    def _names(self, pairs, stated, known_hosts=()):
        by = {
            f"S{i}": SpeakerRole(name=n, role="guest", named=True, source="t")
            for i, n in enumerate(pairs)
        }
        talk = {v: 300.0 for v in by}
        out = _one_name_per_person(by, talk, stated=list(stated), known_hosts=list(known_hosts))
        return {out[v].name for v in out}

    @pytest.mark.parametrize(
        "pair",
        [
            ("Dan Smith", "Dana Smith"),
            ("Maria Silva", "Mario Silva"),
            ("Andrea Rossi", "Andrew Rossi"),
            ("Sergei Ivanov", "Sergei Ivanova"),
            ("Li Qiang", "Li Keqiang"),
            ("Kim Jong-un", "Kim Jong-il"),
            ("Lee Jae-myung", "Lee Jae-yong"),
        ],
    )
    def test_both_introduced_means_two_people(self, pair) -> None:
        assert self._names(pair, stated=list(pair)) == set(pair), (
            f"{pair[0]!r} and {pair[1]!r} are both introduced by this episode, so merging them "
            "publishes one real person under another's name (#876)"
        )

    def test_one_vouched_spelling_is_a_diarizer_split(self) -> None:
        """THE MEASURED CASE. Odd Lots states `Tracy Alloway`; `Tracey` is the publisher
        transcript's spelling of the same host, never introduced as anybody."""
        assert self._names(
            ("Tracy Alloway", "Tracey Alloway"),
            stated=[],
            known_hosts=["Tracy Alloway"],
        ) == {"Tracy Alloway"}

    def test_an_unvouched_pair_is_still_merged(self) -> None:
        """Nobody is claiming these are two people, and a repeated letter is plainly one voice
        transcribed twice. Refusing here would undo the de-duplication this function exists for."""
        assert len(self._names(("Misha Glennie", "Misha Glennnie"), stated=[])) == 1


@pytest.mark.unit
class TestCredentialsAreNotPartOfAName:
    """`Peter Attia, MD` is the same man as `Peter Attia`.

    MEASURED on 400 sampled prod episodes: The Peter Attia Drive carries the pair on 4 of them —
    the credentialled form as the `host` Person and the plain form as a `mentioned` Person, two
    nodes for one human. The feed states the credentialled spelling, so it won the canonical name,
    and "fullest name wins" counted `MD` as a name token.
    """

    def test_a_credential_is_dropped_from_the_published_name(self) -> None:
        by = {
            "S0": SpeakerRole(name="Peter Attia", role="guest", named=True, source="t"),
            "S1": SpeakerRole(name="Peter Attia, MD", role="host", named=True, source="t"),
        }
        out = _one_name_per_person(
            by, {"S0": 300.0, "S1": 900.0}, stated=["Peter Attia, MD"], known_hosts=[]
        )
        assert {r.name for r in out.values()} == {
            "Peter Attia"
        }, "the credential says what he is qualified as, not who he is"

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Peter Attia", "Peter Attia, MD"),
            ("Jane Doe", "Jane Doe, PhD"),
            ("Linus Abrams", "Dr. Linus Abrams"),
        ],
    )
    def test_prefix_and_suffix_forms_are_one_person(self, a: str, b: str) -> None:
        assert _same_person_on_one_episode(a, b)
        assert _same_person_on_one_episode(b, a)

    def test_a_generational_suffix_is_kept(self) -> None:
        """`Jr.` exists to tell a father from a son; dropping it would merge two people. This is
        the one shape where the suffix IS the distinction."""
        from podcast_scraper.providers.ml.diarization.roster import _tidy_published_name

        # The generational token survives; only its trailing dot is punctuation.
        assert _tidy_published_name("Sam Lee Jr.") == "Sam Lee Jr"
        assert _same_person_on_one_episode("Sam Lee", "Sam Lee Jr") is False
        assert _tidy_published_name("Peter Attia, MD") == "Peter Attia"
