"""Seat logic v4 of the speaker roster: when the vacant host seat may be filled by arithmetic, and
when a stated host's name may be put on a voice.

Each test names one behaviour of `_select_host_voices` step 4, the co-host cues, the vocative
vetoes or the forced-name gate in `_name_host_voices`.

All fixtures are synthetic (never-commit-real-episodes): invented shows, hosts and guests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _addresses_rather_than_is,
    _cohost_self_intro_voices,
    _vocative_count,
    host_copresence_from_diagnostics,
    resolve_speaker_roster,
)

pytestmark = pytest.mark.unit

HOST_A = "Tobias Wren"
HOST_B = "Greta Holm"
KEVIN = "Kevin Fairweather"
CASEY = "Casey Lindqvist"

Turn = Tuple[str, str, float]


def _roster(
    turns: Sequence[Turn],
    known_hosts: Sequence[str] = (HOST_A, HOST_B),
    *,
    detected_guests: Sequence[str] = (),
    metadata_named: Sequence[str] = (),
    host_copresence: Optional[Mapping[int, float]] = None,
):
    segs: List[DiarizationSegment] = []
    chunks: Dict[str, List[str]] = {}
    ordered: List[Tuple[str, str]] = []
    t = 0.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        chunks.setdefault(spk, []).append(" " + text)
        ordered.append((spk, " " + text))
    voice_texts = {v: " ".join(c) for v, c in chunks.items()}
    return resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(voice_texts)),
        " ".join(x for _, x in ordered),
        known_hosts=list(known_hosts),
        detected_guests=list(detected_guests),
        metadata_named=list(metadata_named),
        voice_texts=voice_texts,
        ordered_turns=ordered,
        host_copresence=host_copresence,
    )


def _seated_as_b(roster, voice: str) -> bool:
    r = roster.by_voice[voice]
    return bool(r.role == "host" and r.name == HOST_B)


def _is_not_host_b(roster, voice: str) -> bool:
    r = roster.by_voice[voice]
    return bool(r.name != HOST_B and r.role != "host")


STUDIO = "The ports moved north because the river silted up over a decade."
STUDIO2 = "Most of the merchants followed the trade within a generation."


def _two_voice(opening: str, *, b_text: str = STUDIO, a_tail: str = "") -> List[Turn]:
    """Host A opens with ``opening``; voice B (the spare voice) answers; both span the episode."""
    return [
        ("SPEAKER_00", opening, 60.0),
        ("SPEAKER_01", b_text, 300.0),
        ("SPEAKER_00", "And the guilds, what became of them?" + a_tail, 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 40.0),
    ]


def _three_voice(opening: str, b_text: str, c_text: str) -> List[Turn]:
    """Host A opens; voices B and C are both spare voices that speak inside the intro window and
    span the episode, none dominant."""
    return [
        ("SPEAKER_00", opening, 30.0),
        ("SPEAKER_01", b_text, 20.0),
        ("SPEAKER_02", c_text, 20.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_02", "They mostly stayed out of it.", 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]


HELLO_A = f"Hello and welcome, I'm {HOST_A}."


# ---------------------------------------------------------------------------------------------
# 1. host_copresence_from_diagnostics
# ---------------------------------------------------------------------------------------------


def _diag(*voices: Tuple[str, str, str]) -> Dict[str, Any]:
    return {
        "voices": [
            {"resolved_name": n, "named": True, "role": "host", "source": s} for n, _, s in voices
        ]
    }


def _two_host_diag(source: str = "self_intro") -> Dict[str, Any]:
    return _diag((HOST_A, "", source), (HOST_B, "", source))


def test_copresence_ignores_forced_sources_known_hosts_and_feed() -> None:
    """A name that only the feed forced onto a seat is not evidence the host was in the room."""
    forced = [_two_host_diag("known_hosts"), _two_host_diag("feed")] * 2
    result = host_copresence_from_diagnostics(forced, [HOST_A, HOST_B])
    assert result == {1: 0.0, 2: 0.0, 3: 0.0}


def test_copresence_counts_evidence_sources() -> None:
    """Self-intro, publisher transcript and LLM-resolution names all count as evidence."""
    diags = [
        _two_host_diag("self_intro"),
        _two_host_diag("publisher_transcript"),
        _diag((HOST_A, "", "llm_resolution")),
        _diag((HOST_A, "", "self_intro")),
    ]
    result = host_copresence_from_diagnostics(diags, [HOST_A, HOST_B])
    assert result == {1: 1.0, 2: 0.5, 3: 0.0}


def test_copresence_is_none_with_fewer_than_four_siblings() -> None:
    """Three episodes of history say nothing: no prior is returned."""
    diags = [_two_host_diag()] * 3
    assert host_copresence_from_diagnostics(diags, [HOST_A, HOST_B]) is None


def test_copresence_is_none_without_stated_hosts() -> None:
    """With no stated hosts there is nothing to count."""
    assert host_copresence_from_diagnostics([_two_host_diag()] * 5, []) is None


@pytest.mark.parametrize("n_hosts", [1, 2, 3])
def test_copresence_keys_run_from_one_to_hosts_plus_one(n_hosts: int) -> None:
    """The result has a key for every seat count 1..n+1 (n stated hosts)."""
    hosts = [HOST_A, HOST_B, "Odile Marsh"][:n_hosts]
    diag = _diag(*[(h, "", "self_intro") for h in hosts])
    result = host_copresence_from_diagnostics([diag] * 4, hosts)
    assert result is not None
    assert sorted(result) == list(range(1, n_hosts + 2))
    assert result[n_hosts] == 1.0
    assert result[n_hosts + 1] == 0.0


# ---------------------------------------------------------------------------------------------
# 3. step 4 abstains with more candidates than seats
# ---------------------------------------------------------------------------------------------


def test_step4_abstains_with_two_unnamed_candidates_for_one_seat() -> None:
    """Two spare voices for one empty seat is a guess, not arithmetic: nobody is seated."""
    roster = _roster(_three_voice(HELLO_A, STUDIO, "They mostly followed the money."))
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    assert _is_not_host_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


def test_step4_seats_the_one_spare_voice_for_one_seat() -> None:
    """Control: exactly one spare voice for one empty seat is seated as the second host."""
    roster = _roster(_two_voice(HELLO_A))
    assert _seated_as_b(roster, "SPEAKER_01")


# ---------------------------------------------------------------------------------------------
# 4. a stated guest with no voice blocks the fill
# ---------------------------------------------------------------------------------------------

GUEST = "Priya Nandakumar"


def test_a_stated_guest_with_no_voice_blocks_the_fill() -> None:
    """The metadata states a guest and the one spare voice may be them: the seat stays empty."""
    roster = _roster(_two_voice(HELLO_A), detected_guests=[GUEST])
    assert _is_not_host_b(roster, "SPEAKER_01")


def test_a_stated_guest_absorbed_by_a_thanks_for_having_me_voice_does_not_block() -> None:
    """The stated guest is accounted for by an unnamed voice that says "thanks for having me", so
    the other spare voice may still take the co-host seat."""
    turns = [
        ("SPEAKER_00", HELLO_A, 30.0),
        ("SPEAKER_02", "Thanks so much for having me, it is a pleasure.", 20.0),
        ("SPEAKER_01", STUDIO, 20.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_02", "They mostly stayed out of it.", 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]
    roster = _roster(turns, detected_guests=[GUEST])
    assert _seated_as_b(roster, "SPEAKER_01")
    assert roster.by_voice["SPEAKER_02"].role == "guest"


def test_a_dominant_unnamed_voice_absorbs_the_stated_guest() -> None:
    """A voice with at least half the talk is the interviewee: it accounts for the stated guest, so
    the remaining spare voice may take the co-host seat."""
    turns = [
        ("SPEAKER_00", HELLO_A, 30.0),
        ("SPEAKER_02", "The river silted, so the ports moved north.", 20.0),
        ("SPEAKER_01", "Right, and what about the merchants?", 20.0),
        ("SPEAKER_02", STUDIO2, 900.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 200.0),
        ("SPEAKER_01", "Let's leave it there for today.", 200.0),
        ("SPEAKER_02", "They mostly stayed out of it.", 500.0),
        ("SPEAKER_00", "Thanks for listening.", 30.0),
    ]
    roster = _roster(turns, detected_guests=[GUEST])
    assert _seated_as_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


# ---------------------------------------------------------------------------------------------
# 5. an introduction cue in the seated host's opening
# ---------------------------------------------------------------------------------------------

LOW_PRIOR = {1: 0.9, 2: 0.02}


def test_an_introduction_cue_without_a_name_blocks_the_fill() -> None:
    """The seated host says somebody is joining them but the name never reached the host's own
    turns: an unnamed voice is at least as likely to be that person, so the seat stays empty."""
    opening = f"Hello, I'm {HOST_A}, and joining me in the studio is the head of the column."
    roster = _roster(_two_voice(opening))
    assert _is_not_host_b(roster, "SPEAKER_01")


def test_an_introduction_cue_that_names_the_cohost_does_not_block() -> None:
    """The same cue naming the stated co-host is the co-host being introduced: seat filled."""
    opening = (
        f"Hello, I'm {HOST_A}, and I'm joined, as usual, by the big man in New York, "
        f"mister {HOST_B}."
    )
    roster = _roster(_two_voice(opening))
    assert _seated_as_b(roster, "SPEAKER_01")


def test_a_named_cohost_introduction_overrides_a_low_copresence_prior() -> None:
    """A voice saying the co-host is in the room beats a feed history that says they rarely are."""
    opening = (
        f"Hello, I'm {HOST_A}, and I'm joined, as usual, by the big man in New York, "
        f"mister {HOST_B}."
    )
    roster = _roster(_two_voice(opening), host_copresence=LOW_PRIOR)
    assert _seated_as_b(roster, "SPEAKER_01")


def test_without_a_cohost_introduction_a_low_copresence_prior_leaves_the_seat_empty() -> None:
    """Control for the override above: with the same prior and no cue, nobody is seated."""
    roster = _roster(_two_voice(HELLO_A), host_copresence=LOW_PRIOR)
    assert _is_not_host_b(roster, "SPEAKER_01")


# ---------------------------------------------------------------------------------------------
# 6. the co-host cue is read only in the opening 60% of the text
# ---------------------------------------------------------------------------------------------

NEXT_WEEK = "Next week we're going to be joined by a great friend of the show, Ali Ansari."
FILLER = "The tide tables in the old harbour were kept by hand for a long time. " * 12


def test_an_introduction_cue_in_the_closing_part_of_the_text_is_ignored() -> None:
    """ "We're going to be joined by X next week" at the end of the host's text is a teaser, not
    somebody in the room: it does not block the fill."""
    turns = _two_voice(HELLO_A + " " + FILLER, a_tail=" " + NEXT_WEEK)
    turns[-1] = ("SPEAKER_00", NEXT_WEEK, 40.0)
    roster = _roster(turns)
    assert _seated_as_b(roster, "SPEAKER_01")


def test_the_same_cue_in_the_opening_part_of_the_text_blocks_the_fill() -> None:
    """Control: the identical teaser sentence read in the opening blocks the fill."""
    turns = _two_voice(HELLO_A + " " + NEXT_WEEK + " " + FILLER)
    roster = _roster(turns)
    assert _is_not_host_b(roster, "SPEAKER_01")


# ---------------------------------------------------------------------------------------------
# 7. absence cue
# ---------------------------------------------------------------------------------------------


def test_a_host_said_absent_loses_the_seat() -> None:
    """ "Just me today, I'm afraid. Greta has another project": her seat is not filled by the
    spare voice."""
    opening = f"Hello, I'm {HOST_A}. Just me today, I'm afraid. Greta has another project."
    roster = _roster(_two_voice(opening))
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    assert _is_not_host_b(roster, "SPEAKER_01")


SOLO_BODY = " The ports moved north because the river silted up over a decade."


def test_a_host_said_absent_leaves_the_one_present_host_forced_named() -> None:
    """With the other stated host said absent, one name remains for the one seat: the arithmetic
    is exact and the present host gets the remaining name."""
    opening = "Hello and welcome. Just me today, I'm afraid. Greta has another project."
    roster = _roster([("SPEAKER_00", opening + SOLO_BODY, 600.0)])
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    assert roster.by_voice["SPEAKER_00"].source == "known_hosts"


def test_without_the_absence_cue_a_lone_unnamed_host_is_not_forced_named() -> None:
    """Control: two stated names for one seat is not forced, so the seat keeps no name."""
    roster = _roster([("SPEAKER_00", "Hello and welcome." + SOLO_BODY, 600.0)])
    assert roster.by_voice["SPEAKER_00"].name == "SPEAKER_00"


# ---------------------------------------------------------------------------------------------
# 8. vocative veto, single host
# ---------------------------------------------------------------------------------------------

GUEST_VOICE_INTRO = "Hello, I'm Marguerite Odell, thanks for having me."


def _single_host_turns(c_text: str) -> List[Turn]:
    """One stated host; a stated non-host voice (SPEAKER_00) and a spare voice (SPEAKER_01)."""
    return [
        ("SPEAKER_00", GUEST_VOICE_INTRO, 30.0),
        ("SPEAKER_01", c_text, 30.0),
        ("SPEAKER_00", "The ports moved north because the river silted.", 400.0),
        ("SPEAKER_01", "And what became of the guilds?", 300.0),
        ("SPEAKER_00", STUDIO2, 200.0),
        ("SPEAKER_01", "Let's leave it there.", 40.0),
    ]


def test_single_host_a_voice_that_addresses_the_host_is_not_the_host() -> None:
    """On a one-host feed a voice saying "Tobias, ..." is talking to Tobias, so it is not him."""
    text = "Tobias, you asked about the ports. Tobias, I have notes on it."
    roster = _roster(_single_host_turns(text), known_hosts=[HOST_A])
    assert roster.by_voice["SPEAKER_01"].name != HOST_A


def test_single_host_a_voice_that_does_not_address_the_host_is_seated() -> None:
    """Control: the same spare voice without the vocatives takes the single host seat."""
    roster = _roster(_single_host_turns("Right, the ports."), known_hosts=[HOST_A])
    assert roster.by_voice["SPEAKER_01"].name == HOST_A


def test_vocative_helpers_single_host_one_vocative_is_enough() -> None:
    """With no seated host, a single vocative already means the voice addresses the host."""
    assert _addresses_rather_than_is("So, Tobias, what do you think?", HOST_A, [])
    assert not _addresses_rather_than_is("Tobias Wren is a historian.", HOST_A, [])


def test_vocative_is_not_counted_right_after_a_self_intro() -> None:
    """ "I'm Tobias," is a self-introduction, not an address."""
    assert _vocative_count("Hello. I'm Tobias, nice to be here.", HOST_A) == 0
    assert _vocative_count("Hello. Tobias, nice to be here.", HOST_A) == 1


# ---------------------------------------------------------------------------------------------
# 9. vocative ratio on a two-host feed
# ---------------------------------------------------------------------------------------------


def _bleed_text(first: str, n_first: int, other: str, n_other: int) -> str:
    return " ".join(
        [f"Fair point. {first}, go on." for _ in range(n_first)]
        + [f"Good one. {other}, over to you." for _ in range(n_other)]
    )


def _ratio_turns(spare_text: str, seated_intro: str) -> List[Turn]:
    return [
        ("SPEAKER_00", seated_intro, 30.0),
        ("SPEAKER_01", spare_text, 40.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there.", 30.0),
    ]


def test_two_hosts_a_cluster_addressing_the_absent_host_ten_to_one_is_not_that_host() -> None:
    """Casey is seated. A cluster with 10 "Kev," and 1 "Casey," talks to Kevin far more than to
    Casey, so it is not Kevin."""
    spare = _bleed_text("Kev", 10, "Casey", 1)
    roster = _roster(
        _ratio_turns(spare, f"Hello, I'm {CASEY}."),
        known_hosts=[KEVIN, CASEY],
    )
    assert roster.by_voice["SPEAKER_00"].name == CASEY
    r = roster.by_voice["SPEAKER_01"]
    assert r.name != KEVIN and r.role != "host"


def test_two_hosts_a_bleed_cluster_dominated_by_the_seated_name_is_still_the_host() -> None:
    """Kevin is seated. A cluster with 5 "Casey," and 13 "Kevin," is mostly Casey's bleed from
    the seated host's side: 5 is not more than twice 13, so it may still be Casey."""
    spare = _bleed_text("Casey", 5, "Kevin", 13)
    roster = _roster(
        _ratio_turns(spare, f"Hello, I'm {KEVIN}."),
        known_hosts=[KEVIN, CASEY],
    )
    assert roster.by_voice["SPEAKER_00"].name == KEVIN
    r = roster.by_voice["SPEAKER_01"]
    assert r.name == CASEY and r.role == "host"


def test_vocative_ratio_helper_requires_more_than_twice_the_seated_hosts_count() -> None:
    """Ten to one is a refusal; five to thirteen is not; two to one is not (needs > 2x)."""
    assert _addresses_rather_than_is(_bleed_text("Kev", 10, "Casey", 1), KEVIN, [CASEY])
    assert not _addresses_rather_than_is(_bleed_text("Casey", 5, "Kevin", 13), CASEY, [KEVIN])
    assert not _addresses_rather_than_is(_bleed_text("Kev", 2, "Casey", 1), KEVIN, [CASEY])


# ---------------------------------------------------------------------------------------------
# 10. tie-break between two candidates
# ---------------------------------------------------------------------------------------------


def test_tiebreak_the_candidate_with_two_vocatives_to_the_seated_host_is_picked() -> None:
    """Two spare voices, one seat: the one that addresses the seated host twice is the co-host."""
    addressing = "Tobias, you asked about the ports. Tobias, I have notes on it."
    roster = _roster(_three_voice(HELLO_A, addressing, "They mostly followed the money."))
    assert _seated_as_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


def test_tiebreak_one_vocative_is_not_enough() -> None:
    """A single vocative to the seated host does not pick: the seat stays empty."""
    roster = _roster(
        _three_voice(HELLO_A, "Tobias, you asked about the ports.", "They mostly followed.")
    )
    assert _is_not_host_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


def test_tiebreak_with_no_vocatives_abstains() -> None:
    """With no vocatives on either spare voice nothing picks between them: nobody is seated."""
    roster = _roster(_three_voice(HELLO_A, STUDIO, "They mostly followed the money."))
    assert _is_not_host_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


def test_tiebreak_two_voices_both_addressing_the_host_pick_nobody() -> None:
    """If both candidates qualify, more are picked than there are seats: nobody is seated."""
    addressing = "Tobias, you asked about the ports. Tobias, I have notes on it."
    roster = _roster(_three_voice(HELLO_A, addressing, addressing))
    assert _is_not_host_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


# ---------------------------------------------------------------------------------------------
# 11-13. the co-host formula
# ---------------------------------------------------------------------------------------------


def _formula_turns(b_text: str) -> List[Turn]:
    """Host A self-introduces; SPEAKER_02 is an unnamed spare voice that talks right after A;
    SPEAKER_01 says ``b_text``. Both spare voices span the episode."""
    return [
        ("SPEAKER_00", HELLO_A, 30.0),
        ("SPEAKER_02", "Right, the ports then.", 20.0),
        ("SPEAKER_01", b_text, 20.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_02", "They mostly followed the money.", 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]


def test_cohost_formula_names_the_voice_that_says_it_even_with_a_garbled_surname() -> None:
    """ "And me, Greta Holmgren": the cue plus the stated first name is the evidence; the ASR's
    surname does not matter. That voice is Greta Holm, and the other spare voice is not."""
    roster = _roster(_formula_turns("And me, Greta Holmgren, welcome along."))
    r = roster.by_voice["SPEAKER_01"]
    assert (r.name, r.role) == (HOST_B, "host")
    assert roster.by_voice["SPEAKER_02"].name != HOST_B


def test_cohost_formula_is_name_bearing_not_positional() -> None:
    """The formula speaker is named Greta although another spare voice speaks earlier: the name
    follows the evidence, not the order of the voices."""
    roster = _roster(_formula_turns("With me, Greta Holmgren."))
    assert roster.by_voice["SPEAKER_01"].name == HOST_B
    assert roster.by_voice["SPEAKER_01"].source == "self_intro"


def test_without_the_formula_two_spare_voices_name_nobody_greta() -> None:
    """Control: with no formula there are two candidates for one seat, so nobody is named."""
    roster = _roster(_formula_turns("Right, over to the guilds."))
    assert _is_not_host_b(roster, "SPEAKER_01")
    assert _is_not_host_b(roster, "SPEAKER_02")


MERGED = "Hello, with me Tobias and me Greta, welcome."


def test_cohost_formula_helper_ignores_a_merged_intro_cluster() -> None:
    """A cluster carrying the formula for both stated hosts is the opening exchange merged by
    diarization: it names neither host."""
    vt = {"SPEAKER_00": MERGED, "SPEAKER_01": "And the guilds?"}
    assert _cohost_self_intro_voices(vt, HOST_A, set(), [HOST_A, HOST_B]) == []
    assert _cohost_self_intro_voices(vt, HOST_B, set(), [HOST_A, HOST_B]) == []


def test_cohost_formula_helper_names_a_cluster_with_one_hosts_formula() -> None:
    """Control: a cluster with the formula for one host only does name that host."""
    vt = {"SPEAKER_00": f"Hello, and me {HOST_B}.", "SPEAKER_01": "And the guilds?"}
    assert _cohost_self_intro_voices(vt, HOST_B, set(), [HOST_A, HOST_B]) == ["SPEAKER_00"]


def test_a_merged_intro_cluster_names_nobody_by_the_formula() -> None:
    """End to end: the merged cluster is not given either host's name by the formula."""
    turns = [
        ("SPEAKER_00", MERGED, 30.0),
        ("SPEAKER_01", STUDIO, 20.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]
    roster = _roster(turns)
    assert all(r.source != "self_intro" for r in roster.by_voice.values())


def test_other_host_and_i_names_the_speaker_as_the_remaining_host() -> None:
    """On a two-host feed "Tobias and I are here with Dr Okafor" is Greta speaking."""
    vt = {
        "SPEAKER_00": "Hello everyone.",
        "SPEAKER_01": "Tobias and I are here with Doctor Ines Okafor today.",
    }
    assert _cohost_self_intro_voices(vt, HOST_B, set(), [HOST_A, HOST_B]) == ["SPEAKER_01"]
    assert _cohost_self_intro_voices(vt, HOST_A, set(), [HOST_A, HOST_B]) == []


def test_other_host_and_i_end_to_end_names_the_speaker_as_the_remaining_host() -> None:
    """End to end: the speaker of "Tobias and I are here with <guest>" is seated as Greta Holm."""
    turns = _formula_turns("Tobias and I are here with Doctor Ines Okafor today.")
    roster = _roster(turns)
    r = roster.by_voice["SPEAKER_01"]
    assert (r.name, r.role) == (HOST_B, "host")


def test_other_host_and_i_is_not_read_on_a_three_host_feed() -> None:
    """The "<other> and I" reading needs exactly two stated hosts: on three it names nobody."""
    vt = {"SPEAKER_01": "Tobias and I are here with Doctor Ines Okafor today."}
    hosts = [HOST_A, HOST_B, "Odile Marsh"]
    assert _cohost_self_intro_voices(vt, HOST_B, set(), hosts) == []


# ---------------------------------------------------------------------------------------------
# 14. forced-name gate
# ---------------------------------------------------------------------------------------------


def _step2_second_seat_turns() -> List[Turn]:
    """Host A self-introduces; SPEAKER_01 performs the host role ("welcome to ...") without
    naming itself, so it is seated by step 2 and the one spare name could be forced onto it."""
    return [
        ("SPEAKER_00", HELLO_A, 30.0),
        ("SPEAKER_01", "Welcome back to the harbour hour, glad you could join us.", 20.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]


def test_forced_name_is_withheld_when_the_hosts_do_not_copresent() -> None:
    """A second seat filled by a host speech act is a SEAT; Greta's name goes on it only if the
    feed's history says the two hosts co-present. With copresence[2]=0.02 it keeps no name."""
    roster = _roster(_step2_second_seat_turns(), host_copresence={1: 0.9, 2: 0.02})
    r = roster.by_voice["SPEAKER_01"]
    assert r.role == "host"
    assert r.name != HOST_B and not r.named


def test_forced_name_lands_when_the_hosts_copresent() -> None:
    """Control: with copresence[2]=0.55 the same seat is named Greta Holm."""
    roster = _roster(_step2_second_seat_turns(), host_copresence={1: 0.9, 2: 0.55})
    r = roster.by_voice["SPEAKER_01"]
    assert (r.name, r.role) == (HOST_B, "host")


def test_forced_name_lands_without_any_feed_history() -> None:
    """Control: no history at all (None) leaves the arithmetic ungated, as before."""
    roster = _roster(_step2_second_seat_turns(), host_copresence=None)
    assert roster.by_voice["SPEAKER_01"].name == HOST_B


# ---------------------------------------------------------------------------------------------
# 15. candidates must span half of the episode
# ---------------------------------------------------------------------------------------------


def _span_turns(spare_turns_at: Sequence[Tuple[float, float]]) -> List[Turn]:
    """A 1000s episode: host A talks throughout; SPEAKER_01 (the only spare voice) speaks 30-50s,
    inside the intro window, and again at each ``(start, duration)`` given."""
    events = [(0.0, 30.0, "SPEAKER_00", HELLO_A), (30.0, 20.0, "SPEAKER_01", "Right, the ports.")]
    events += [(a, d, "SPEAKER_01", STUDIO2) for a, d in spare_turns_at]
    events.sort()
    turns: List[Turn] = []
    t = 0.0
    for start, dur, spk, text in events:
        if start > t:
            turns.append(("SPEAKER_00", "And the guilds, what became of them?", start - t))
        turns.append((spk, text, dur))
        t = start + dur
    turns.append(("SPEAKER_00", "Let's leave it there for today.", 1000.0 - t))
    return turns


def test_a_spare_voice_spanning_thirty_percent_of_the_episode_is_not_seated() -> None:
    """An interview guest present for a middle segment is not a co-host even when it is the only
    spare voice: its first turn to its last covers 30% of the episode."""
    roster = _roster(_span_turns([(250.0, 100.0)]))
    assert _is_not_host_b(roster, "SPEAKER_01")


def test_a_spare_voice_spanning_ninety_percent_of_the_episode_is_seated() -> None:
    """Control: the same voice whose turns cover 90% of the episode is seated."""
    roster = _roster(_span_turns([(850.0, 100.0)]))
    assert _seated_as_b(roster, "SPEAKER_01")


# ---------------------------------------------------------------------------------------------
# Two stated hosts, neither self-introduces, each addresses the other (#2276)
# ---------------------------------------------------------------------------------------------

NO_HISTORY_OF_BOTH = {1: 1.0, 2: 0.0, 3: 0.0}


def _pair_episode(a_says: str, b_says: str) -> List[Turn]:
    return [
        ("SPEAKER_00", "Welcome back to the show. Today, the silk road.", 60.0),
        ("SPEAKER_01", b_says, 300.0),
        ("SPEAKER_00", a_says, 300.0),
        ("SPEAKER_01", STUDIO2, 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 40.0),
    ]


def test_pair_by_address_each_voice_is_the_host_the_other_addresses() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _pair_by_address

    texts = {"A": "So, Greta, where do we start?", "B": "Well, Tobias, with the caravans."}
    assert _pair_by_address(texts, "A", "B", [HOST_A, HOST_B]) == {"A": HOST_A, "B": HOST_B}


def test_pair_by_address_abstains_when_both_address_the_same_host_or_one_addresses_both() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _pair_by_address

    same = {"A": "So, Greta, where?", "B": "Right, Greta. The caravans."}
    assert _pair_by_address(same, "A", "B", [HOST_A, HOST_B]) is None
    bleed = {"A": "So, Greta, where? Yes, Tobias.", "B": "Well, Tobias, the caravans."}
    assert _pair_by_address(bleed, "A", "B", [HOST_A, HOST_B]) is None


def test_two_hosts_addressing_each_other_are_both_seated_and_named() -> None:
    """The feed's history never named both by evidence (co-presence 0), and neither host says his
    own name: the address pair is the evidence for the second seat AND for which host is which."""
    roster = _roster(
        _pair_episode("Greta, what became of the guilds?", "Well, Tobias, the ports moved north."),
        host_copresence=NO_HISTORY_OF_BOTH,
    )
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    assert roster.by_voice["SPEAKER_01"].name == HOST_B
    assert roster.by_voice["SPEAKER_01"].role == "host"


def test_without_the_pair_the_history_still_holds_the_second_seat() -> None:
    """Control: the spare voice addresses nobody, so the co-presence history keeps it unseated and
    no pool name is painted on either voice."""
    roster = _roster(
        _pair_episode("Greta, what became of the guilds?", STUDIO),
        host_copresence=NO_HISTORY_OF_BOTH,
    )
    assert roster.by_voice["SPEAKER_01"].role != "host"
    assert roster.by_voice["SPEAKER_01"].name not in (HOST_A, HOST_B)


# ---------------------------------------------------------------------------------------------
# A voice that talks ABOUT a host by given name is not that host (#2276)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Greta's a great believer in the honour system, honestly.",
        "When Greta and I were touring the north, it rained.",
        "This is the tension that Greta was describing earlier.",
    ],
)
def test_speaks_of_by_first_name_catches_third_person_mentions(text: str) -> None:
    from podcast_scraper.providers.ml.diarization.roster import _speaks_of_by_first_name

    assert _speaks_of_by_first_name(text, HOST_B)


@pytest.mark.parametrize(
    "text, feed_title",
    [
        ("Greta, was that the reason?", None),  # a vocative is not a mention
        ("I'm Greta Holm and this is the show.", None),  # her own introduction
        ("Sign up to Greta's newsletter for the notes.", "Greta's Newsletter"),  # the show's name
        ("The ports moved north over a decade.", None),
    ],
)
def test_speaks_of_by_first_name_leaves_the_rest(text: str, feed_title: Optional[str]) -> None:
    from podcast_scraper.providers.ml.diarization.roster import _speaks_of_by_first_name

    assert not _speaks_of_by_first_name(text, HOST_B, feed_title)


def test_a_forced_pool_name_is_refused_to_a_voice_that_talks_about_that_host() -> None:
    """One spare name, one spare seat -- but the seat says "Greta's a great believer...": it is
    not Greta, so it stays unnamed rather than wear her name."""
    about = "Greta's a great believer in the honour system. The ports moved north."
    roster = _roster(_span_turns([(850.0, 100.0)]))
    assert _seated_as_b(roster, "SPEAKER_01")  # control: the same shape names the seat
    turns = [(v, about if v == "SPEAKER_01" else t, d) for v, t, d in _span_turns([(850.0, 100.0)])]
    assert roster.by_voice["SPEAKER_01"].name == HOST_B
    vetoed = _roster(turns)
    assert vetoed.by_voice["SPEAKER_01"].name != HOST_B


def test_the_shows_own_name_is_not_a_mention_of_its_host() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _speaks_of_by_first_name

    outro = "Conversations with Greta is produced by the Harbour Institute."
    assert not _speaks_of_by_first_name(outro, HOST_B, "Conversations with Greta")
    assert _speaks_of_by_first_name(outro, HOST_B, None)


def test_a_seat_that_keeps_addressing_the_cohost_keeps_its_forced_name() -> None:
    """The co-host's "Greta's website" line bled into Greta's cluster, but the cluster talks TO
    Tobias, twice, and never to Greta: it is Greta, and the veto stands aside (Hard Fork)."""
    bled = "Tobias, will you read this for me? To say nothing of Greta's website. Well, Tobias, go."
    turns = [(v, bled if v == "SPEAKER_01" else t, d) for v, t, d in _span_turns([(850.0, 100.0)])]
    assert _seated_as_b(_roster(turns), "SPEAKER_01")


def test_thanking_a_guest_for_joining_is_not_an_introduction_cue() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _INTRODUCTION_CUE

    assert not _INTRODUCTION_CUE.search(
        "So, Rob, so much to talk about, but thank you for joining us."
    )
    assert _INTRODUCTION_CUE.search("Joining us today is the head of the Lex column.")


def test_a_self_introduced_spelling_takes_back_a_name_that_was_forced_onto_another_voice() -> None:
    """The guest said only "Kashmir"; the forced one-name rule had painted "Kashmir Hill" on a
    tape insert. Recovering the guest's spelling unnames the forced voice (#2276, The Daily)."""
    from podcast_scraper.providers.ml.diarization.roster import _recover_stated_names, SpeakerRole

    by_voice = {
        "A": SpeakerRole(name="Kashmir", role="guest", named=True, source="self_intro"),
        "B": SpeakerRole(
            name="Kashmir Hill", role="guest", named=True, source="forced", forced=True
        ),
    }
    _recover_stated_names(by_voice, ["Kashmir Hill"])
    assert by_voice["A"].name == "Kashmir Hill"
    assert not by_voice["B"].named


def test_one_introduction_read_twice_names_one_voice() -> None:
    """The capitalized pass binds the spoken "Brendan Futi"; the case-blind pass resolves the same
    sentence to the stated "Brendan Foody". The co-host who speaks after the guest must not get it.
    """
    from podcast_scraper.providers.ml.diarization.roster import _voice_named_by_the_introduction

    turns = [
        ("HOST", "Welcome back. Today we're chatting with Brendan Futi, cofounder of a company."),
        ("GUEST", "Thanks. So at a high level we train models that predict performance."),
        ("COHOST", "I think it's very funny when the proofs come out."),
    ]
    out = _voice_named_by_the_introduction(
        turns, {"HOST"}, None, frozenset(), metadata_named=["Brendan Foody"]
    )
    assert "COHOST" not in out
    assert out.get("GUEST") in ("Brendan Futi", "Brendan Foody")


def test_two_voices_that_talk_to_each_other_are_not_unified_into_one_person() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _one_name_per_person, SpeakerRole

    by_voice = {
        "A": SpeakerRole(name="Kevin Roose", role="host", named=True, source="self_intro"),
        "B": SpeakerRole(name="Kevin Rose", role="host", named=True, source="llm_resolution"),
    }
    talk = {"A": 400.0, "B": 350.0}
    out = _one_name_per_person(
        by_voice, talk, [], ["Kevin Roose"], alternations={frozenset(("A", "B")): 30}
    )
    assert out["A"].name == "Kevin Roose" and out["A"].named
    assert not out["B"].named
    # The keeper takes the stated spelling even when its own is the bare given name.
    # (the LLM only matches stated names, so the other voice carries the stated spelling)
    bare = {
        "A": SpeakerRole(name="Kevin", role="host", named=True, source="self_intro"),
        "B": SpeakerRole(name="Kevin Roose", role="host", named=True, source="llm_resolution"),
    }
    kept = _one_name_per_person(
        bare, talk, [], ["Kevin Roose"], alternations={frozenset(("A", "B")): 30}
    )
    assert kept["A"].name == "Kevin Roose" and not kept["B"].named
    # Control: a fragment that never converses is still unified (a diarizer split).
    split = _one_name_per_person(by_voice, {"A": 400.0, "B": 8.0}, [], ["Kevin Roose"])
    assert split["B"].name == "Kevin Roose"


def test_a_thank_you_by_name_is_an_address_without_punctuation() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _thanked_by_name

    assert _thanked_by_name("so thank you very much indeed thank you Greta see you soon", HOST_B)
    assert not _thanked_by_name("Greta thanked everyone at the end", HOST_B)


def test_one_introduction_read_twice_keeps_the_voice_that_then_talks() -> None:
    """Latent Space: the co-host interjects right after the introduction, the guest answers at
    length. Whichever reading bound first, the guest keeps the name."""
    from podcast_scraper.providers.ml.diarization.roster import _voice_named_by_the_introduction

    long_answer = "Yes, so the database work started years ago and it grew from there. " * 8
    turns = [
        ("HOST", "Today we're chatting with Brendan Futi, cofounder of a company."),
        ("COHOST", "Great to have you."),
        ("GUEST", long_answer),
    ]
    out = _voice_named_by_the_introduction(
        turns, {"HOST"}, None, frozenset(), metadata_named=["Brendan Foody"]
    )
    assert "COHOST" not in out
    assert out.get("GUEST") in ("Brendan Futi", "Brendan Foody")
