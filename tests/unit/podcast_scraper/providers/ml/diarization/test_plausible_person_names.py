"""A group or an institution is not one speaker (#2304 Phase C, prod 2026-10-13).

The episode-description extraction handed the roster "NTT's Japanese owners" and "Africa Bitcoin
Corporation board" as people: they were listed as unplaced guests in the speaker record, and one
entered the guest pool as a spare name the forced one-name-one-voice rule could paint onto a voice.
The cases below are every name the rule touched in the 2,014 distinct names stored on prod, plus
the real names next to them that it must keep.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    plausible_person_names,
    resolve_speaker_roster,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "junk",
    [
        "NTT's Japanese owners",
        "NTT’s Japanese owners",
        "Africa Bitcoin Corporation board",
        "Madlanga Commission",
        "math teacher",
        "The Ord brothers",
        "Biden's Envoy",  # caught by the possessive alone: no group word, nothing lowercase
    ],
)
def test_a_group_or_an_institution_is_dropped(junk: str) -> None:
    assert plausible_person_names([junk]) == []


@pytest.mark.parametrize(
    "real",
    [
        "Jasmine Opperman",
        "swyx",  # a host's lowercase handle: one word, kept
        "Yushan",
        "Ludwig van Beethoven",
        "Osama bin Laden",
        "Bill Banks",  # "Banks" is not the institution word "bank"
        "unknown_guest_1",  # the pipeline's placeholder is a signal, never dropped
        "Host",
    ],
)
def test_a_real_name_and_the_placeholders_are_kept(real: str) -> None:
    assert plausible_person_names([real]) == [real]


def test_order_is_kept_and_blanks_dropped() -> None:
    assert plausible_person_names(["B One", "", "math teacher", "A Two"]) == ["B One", "A Two"]


def _roster(turns: List[Tuple[str, str, float]], detected_guests: List[str]):
    segs: List[DiarizationSegment] = []
    texts: Dict[str, str] = {}
    ordered: List[Tuple[str, str]] = []
    t = 0.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        texts[spk] = (texts.get(spk, "") + " " + text).strip()
        ordered.append((spk, text))
    return resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(texts)),
        " ".join(x for _, x in ordered),
        detected_guests=detected_guests,
        known_hosts=["Alec Hogg"],
        voice_texts=texts,
        ordered_turns=ordered,
        metadata_named=list(detected_guests),
    )


def test_a_junk_phrase_is_never_painted_onto_the_one_unnamed_voice() -> None:
    """One spare "name", one unnamed voice: the forced rule would bind them."""
    roster = _roster(
        [
            ("SPEAKER_00", "Welcome to the show, I'm Alec Hogg. Let's talk markets today.", 300.0),
            ("SPEAKER_01", "The rand weakened again and the bond market followed it down.", 400.0),
            ("SPEAKER_00", "And what happens to the miners after that?", 60.0),
            ("SPEAKER_01", "They benefit from the weaker currency for a while at least.", 400.0),
        ],
        detected_guests=["NTT's Japanese owners"],
    )
    # Unfixed, the forced rule bound it (one_name_one_voice) and the canonicaliser published it as
    # "Japanese owners" — so the check is on what is PUBLISHED, not on the raw phrase.
    assert not roster.by_voice["SPEAKER_01"].named, roster.by_voice["SPEAKER_01"]
    assert all("owners" not in (r.name or "").lower() for r in roster.by_voice.values())
    assert roster.by_voice["SPEAKER_00"].name == "Alec Hogg"
