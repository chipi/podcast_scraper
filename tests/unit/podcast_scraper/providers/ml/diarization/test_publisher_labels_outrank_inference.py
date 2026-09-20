"""A transcript that tags its own turns states who spoke; the roster must not overrule it (#876).

THE DEFECT THIS EXISTS TO PREVENT. An earlier attempt (be1ed96d / ac1751b4, both reverted) routed
publisher transcripts through the roster but threw the names away first: the cue label was replaced
with an anonymous ``SPEAKER_NN`` cluster id before the roster ever saw it, so names were re-derived
from the feed. Measured on ``p01_multi_e02.vtt``, whose publisher states ``Maya`` and ``Sophie``::

    feed_hosts=['Liam Verbeek'] -> [('Liam Verbeek', 'host'), ('SPEAKER_01', None)]

``Liam Verbeek`` published on a voice the file names ``Maya``. The source told us the answer and we
overrode it with a guess — the one thing #876 forbids, and worse than leaving the voice unnamed.

The test written alongside that attempt passed, because it supplied ``feed_hosts=["Maya"]`` — a
value that happened to agree with the fixture. **Every naming assertion here is therefore written
with the feed DISAGREEING with the file**, because agreement cannot distinguish "respected the
publisher" from "re-derived and got lucky".

WHAT IS AND IS NOT TAKEN FROM THE FILE. The publisher states WHO speaks. It never states who HOSTS,
so roles are still resolved from the roster's own evidence with the name held fixed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

pytestmark = [pytest.mark.unit]


_TURNS: List[Tuple[str, str]] = [
    ("SPEAKER_00", "Welcome back to Singletrack Sessions. Today is about trail building."),
    ("SPEAKER_01", "Thanks for having me - great to be here."),
    ("SPEAKER_00", "What is the most underrated piece of trail building?"),
    ("SPEAKER_01", "Drainage. If water leaves on its own the trail lasts a decade."),
]


def _roster(
    stated: Optional[Dict[str, str]],
    known_hosts: Sequence[str] = (),
    detected_guests: Sequence[str] = (),
) -> Dict[str, Tuple[str, str, str]]:
    segments, clock = [], 0.0
    for speaker, _text in _TURNS:
        segments.append(DiarizationSegment(start=clock, end=clock + 8.0, speaker=speaker))
        clock += 8.0

    voice_texts: Dict[str, str] = {}
    for speaker, text in _TURNS:
        voice_texts[speaker] = f"{voice_texts.get(speaker, '')} {text}".strip()

    result = resolve_speaker_roster(
        DiarizationResult(segments=segments, num_speakers=2),
        " ".join(text for _s, text in _TURNS),
        known_hosts=list(known_hosts),
        detected_guests=list(detected_guests),
        voice_texts=voice_texts,
        ordered_turns=_TURNS,
        stated_voice_names=stated,
    )
    return {
        voice: (role.name, role.role, getattr(role, "source", ""))
        for voice, role in result.by_voice.items()
    }


_STATED = {"SPEAKER_00": "Maya", "SPEAKER_01": "Liam"}


def test_a_publisher_label_beats_a_wrong_feed_host() -> None:
    """THE regression test. The feed insists the host is someone the transcript never mentions."""
    by_voice = _roster(_STATED, known_hosts=["Liam Verbeek"])
    assert by_voice["SPEAKER_00"][0] == "Maya", (
        "the feed's host name was published on a voice the transcript names 'Maya' — this is the "
        "ac1751b4 defect: authoring a name the source contradicts (#876)"
    )
    assert by_voice["SPEAKER_01"][0] == "Liam"


def test_a_stale_detected_guest_cannot_rename_a_stated_voice() -> None:
    """The other half of the same defect: show-notes guests are who the episode is ABOUT, which is
    not the same as who spoke.

    The guest named here is a DIFFERENT PERSON from either stated voice. Naming one of them
    ``Sophie van Dalen`` would be authoring a name the transcript contradicts.
    """
    by_voice = _roster(_STATED, known_hosts=["Maya Koster"], detected_guests=["Sophie van Dalen"])
    assert by_voice["SPEAKER_01"][0] == "Liam", by_voice
    assert "Sophie" not in by_voice["SPEAKER_00"][0]


def test_a_stated_mononym_still_snaps_to_the_same_persons_fuller_name() -> None:
    """Canonicalisation is NOT overruling. When the episode states ``Liam Verbeek`` and the
    publisher writes ``Liam``, those are one person, and publishing both spellings would mint two
    KG Person ids for him (ADR-130).

    The distinction that makes this safe is ``_same_person``: a fuller form of the SAME name is
    adopted; an unrelated name is not (asserted directly above).
    """
    by_voice = _roster(_STATED, known_hosts=["Maya Koster"], detected_guests=["Liam Verbeek"])
    assert by_voice["SPEAKER_01"][0] == "Liam Verbeek", by_voice


def test_names_resolve_with_no_feed_host_at_all() -> None:
    """The publisher is sufficient on its own — this is why the earlier `feed_hosts` gate was
    wrong to skip the pass when no host was known."""
    by_voice = _roster(_STATED)
    assert by_voice["SPEAKER_00"][0] == "Maya"
    assert by_voice["SPEAKER_01"][0] == "Liam"


def test_roles_come_from_the_conversation_not_from_the_file() -> None:
    """A publisher says who speaks, never who hosts. The opener welcomes; the other thanks."""
    by_voice = _roster(_STATED)
    assert by_voice["SPEAKER_00"][1] == "host"
    assert by_voice["SPEAKER_01"][1] == "guest"


def test_a_stated_name_snaps_to_the_feed_spelling_of_the_same_person() -> None:
    """ADR-130 canonicalisation still applies: one person, one spelling, one KG id."""
    by_voice = _roster(_STATED, known_hosts=["Maya Koster"])
    assert by_voice["SPEAKER_00"][0] == "Maya Koster"
    # ...but only for the person the feed actually states. Liam has no fuller form to snap to.
    assert by_voice["SPEAKER_01"][0] == "Liam"


@pytest.mark.parametrize("label", ["Ad", "ad", "Sponsor", "Narrator", "Announcer", "Music"])
def test_a_non_person_voice_label_never_names_anybody(label: str) -> None:
    """A sponsor read is not a person. ``Ad`` passes ``is_publishable_speaker_name`` and is NOT
    caught by ``is_bare_speaker_label``, so without an explicit rule it would be published as a
    speaker called "Ad"."""
    by_voice = _roster({"SPEAKER_00": "Maya", "SPEAKER_01": label}, known_hosts=["Maya Koster"])
    assert (
        by_voice["SPEAKER_01"][0] == "SPEAKER_01"
    ), f"{label!r} was published as a person; a production credit is not a speaker (#876)"


@pytest.mark.parametrize("label", ["Speaker 2", "SPEAKER_01", "Speaker"])
def test_a_bare_cluster_label_never_names_anybody(label: str) -> None:
    """Providers that tag turns with positional ids (deepgram/moss, SRT ``Speaker N``) must keep
    behaving exactly as before — those are ids, not names."""
    by_voice = _roster({"SPEAKER_00": "Maya", "SPEAKER_01": label}, known_hosts=["Maya Koster"])
    assert by_voice["SPEAKER_01"][0] == "SPEAKER_01"


def test_the_provenance_says_the_publisher_said_it() -> None:
    """An audit that cannot tell a name the SOURCE stated from one a model inferred cannot audit
    the model at all. ``publisher_transcript`` is its own source, not ``self_intro``."""
    by_voice = _roster(_STATED, known_hosts=["Maya Koster"])
    assert by_voice["SPEAKER_00"][2] == "publisher_transcript"
    assert by_voice["SPEAKER_01"][2] == "publisher_transcript"


def test_without_stated_names_the_roster_is_unchanged() -> None:
    """The ASR path must not move. No stated names -> the old behaviour, byte for byte."""
    by_voice = _roster(None, known_hosts=["Maya Koster"])
    assert by_voice["SPEAKER_00"][0] == "Maya Koster"
    assert by_voice["SPEAKER_00"][2] != "publisher_transcript"
    assert by_voice["SPEAKER_01"][0] == "SPEAKER_01"
