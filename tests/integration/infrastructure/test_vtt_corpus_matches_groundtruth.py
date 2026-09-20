"""The speaker-labelled VTT corpus must agree with the ground truth it was generated beside.

``tests/fixtures/scripts/transcripts_to_vtt.py`` turns each ``.txt`` fixture into a WebVTT twin
that names every turn (``<v Maya>``). Those labels are the corpus's answer key for speaker
attribution: stack-test mounts feeds that serve them, and the roster publishes them as speakers
without Whisper or a diarizer ever running. A generator that drifts from ``groundtruth.json``
would quietly teach the pipeline the wrong answer and still look green.

Two independent properties, because they fail differently:

* the people named are exactly the people the ground truth says spoke — a MISSING name means an
  episode silently loses a speaker; an EXTRA one means we would publish somebody nobody said;
* an episode with a sponsor read carries a voice that is not a person, so the ad is its own
  cluster rather than being folded onto whoever spoke before it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

import pytest

from podcast_scraper.providers.ml.diarization.roster import _is_person_voice_label
from podcast_scraper.transcript_formats import parse_webvtt

pytestmark = [pytest.mark.integration]

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transcripts" / "v3"


def _vtt_files() -> list[Path]:
    return sorted(
        p for p in _FIXTURES.glob("*.vtt") if (_FIXTURES / f"{p.stem}.groundtruth.json").is_file()
    )


def _spans(vtt: Path) -> Set[str]:
    _plain, segments = parse_webvtt(vtt.read_text(encoding="utf-8"))
    return {s["speaker"] for s in segments if s.get("speaker")}


def test_the_corpus_has_vtt_twins_at_all() -> None:
    """Guards the whole file: an empty glob would make every test below vacuously pass."""
    assert (
        len(_vtt_files()) >= 40
    ), "the v3 VTT twins are missing — regenerate with transcripts_to_vtt.py"


@pytest.mark.parametrize("vtt", _vtt_files(), ids=lambda p: p.stem)
def test_the_named_people_are_exactly_the_groundtruth_speakers(vtt: Path) -> None:
    """Set equality, both directions.

    ``groundtruth.speakers`` may contain a label that is deliberately NOT a publishable person
    name — ``p08``'s host is ``A. correspondent`` ("non-NER" in FIXTURES_SPEC.md), an intentional
    edge case for a voice that must never become a Person node. Those are compared as voices that
    exist but stay unnamed, which is exactly what the pipeline does with them.
    """
    ground = json.loads((_FIXTURES / f"{vtt.stem}.groundtruth.json").read_text(encoding="utf-8"))
    expected = {s for s in (ground.get("speakers") or []) if _is_person_voice_label(s)}
    named = {s for s in _spans(vtt) if _is_person_voice_label(s)}

    assert named == expected, (
        f"{vtt.name} names {sorted(named)} but the ground truth says {sorted(expected)}; "
        "a missing name loses a speaker, an extra one publishes somebody nobody said (#876)"
    )


@pytest.mark.parametrize(
    "vtt",
    [
        p
        for p in _vtt_files()
        if json.loads((_FIXTURES / f"{p.stem}.groundtruth.json").read_text()).get("num_ad_voices")
    ],
    ids=lambda p: p.stem,
)
def test_an_ad_bearing_episode_carries_a_non_person_voice(vtt: Path) -> None:
    """The sponsor read must be ITS OWN labelled voice.

    With no span, ``alignment.py`` attributes the cue to ``last_speaker`` — putting ad copy in a
    real person's mouth in ``.segments.json`` and reporting one voice fewer than
    ``expected_diarized_voices``, which counts ``num_ad_voices``.
    """
    non_people = {s for s in _spans(vtt) if not _is_person_voice_label(s)}
    assert non_people, (
        f"{vtt.name} has num_ad_voices set but every span is a person's name, so the sponsor read "
        "would be folded onto whoever spoke before it"
    )


@pytest.mark.parametrize("vtt", _vtt_files(), ids=lambda p: p.stem)
def test_every_cue_is_attributed(vtt: Path) -> None:
    """No unlabelled cue anywhere — an unlabelled cue is the thing that silently inherits the
    previous speaker."""
    _plain, segments = parse_webvtt(vtt.read_text(encoding="utf-8"))
    unattributed = [s for s in segments if not s.get("speaker")]
    assert not unattributed, f"{vtt.name} has {len(unattributed)} cue(s) with no <v> span"
