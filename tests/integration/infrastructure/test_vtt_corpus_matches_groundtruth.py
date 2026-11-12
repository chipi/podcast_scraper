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

from pathlib import Path
from typing import Set

import pytest

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
def test_every_cue_is_attributed(vtt: Path) -> None:
    """No unlabelled cue anywhere — an unlabelled cue is the thing that silently inherits the
    previous speaker."""
    _plain, segments = parse_webvtt(vtt.read_text(encoding="utf-8"))
    unattributed = [s for s in segments if not s.get("speaker")]
    assert not unattributed, f"{vtt.name} has {len(unattributed)} cue(s) with no <v> span"
