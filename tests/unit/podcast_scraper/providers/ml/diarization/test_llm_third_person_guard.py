"""An LLM name on a voice whose own words speak of that person in the third person comes off (v54).

The resolver's third-person check reads only the passages it hands the model; a voice's full text
can still say "That's <name>, the co-founder ..." or read the credits. Measured over the stored DGX
runs and step 5 (2026-10-25): 6 bad names removed, 3 right lost (two in one episode whose host
sign-off the diarization put in the guests' voices; one host introducing itself "with me <name>").
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.pipeline import _apply_llm_third_person_guard
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

pytestmark = [pytest.mark.unit]


def _roster(**voices):
    return SpeakerRoster(by_voice=dict(voices), num_speakers=len(voices))


def _seg(voice, text, start=0.0):
    return ({"start": start, "end": start + 10.0, "text": text}, voice)


def test_an_llm_name_the_voice_speaks_of_in_the_third_person_comes_off() -> None:
    roster = _roster(
        SPEAKER_00=SpeakerRole(name="Jane Doe", role="guest", named=True, source="llm_resolution"),
        SPEAKER_01=SpeakerRole(name="Sam Roe", role="host", named=True, source="llm_resolution"),
    )
    aligned = [
        _seg("SPEAKER_00", "That's Jane Doe, the co-founder and COO of Acme, and her partner."),
        _seg("SPEAKER_01", "Thanks for having me on the show today.", 10.0),
    ]
    out = _apply_llm_third_person_guard(roster, aligned)
    v = out.by_voice["SPEAKER_00"]
    assert (v.named, v.name, v.role, v.source) == (
        False,
        "SPEAKER_00",
        "unknown",
        "llm_third_person",
    )
    assert out.by_voice["SPEAKER_01"] == roster.by_voice["SPEAKER_01"]


def test_a_refused_host_keeps_the_host_role() -> None:
    roster = _roster(
        SPEAKER_00=SpeakerRole(
            name="Sam Rowley", role="host", named=True, source="llm_resolution"
        )
    )
    aligned = [_seg("SPEAKER_00", "In this episode, Sam Rowly sits down with a general partner.")]
    v = _apply_llm_third_person_guard(roster, aligned).by_voice["SPEAKER_00"]
    assert (v.named, v.role) == (False, "host")


def test_names_from_other_rungs_are_untouched() -> None:
    roster = _roster(
        SPEAKER_00=SpeakerRole(name="Jane Doe", role="guest", named=True, source="self_intro")
    )
    aligned = [_seg("SPEAKER_00", "That's Jane Doe, the co-founder and COO of Acme.")]
    assert _apply_llm_third_person_guard(roster, aligned) is roster


def test_a_voice_that_introduces_itself_keeps_the_name() -> None:
    roster = _roster(
        SPEAKER_00=SpeakerRole(name="Jane Doe", role="host", named=True, source="llm_resolution")
    )
    aligned = [_seg("SPEAKER_00", "I'm Jane Doe, and that's Jane Doe's brother on the line.")]
    assert _apply_llm_third_person_guard(roster, aligned) is roster
