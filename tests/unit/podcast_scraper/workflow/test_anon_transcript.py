"""D-40: the pre-naming render is kept as `<base>.anon.txt`.

WHY. The order is diarize → write anonymous → name → re-render, and that final re-render
overwrites `.txt`. The anonymous
screenplay is the only human-readable view of what the pipeline saw BEFORE it decided who was
speaking, and it is the first thing anyone debugging a naming failure asks for. D-40's rule is
that no stage overwrites another stage's output.

IT IS DERIVED FROM `speaker`, NOT `speaker_label`. The voice id is frozen at diarization and
naming never touches it — only the label is updated — so the anonymous render is reconstructible
at any point afterwards. Keeping it is cheapness and clarity, not recoverability, and that is
worth stating because it bounds how much the loss of this file costs: nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.workflow.transcript_resolution import (
    anon_transcript_relpath,
    text_relpath_candidates,
    TranscriptPurpose,
)

pytestmark = pytest.mark.unit

REL = "transcripts/01 - ep.txt"


def _segments(named: bool) -> List[Dict[str, Any]]:
    """Two voices. `named=True` is post-naming (labels are people), False is pre-naming."""
    rows = [
        (0.0, 30.0, "SPEAKER_00", "Dana Reyes", "Welcome back. I'm Dana Reyes."),
        (30.0, 60.0, "SPEAKER_01", "Marcus Webb", "Thanks for having me."),
        (60.0, 90.0, "SPEAKER_00", "Dana Reyes", "Tell me what you build."),
    ]
    return [
        {
            "start": s,
            "end": e,
            "speaker": voice,
            "speaker_label": person if named else voice,
            "text": text,
        }
        for s, e, voice, person, text in rows
    ]


class TestItIsNotARESOLUTIONCandidate:
    """A reader that resolved to the anonymous render would show `SPEAKER_01` where a person's
    name belongs — a silent, plausible-looking regression on every surface."""

    @pytest.mark.parametrize("purpose", [TranscriptPurpose.ANALYSIS, TranscriptPurpose.TIMELINE])
    def test_no_purpose_lists_it(self, purpose: TranscriptPurpose) -> None:
        candidates = text_relpath_candidates(REL, purpose=purpose)
        assert not any(".anon." in c for c in candidates), candidates

    def test_but_a_reference_TO_it_still_canonicalizes(self) -> None:
        """A `transcript_ref` pointing at the anonymous body must still resolve like its base —
        the suffixes stack, and stripping only the outermost one would leave `ep1.anon` and
        resolve nothing at all."""
        from podcast_scraper.workflow.transcript_resolution import _canonical_relpath

        assert _canonical_relpath("transcripts/ep1.anon.txt") == "transcripts/ep1.txt"
        assert _canonical_relpath("transcripts/ep1.anon.adfree.txt") == "transcripts/ep1.txt"


class TestTheName:
    def test_the_suffix_sits_before_the_extension(self) -> None:
        assert anon_transcript_relpath("transcripts/01 - ep.txt") == (
            "transcripts/01 - ep.anon.txt"
        )

    def test_an_extensionless_ref_still_gets_txt(self) -> None:
        assert anon_transcript_relpath("transcripts/ep") == "transcripts/ep.anon.txt"
