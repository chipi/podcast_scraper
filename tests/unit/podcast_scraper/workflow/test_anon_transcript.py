"""D-40: the pre-naming render is kept as `<base>.anon.txt`.

WHY. With naming moved after translation (D-34) the order becomes diarize → write anonymous →
translate → name → re-render, and that final re-render overwrites `.txt`. The anonymous
screenplay is the only human-readable view of what the pipeline saw BEFORE it decided who was
speaking, and it is the first thing anyone debugging a naming failure asks for. D-40's rule is
that no stage overwrites another stage's output.

IT IS DERIVED FROM `speaker`, NOT `speaker_label`. The voice id is frozen at diarization and
naming never touches it — only the label is updated — so the anonymous render is reconstructible
at any point afterwards. Keeping it is cheapness and clarity, not recoverability, and that is
worth stating because it bounds how much the loss of this file costs: nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.workflow.episode_processor import _write_anon_transcript
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


class TestWhatItWrites:
    def test_it_writes_the_anonymous_render(self, tmp_path: Path) -> None:
        rel = _write_anon_transcript(_segments(named=True), REL, str(tmp_path))
        assert rel == "transcripts/01 - ep.anon.txt"
        body = (tmp_path / rel).read_text(encoding="utf-8")
        assert "SPEAKER_00:" in body and "SPEAKER_01:" in body

    def test_no_LABEL_is_a_resolved_name(self, tmp_path: Path) -> None:
        """The whole point. A name in the label position would mean it rendered
        `speaker_label`, which is the named view `.txt` already holds.

        Checked on the LABELS, not on the whole body: the names also appear inside the SPEECH
        ("I'm Dana Reyes") and must survive there verbatim — that is the sentence the roster
        reads to resolve the voice in the first place. A blanket "the name is absent" assertion
        is what this test said first, and it was wrong in a way that would have pushed the fix
        toward stripping real transcript content.
        """
        rel = _write_anon_transcript(_segments(named=True), REL, str(tmp_path))
        assert rel is not None
        body = (tmp_path / rel).read_text(encoding="utf-8")
        labels = {
            line.split(":", 1)[0]
            for line in body.splitlines()
            if ":" in line and not line.startswith(" ")
        }
        assert labels == {"SPEAKER_00", "SPEAKER_01"}, sorted(labels)

    def test_the_name_still_appears_in_the_SPEECH(self, tmp_path: Path) -> None:
        """The other side of it. The self-intro is the evidence naming runs ON, so stripping it
        would break the thing this file exists to serve."""
        rel = _write_anon_transcript(_segments(named=True), REL, str(tmp_path))
        assert rel is not None
        assert "I'm Dana Reyes" in (tmp_path / rel).read_text(encoding="utf-8")

    def test_the_SPEECH_survives_verbatim(self, tmp_path: Path) -> None:
        """Only the labels change. If the text moved too, this would not be the same episode."""
        rel = _write_anon_transcript(_segments(named=True), REL, str(tmp_path))
        assert rel is not None
        body = (tmp_path / rel).read_text(encoding="utf-8")
        for expected in ("Welcome back. I'm Dana Reyes.", "Thanks for having me."):
            assert expected in body

    def test_it_does_not_touch_the_named_transcript(self, tmp_path: Path) -> None:
        """D-40's rule, asserted directly: no stage overwrites another stage's output."""
        named_path = tmp_path / REL
        named_path.parent.mkdir(parents=True, exist_ok=True)
        named_path.write_text("Dana Reyes: original\n", encoding="utf-8")
        _write_anon_transcript(_segments(named=True), REL, str(tmp_path))
        assert named_path.read_text(encoding="utf-8") == "Dana Reyes: original\n"


class TestWhenItWritesNOTHING:
    def test_undiarized_segments_produce_no_file(self, tmp_path: Path) -> None:
        """No `speaker` field means no diarization ran, so there is no anonymous view to keep."""
        segments = [{"start": 0.0, "end": 30.0, "text": "Hello.", "speaker_label": "SPEAKER_00"}]
        assert _write_anon_transcript(segments, REL, str(tmp_path)) is None
        assert not (tmp_path / "transcripts").exists()

    def test_an_IDENTICAL_render_produces_no_file(self, tmp_path: Path) -> None:
        """Naming resolved nothing, so the two renders are the same bytes. An identical copy is
        not a second artifact — writing one for every unnamed episode would double the transcript
        corpus to say nothing."""
        assert _write_anon_transcript(_segments(named=False), REL, str(tmp_path)) is None

    def test_empty_segments_produce_no_file(self, tmp_path: Path) -> None:
        assert _write_anon_transcript([], REL, str(tmp_path)) is None

    def test_a_write_failure_is_not_fatal(self, tmp_path: Path) -> None:
        """A debugging view must not cost the episode. Its real artifacts are already on disk by
        the time this runs."""
        blocker = tmp_path / "transcripts"
        blocker.write_text("not a directory", encoding="utf-8")
        assert _write_anon_transcript(_segments(named=True), REL, str(tmp_path)) is None


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
