"""A publisher transcript's speakers survive parsing, and its hosts are named, not guessed (#2075).

Found verifying `retranscript_only` on one Odd Lots episode on the DGX. Synthetic cues; the shapes
are the measured ones:

* its SRT writes the speaker as a line prefix (`Speaker 3: …`), which `parse_srt` never read — the
  feed lists the SRT first, so every ingest lost the speakers the file carried;
* one host's voice says "And I'm Joe Weisenthal" at the open and, garbled, "And I'm Jill
  Wiesenthal" at the close: counted as two people it was suppressed as a montage, and the
  introduction reader then gave the GUEST's name to that host's voice;
* the other host's "I'm Tracy Allaway" was snapped only for host-candidate voices, so she was
  published under the ASR spelling, discarded by the resolver as unstated, and cast as a guest.
"""

from __future__ import annotations

import pytest

from podcast_scraper.transcript_formats import parse_srt

pytestmark = pytest.mark.unit


class TestSrtSpeakerPrefix:
    SRT = (
        "1\n00:00:18,079 --> 00:00:20,619\nSpeaker 3: Welcome back. I'm Ada Brook.\n\n"
        "2\n00:00:22,400 --> 00:00:23,570\nSpeaker 4: And I'm Ben Carver.\n\n"
        "3\n00:00:24,000 --> 00:00:25,000\nNote: this cue has no speaker.\n"
    )

    def test_the_prefix_becomes_the_segment_speaker_and_leaves_the_text(self) -> None:
        plain, segs = parse_srt(self.SRT)
        assert [s.get("speaker") for s in segs] == ["Speaker 3", "Speaker 4", None]
        assert segs[0]["text"].startswith("Welcome")
        assert "Speaker 3:" not in plain

    def test_a_prose_colon_is_not_a_speaker(self) -> None:
        _plain, segs = parse_srt(self.SRT)
        assert segs[2]["text"].startswith("Note: this cue")
