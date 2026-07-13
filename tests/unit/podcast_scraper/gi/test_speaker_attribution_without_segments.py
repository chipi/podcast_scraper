"""An insight must know who said it, even when there are no transcript segments.

The pipeline's only speaker path read `speaker_id` off diarized transcript SEGMENTS. No profile
produces them — `backfill_transcript_segments` is false everywhere, and enabling it forces a
re-transcription — so every quote shipped with `speaker_id: None` and no insight has ever known who
said it. Meanwhile the diarized transcript names its own speakers in plain text ("Kevin Roose: ...")
and every quote carries a char offset, so the answer was sitting in the data, unread.

This matters beyond tidiness: an insight is a claim a PERSON made. A host summarising a deal is a
headline; the guest expert taking a position is why the episode exists. Without the speaker the two
are indistinguishable, and "stance" is unfalsifiable — a position needs an owner.
"""

from __future__ import annotations

from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.speakers import build_unverified_named_turns, speaker_for_char

TRANSCRIPT = (
    "Kevin Roose: OpenAI rewrote its Microsoft deal this week.\n"
    "Casey Newton: And the AGI revenue clause is gone entirely.\n"
    "Dr. Adam Rodman: Younger doctors lean on these tools far more than we admit.\n"
)


class TestUnverifiedNamedTurns:
    def test_reads_speakers_straight_out_of_the_transcript(self) -> None:
        turns = build_unverified_named_turns(TRANSCRIPT)
        assert [name for _, name in turns] == [
            "Kevin Roose",
            "Casey Newton",
            "Dr. Adam Rodman",
        ]

    def test_a_quote_offset_resolves_to_its_speaker(self) -> None:
        turns = build_unverified_named_turns(TRANSCRIPT)
        offset = TRANSCRIPT.index("Younger doctors")
        assert speaker_for_char(offset, turns) == "Dr. Adam Rodman"

        offset = TRANSCRIPT.index("AGI revenue clause")
        assert speaker_for_char(offset, turns) == "Casey Newton"

    def test_prose_and_publisher_labels_are_not_speakers(self) -> None:
        """No whitelist is available here, so the person heuristic has to do the rejecting.

        Under-attributing beats attributing wrongly — the module's stated contract.
        """
        text = (
            "Note: this is an editorial aside.\n"
            "Bloomberg: markets closed lower.\n"
            "Kevin Roose: but the deal still stands.\n"
        )
        names = [name for _, name in build_unverified_named_turns(text)]
        assert names == ["Kevin Roose"]
        assert "Note" not in names
        assert "Bloomberg" not in names


class TestInsightCarriesItsSpeaker:
    def test_speaker_comes_from_the_insights_first_grounded_quote(self) -> None:
        from podcast_scraper.gi.pipeline import _speaker_for_insight

        turns = build_unverified_named_turns(TRANSCRIPT)
        start = TRANSCRIPT.index("Younger doctors")
        quote = GroundedQuote(
            text="Younger doctors lean on these tools far more than we admit.",
            char_start=start,
            char_end=start + 58,
            qa_score=0.9,
            nli_score=0.9,
        )

        who = _speaker_for_insight([quote], TRANSCRIPT, None, turns)
        assert who == "Dr. Adam Rodman"

    def test_an_ungrounded_insight_has_no_speaker(self) -> None:
        """No quote means no evidence, and no evidence means nobody to attribute it to."""
        from podcast_scraper.gi.pipeline import _speaker_for_insight

        turns = build_unverified_named_turns(TRANSCRIPT)
        assert _speaker_for_insight([], TRANSCRIPT, None, turns) is None
