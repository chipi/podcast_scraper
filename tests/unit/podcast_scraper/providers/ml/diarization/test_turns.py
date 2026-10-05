"""`build_turns`: turns, sentences, and the invariants that keep them honest (RFC-123 / S1.1).

THE FAILURE THESE GUARD AGAINST is not a crash. A turn whose char span disagrees with the rendered
text produces a quote attributed to the WRONG SPEAKER, silently, and every downstream stage reports
success. So the tests here are mostly about the identity between a turn's span and the screenplay
it was built from — and they drive the REAL formatter rather than hand-written offsets, because
hand-written offsets would test my arithmetic instead of the contract.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.providers.ml.diarization.turns import (
    BACKCHANNEL_MAX_SEC,
    build_turns,
    TurnInvariantError,
)

pytestmark = pytest.mark.unit


def _seg(start: float, end: float, text: str, label: str, **extra: Any) -> Dict[str, Any]:
    return {"start": start, "end": end, "text": text, "speaker_label": label, **extra}


def _render(segments: List[Dict[str, Any]]):
    """Screenplay + offset segments, from the REAL formatter."""
    return format_diarized_screenplay_with_offsets(segments)


class TestTurnsAreTheScreenplayLines:
    def test_consecutive_same_label_segments_coalesce_into_one_turn(self) -> None:
        segs = [
            _seg(0.0, 2.0, "First part.", "Maya"),
            _seg(2.0, 4.0, "Second part.", "Maya"),
            _seg(4.0, 6.0, "A reply.", "Liam"),
        ]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text)

        assert [t.turn_id for t in turns.turns] == ["t0000", "t0001"]
        assert turns.turns[0].segment_idx == [0, 1]
        assert turns.turns[1].segment_idx == [2]

    def test_the_span_is_the_rendered_text(self) -> None:
        """The identity that makes char spans trustworthy rather than plausible."""
        segs = [
            _seg(0.0, 2.0, "First part.", "Maya"),
            _seg(2.0, 4.0, "Second part.", "Maya"),
            _seg(4.0, 6.0, "A reply.", "Liam"),
        ]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text)

        assert text[turns.turns[0].char_start : turns.turns[0].char_end] == (
            "First part. Second part."
        )
        assert text[turns.turns[1].char_start : turns.turns[1].char_end] == "A reply."

    def test_the_span_excludes_the_label_prefix(self) -> None:
        """RFC-123: char_start points at the first speech character, AFTER `Label: `."""
        segs = [_seg(0.0, 2.0, "Hello there.", "Maya")]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text)

        assert text.startswith("Maya: ")
        assert turns.turns[0].char_start == len("Maya: ")
        assert not text[turns.turns[0].char_start :].startswith("Maya")

    def test_an_alternating_conversation_yields_one_turn_per_line(self) -> None:
        segs = []
        for i in range(6):
            segs.append(
                _seg(i * 2.0, i * 2.0 + 2.0, f"Line {i}.", "Maya" if i % 2 == 0 else "Liam")
            )
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text)

        assert len(turns.turns) == 6
        assert [t.speaker_label for t in turns.turns] == ["Maya", "Liam"] * 3
        assert len(text.strip().splitlines()) == 6


class TestBackchannels:
    def test_a_short_interjection_is_flagged_not_merged(self) -> None:
        """RFC-123 §2.2: merging would attribute the interjection to the surrounding speaker,
        which is falsification."""
        segs = [
            _seg(0.0, 5.0, "A long thought that runs on for a while.", "Maya"),
            _seg(5.0, 5.8, "Yeah.", "Liam"),
            _seg(5.8, 10.0, "And the continuation of it.", "Maya"),
        ]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text)

        assert len(turns.turns) == 3, "the interjection must not be merged away"
        assert turns.turns[1].speaker_label == "Liam"
        assert turns.turns[1].backchannel is True
        assert turns.turns[0].backchannel is False
        assert turns.turns[2].backchannel is False

    def test_a_long_utterance_is_not_a_backchannel(self) -> None:
        segs = [_seg(0.0, BACKCHANNEL_MAX_SEC + 2.0, "Yeah.", "Liam")]
        _text, offsets = _render(segs)
        assert build_turns(offsets).turns[0].backchannel is False

    def test_a_wordy_short_utterance_is_not_a_backchannel(self) -> None:
        segs = [_seg(0.0, 1.0, "One two three four five.", "Liam")]
        _text, offsets = _render(segs)
        assert build_turns(offsets).turns[0].backchannel is False


class TestSentences:
    def test_sentences_partition_the_turn(self) -> None:
        segs = [_seg(0.0, 6.0, "First one. Second one? Third one!", "Maya")]
        text, offsets = _render(segs)
        turn = build_turns(offsets, screenplay_text=text).turns[0]

        assert [text[s.char_start : s.char_end] for s in turn.sentences] == [
            "First one.",
            "Second one?",
            "Third one!",
        ]
        assert turn.sentences[0].char_start == turn.char_start
        assert turn.sentences[-1].char_end == turn.char_end

    def test_a_protected_abbreviation_does_not_split(self) -> None:
        segs = [_seg(0.0, 4.0, "We asked Dr. Fischer about it. She agreed.", "Maya")]
        text, offsets = _render(segs)
        turn = build_turns(offsets, screenplay_text=text).turns[0]

        assert [text[s.char_start : s.char_end] for s in turn.sentences] == [
            "We asked Dr. Fischer about it.",
            "She agreed.",
        ]

    def test_timing_says_whether_it_was_interpolated(self) -> None:
        """A consumer needing real precision must be able to tell, not infer from behaviour.

        My first version of this asserted the FIRST sentence is `segment_exact`. It is not, and
        the code is right: three sentences share one segment, so only the LAST one ends at the
        segment's boundary. A sentence is exact only when BOTH ends coincide with segment edges.
        """
        segs = [_seg(0.0, 9.0, "One. Two. Three.", "Maya")]
        text, offsets = _render(segs)
        turn = build_turns(offsets, screenplay_text=text).turns[0]

        assert len(turn.sentences) == 3
        # None can be exact: each shares a segment with its neighbours.
        assert [s.timing for s in turn.sentences] == ["segment_interpolated"] * 3
        # And the interpolation is monotonic and inside the segment.
        assert turn.sentences[0].start_ms == 0
        assert turn.sentences[-1].end_ms == 9000
        starts = [s.start_ms for s in turn.sentences]
        assert starts == sorted(starts), starts

    def test_one_sentence_per_segment_is_exact(self) -> None:
        segs = [
            _seg(0.0, 2.0, "First.", "Maya"),
            _seg(2.0, 4.0, "Second.", "Maya"),
        ]
        text, offsets = _render(segs)
        turn = build_turns(offsets, screenplay_text=text).turns[0]

        assert len(turn.sentences) == 2
        assert [s.timing for s in turn.sentences] == ["segment_exact", "segment_exact"]
        assert turn.sentences[0].start_ms == 0
        assert turn.sentences[1].end_ms == 4000

    def test_sentence_times_are_monotonic_within_a_turn(self) -> None:
        segs = [
            _seg(0.0, 3.0, "One. Two.", "Maya"),
            _seg(3.0, 6.0, "Three. Four.", "Maya"),
        ]
        text, offsets = _render(segs)
        turn = build_turns(offsets, screenplay_text=text).turns[0]

        times = [(s.start_ms, s.end_ms) for s in turn.sentences]
        for (s1, e1), (s2, _e2) in zip(times, times[1:]):
            assert s1 <= e1, times
            assert s1 <= s2, times


class TestRoleTruthPassesThrough:
    def test_speaker_role_and_voice_type_are_carried_not_rederived(self) -> None:
        """The formatter passes these three keys through; re-deriving role here is what
        resurrected the guest-as-host bug once already."""
        segs = [
            _seg(0.0, 2.0, "Hello.", "Maya", speaker="person:maya", speaker_role="host"),
            _seg(2.0, 4.0, "Hi.", "Ad", voice_type="commercial"),
        ]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text).turns

        assert turns[0].speaker == "person:maya"
        assert turns[0].speaker_role == "host"
        assert turns[0].voice_type is None
        assert turns[1].voice_type == "commercial"
        assert turns[1].speaker_role is None


class TestTheInvariantsCanFire:
    """An invariant that cannot fail is documentation, not a check."""

    def test_a_span_disagreeing_with_the_text_raises(self) -> None:
        """Shift a span FORWARD, so ordering still holds and only the text identity is broken.

        Corrupting char_start to 0 (my first attempt) trips the ORDERING invariant first, which
        is correct behaviour but tests a different guard. This isolates the text check.
        """
        segs = [
            _seg(0.0, 2.0, "First part.", "Maya"),
            _seg(2.0, 4.0, "A reply.", "Liam"),
        ]
        text, offsets = _render(segs)
        offsets[1]["char_start"] += 3
        offsets[1]["char_end"] += 3

        with pytest.raises(TurnInvariantError) as exc:
            build_turns(offsets, screenplay_text=text)
        assert "disagrees with the rendered text" in str(exc.value)

    def test_a_span_starting_before_the_previous_turn_ended_raises(self) -> None:
        """The ordering guard, which is what a wholly wrong offset hits first."""
        segs = [
            _seg(0.0, 2.0, "First part.", "Maya"),
            _seg(2.0, 4.0, "A reply.", "Liam"),
        ]
        text, offsets = _render(segs)
        offsets[1]["char_start"] = 0

        with pytest.raises(TurnInvariantError) as exc:
            build_turns(offsets, screenplay_text=text)
        assert "non-overlapping" in str(exc.value)

    def test_overlapping_turns_raise(self) -> None:
        segs = [
            _seg(0.0, 2.0, "First.", "Maya"),
            _seg(2.0, 4.0, "Second.", "Liam"),
        ]
        _text, offsets = _render(segs)
        offsets[1]["char_start"] = 0  # now starts before turn 0 ended

        with pytest.raises(TurnInvariantError) as exc:
            build_turns(offsets)
        assert "non-overlapping" in str(exc.value)

    def test_reversed_times_raise(self) -> None:
        segs = [_seg(5.0, 1.0, "Backwards.", "Maya")]
        _text, offsets = _render(segs)
        with pytest.raises(TurnInvariantError) as exc:
            build_turns(offsets)
        assert "end_ms < start_ms" in str(exc.value)

    def test_every_segment_lands_in_exactly_one_turn(self) -> None:
        segs = [_seg(i, i + 1, f"Line {i}.", "Maya" if i % 2 else "Liam") for i in range(8)]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text)

        seen = [i for t in turns.turns for i in t.segment_idx]
        assert sorted(seen) == list(range(len(offsets)))
        assert len(seen) == len(set(seen)), "a segment appeared in two turns"


class TestEdges:
    def test_no_segments_yields_no_turns(self) -> None:
        turns = build_turns([])
        assert turns.turns == []
        assert turns.to_dict()["version"] == "1.0"

    def test_blank_segments_are_dropped_by_the_formatter_not_here(self) -> None:
        """build_turns must account for every segment it RECEIVES; the formatter is what drops
        blanks, and it drops them from the offsets too, so the counts agree."""
        segs = [
            _seg(0.0, 1.0, "Real.", "Maya"),
            _seg(1.0, 2.0, "   ", "Maya"),
            _seg(2.0, 3.0, "Also real.", "Liam"),
        ]
        text, offsets = _render(segs)
        assert len(offsets) == 2, "the formatter dropped the blank"
        turns = build_turns(offsets, screenplay_text=text)
        assert sum(len(t.segment_idx) for t in turns.turns) == 2

    def test_unnamed_voices_group_by_their_label(self) -> None:
        """The pre-naming state: SPEAKER_NN labels, which is what a fresh diarization yields."""
        segs = [
            _seg(0.0, 2.0, "Primera parte.", "SPEAKER_00"),
            _seg(2.0, 4.0, "Segunda parte.", "SPEAKER_00"),
            _seg(4.0, 6.0, "Una respuesta.", "SPEAKER_01"),
        ]
        text, offsets = _render(segs)
        turns = build_turns(offsets, screenplay_text=text).turns

        assert [t.speaker_label for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
        assert turns[0].segment_idx == [0, 1]

    def test_turn_ids_are_stable_across_rebuilds(self) -> None:
        segs = [_seg(i, i + 1, f"Line {i}.", "Maya" if i % 2 else "Liam") for i in range(5)]
        text, offsets = _render(segs)
        a = build_turns(offsets, screenplay_text=text)
        b = build_turns(offsets, screenplay_text=text)
        assert a.to_dict() == b.to_dict()


class TestOverRealFixtureSegments:
    """The builder against a committed corpus sidecar, not constructed input."""

    def test_it_builds_over_the_app_validation_corpus(self) -> None:
        import json
        from pathlib import Path

        repo = Path(__file__).resolve().parents[6]
        seg_path = (
            repo
            / "tests/fixtures/app-validation-corpus/v3/feeds/p01/run_20260101-000000"
            / "transcripts/p01_e01.segments.json"
        )
        raw = json.loads(seg_path.read_text(encoding="utf-8"))
        assert raw, "fixture segments missing"

        # The committed sidecar is the RAW shape (no char offsets), so render first -- which is
        # exactly what S1.2 will do at the point the sidecar is written.
        text, offsets = format_diarized_screenplay_with_offsets(raw)
        turns = build_turns(offsets, screenplay_text=text)

        assert turns.turns, "no turns built from a real episode"
        assert len(turns.turns) <= len(offsets), "turns cannot outnumber segments"
        assert sum(len(t.segment_idx) for t in turns.turns) == len(offsets)
        assert all(t.sentences for t in turns.turns if text[t.char_start : t.char_end].strip())
        # Every turn's span still reproduces its text on real data.
        for t in turns.turns:
            expected = " ".join((offsets[i].get("text") or "").strip() for i in t.segment_idx)
            assert text[t.char_start : t.char_end] == expected
