"""Unit tests for the segments contract mapper (#1067, RFC-098 §5).

Pure functions — no HTTP, no disk.
"""

from __future__ import annotations

from podcast_scraper.server.segments_view import (
    segments_relpaths_for_transcript,
    to_contract_segments,
)


class TestSegmentsRelpaths:
    def test_prefers_raw_canonical_then_adfree(self) -> None:
        # Player streams the ORIGINAL audio → raw canonical segments first (matching the
        # original timeline); ad-free is only a last-resort fallback.
        assert segments_relpaths_for_transcript("transcripts/ep1.txt") == [
            "transcripts/ep1.segments.json",
            "transcripts/ep1.adfree.segments.json",
        ]

    def test_strips_adfree_stem(self) -> None:
        assert segments_relpaths_for_transcript("transcripts/ep1.adfree.txt") == [
            "transcripts/ep1.segments.json",
            "transcripts/ep1.adfree.segments.json",
        ]

    def test_backslashes_normalised(self) -> None:
        assert segments_relpaths_for_transcript("transcripts\\ep1.txt")[0] == (
            "transcripts/ep1.segments.json"
        )

    def test_empty_returns_empty(self) -> None:
        assert segments_relpaths_for_transcript("") == []
        assert segments_relpaths_for_transcript("   ") == []


class TestToContractSegments:
    def test_maps_core_fields_and_speaker_precedence(self) -> None:
        raw = [
            {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello.", "speaker_label": "Alice"},
            {"id": 1, "start": 2.5, "end": 5.0, "text": "World.", "speaker_id": "SPEAKER_01"},
            {"start": 5.0, "end": 6.0, "text": "No id.", "speaker": "raw-tag"},
        ]
        out = to_contract_segments(raw)
        assert [s.id for s in out] == ["seg_0000", "seg_0001", "seg_0002"]
        assert out[0].start == 0.0 and out[0].end == 2.5 and out[0].text == "Hello."
        assert out[0].speaker == "Alice"  # speaker_label wins
        assert out[1].speaker == "SPEAKER_01"  # speaker_id fallback
        assert out[2].speaker == "raw-tag"  # speaker fallback

    def test_voice_type_renders_friendly_label(self) -> None:
        # A cameo/commercial voice shows a friendly label; a substantive unknown keeps its raw id;
        # a named voice with no voice_type is unchanged. The id-bearing speaker_label is untouched.
        raw = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "a",
                "speaker_label": "SPEAKER_02",
                "voice_type": "cameo",
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "b",
                "speaker_label": "S_03",
                "voice_type": "commercial",
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "c",
                "speaker_label": "SPEAKER_04",
                "voice_type": "unknown",
            },
            {"start": 3.0, "end": 4.0, "text": "d", "speaker_label": "Alice"},  # named
        ]
        out = to_contract_segments(raw)
        assert out[0].speaker == "Brief speaker"
        assert out[1].speaker == "Advertisement"
        assert out[2].speaker == "SPEAKER_04"  # substantive unknown keeps its raw id
        assert out[3].speaker == "Alice"

    def test_skips_malformed_entries(self) -> None:
        raw = [
            {"start": 0.0, "end": 1.0, "text": "ok"},
            {"start": "bad", "end": 1.0, "text": "x"},  # non-numeric
            {"start": 1.0, "text": "missing end"},  # missing end
            {"start": 1.0, "end": 2.0, "text": 123},  # non-str text
            "not a dict",
        ]
        out = to_contract_segments(raw)
        assert len(out) == 1
        assert out[0].text == "ok"
        assert out[0].speaker is None

    def test_non_list_returns_empty(self) -> None:
        assert to_contract_segments({"start": 0}) == []
        assert to_contract_segments(None) == []
