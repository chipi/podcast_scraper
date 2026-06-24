"""Unit tests for the GI → consumer insights projection (#1068).

Pure dict-in / model-out — no HTTP, no disk.
"""

from __future__ import annotations

from podcast_scraper.server.app_gi_view import insights_from_gi


def _gi() -> dict:
    return {
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {
                    "text": "Transformers scale with data.",
                    "grounded": True,
                    "insight_type": "claim",
                    "confidence": 0.8,
                    "position_hint": "0.2",
                },
            },
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {
                    "text": "the thing about transformers is",
                    "speaker_id": "SPEAKER_00",
                    "char_start": 10,
                    "char_end": 41,
                    "timestamp_start_ms": 12400,
                    "timestamp_end_ms": 18700,
                },
            },
            {"id": "insight:2", "type": "Insight", "properties": {"text": "Ungrounded take."}},
            {"id": "insight:3", "type": "Insight", "properties": {}},  # no text → skipped
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
            {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:jane-doe"},
        ],
    }


class TestInsightsFromGi:
    def test_maps_insight_with_supporting_quote(self) -> None:
        out = insights_from_gi(_gi())
        ids = [i.id for i in out]
        assert ids == ["insight:1", "insight:2"]  # insight:3 (no text) skipped

        i1 = out[0]
        assert i1.text == "Transformers scale with data."
        assert i1.grounded is True
        assert i1.insight_type == "claim"
        assert i1.confidence == 0.8
        assert i1.position_hint == "0.2"
        assert len(i1.quotes) == 1
        q = i1.quotes[0]
        assert q.text == "the thing about transformers is"
        assert q.speaker == "SPEAKER_00"
        assert (q.char_start, q.char_end) == (10, 41)
        assert (q.start_ms, q.end_ms) == (12400, 18700)

    def test_ungrounded_insight_has_no_quotes_and_grounded_false(self) -> None:
        i2 = insights_from_gi(_gi())[1]
        assert i2.text == "Ungrounded take."
        assert i2.quotes == []
        # no explicit `grounded` property and no quotes → False
        assert i2.grounded is False

    def test_speaker_falls_back_to_spoken_by_person(self) -> None:
        gi = {
            "nodes": [
                {"id": "insight:1", "type": "Insight", "properties": {"text": "x"}},
                {"id": "quote:1", "type": "Quote", "properties": {"text": "q"}},  # no speaker_id
            ],
            "edges": [
                {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
                {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:jane-doe"},
            ],
        }
        assert insights_from_gi(gi)[0].quotes[0].speaker == "jane-doe"

    def test_malformed_inputs_return_empty(self) -> None:
        assert insights_from_gi(None) == []
        assert insights_from_gi({"nodes": "nope"}) == []
        assert insights_from_gi({}) == []
