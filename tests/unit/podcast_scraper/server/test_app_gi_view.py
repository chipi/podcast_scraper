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


def _ranked_gi() -> dict:
    """Four insights in EXTRACTION order with varying salience/tier/routing_tag (ADR-135/#1191)."""

    def _ins(iid: str, text: str, **props: object) -> dict:
        return {"id": iid, "type": "Insight", "properties": {"text": text, **props}}

    return {
        "nodes": [
            _ins("insight:a", "mid", salience=0.60, rank=2, routing_tag="connect", tier=2),
            _ins("insight:b", "top", salience=0.90, rank=0, routing_tag="surface", tier=3),
            _ins("insight:c", "filler", salience=0.10, rank=3, routing_tag="drop", tier=0),
            _ins("insight:d", "high", salience=0.77, rank=1, routing_tag="surface", tier=2),
        ],
        "edges": [],
    }


class TestInsightRankingAndTagging:
    """ADR-135/#1191 route-and-tag: sort by salience, carry fields, drop `drop`, cap by limit."""

    def test_sorts_by_salience_descending(self) -> None:
        out = insights_from_gi(_ranked_gi())
        # `drop`-tagged filler removed; the rest ordered b(0.90) > d(0.77) > a(0.60).
        assert [i.id for i in out] == ["insight:b", "insight:d", "insight:a"]

    def test_carries_ranking_fields(self) -> None:
        top = insights_from_gi(_ranked_gi())[0]
        assert top.id == "insight:b"
        assert top.salience == 0.90
        assert top.rank == 0
        assert top.routing_tag == "surface"
        assert top.tier == 3

    def test_excludes_drop_tagged(self) -> None:
        out = insights_from_gi(_ranked_gi())
        assert "insight:c" not in {i.id for i in out}
        assert all(i.routing_tag != "drop" for i in out)

    def test_limit_caps_to_top_n_after_sorting(self) -> None:
        out = insights_from_gi(_ranked_gi(), limit=2)
        # top-2 by salience, NOT extraction order
        assert [i.id for i in out] == ["insight:b", "insight:d"]

    def test_limit_none_returns_all_surfaceable(self) -> None:
        assert len(insights_from_gi(_ranked_gi(), limit=None)) == 3

    def test_limit_zero_returns_empty(self) -> None:
        assert insights_from_gi(_ranked_gi(), limit=0) == []

    def test_missing_salience_falls_back_to_extraction_order(self) -> None:
        # Pre-3.1 artifact: no ranking fields → stable extraction order, fields default None.
        out = insights_from_gi(_gi())
        assert [i.id for i in out] == ["insight:1", "insight:2"]
        assert out[0].salience is None and out[0].routing_tag is None and out[0].tier is None

    def test_tie_on_salience_preserves_extraction_order(self) -> None:
        gi = {
            "nodes": [
                {
                    "id": "insight:x",
                    "type": "Insight",
                    "properties": {"text": "x", "salience": 0.5},
                },
                {
                    "id": "insight:y",
                    "type": "Insight",
                    "properties": {"text": "y", "salience": 0.5},
                },
            ],
            "edges": [],
        }
        assert [i.id for i in insights_from_gi(gi)] == ["insight:x", "insight:y"]

    def test_surfaceable_false_still_excluded_alongside_ranking(self) -> None:
        gi = _ranked_gi()
        gi["nodes"][1]["properties"]["surfaceable"] = False  # the top (b) is unsurfaceable
        out = insights_from_gi(gi)
        assert "insight:b" not in {i.id for i in out}
        assert [i.id for i in out] == ["insight:d", "insight:a"]
