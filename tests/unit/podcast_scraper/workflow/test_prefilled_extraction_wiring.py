"""#643 Phase 3C wiring tests.

Verifies that ``prefilled_extraction`` produced by mega_bundled /
extraction_bundled is plumbed through GIL and KG stages so they skip their own
LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from podcast_scraper.gi.pipeline import build_artifact as gi_build_artifact
from podcast_scraper.kg.pipeline import build_artifact as kg_build_artifact
from podcast_scraper.providers.common.megabundle_parser import MegaBundleResult


def _make_result() -> MegaBundleResult:
    return MegaBundleResult(
        title="Ep",
        summary="Summary paragraph.",
        bullets=["b1", "b2"],
        insights=[
            {"text": "Insight one.", "insight_type": "claim"},
            {"text": "Insight two.", "insight_type": "fact"},
        ],
        topics=["blockchain governance", "roman empire", "algorithmic trading"],
        entities=[
            {"name": "Alice", "kind": "person", "role": "host"},
            {"name": "Acme Corp", "kind": "org", "role": "mentioned"},
        ],
        raw={},
    )


class TestKGPrefilled:
    def test_prefilled_partial_short_circuits_provider(self):
        provider = MagicMock()
        partial = _make_result().to_extraction_partial()

        out = kg_build_artifact(
            "ep1",
            "transcript text",
            podcast_id="feed",
            episode_title="Ep",
            kg_extraction_provider=provider,
            prefilled_partial=partial,
        )

        provider.extract_kg_graph.assert_not_called()
        provider.extract_kg_from_summary_bullets.assert_not_called()

        topic_labels = [n["properties"]["label"] for n in out["nodes"] if n["type"] == "Topic"]
        assert topic_labels == ["blockchain governance", "roman empire", "algorithmic trading"]
        entity_names = [n["properties"]["name"] for n in out["nodes"] if n["type"] == "Entity"]
        assert "Alice" in entity_names
        assert "Acme Corp" in entity_names

    def test_empty_prefilled_falls_back_to_dispatch(self):
        provider = MagicMock()
        provider.extract_kg_graph.return_value = None

        out = kg_build_artifact(
            "ep1",
            "transcript",
            podcast_id="feed",
            episode_title="Ep",
            kg_extraction_provider=provider,
            prefilled_partial={"topics": [], "entities": []},
        )

        assert any(n["type"] == "Episode" for n in out["nodes"])


class TestGIPrefilled:
    def test_prefilled_insights_short_circuits_provider(self):
        provider = MagicMock()
        prefilled = _make_result().to_extraction_partial()["insights"]

        out = gi_build_artifact(
            "ep1",
            "transcript " * 50,
            podcast_id="feed",
            episode_title="Ep",
            insight_provider=provider,
            prefilled_insights=prefilled,
        )

        provider.generate_insights.assert_not_called()
        insights = [n for n in out["nodes"] if n["type"] == "Insight"]
        assert len(insights) == 2
        types = {n["properties"].get("insight_type") for n in insights}
        assert "claim" in types
        assert "fact" in types

    def test_empty_prefilled_falls_back_to_stub(self):
        out = gi_build_artifact(
            "ep1",
            "transcript " * 20,
            podcast_id="feed",
            episode_title="Ep",
            prefilled_insights=[],
        )
        insights = [n for n in out["nodes"] if n["type"] == "Insight"]
        assert len(insights) >= 1

    def test_prefilled_skips_blank_items(self):
        out = gi_build_artifact(
            "ep1",
            "transcript " * 20,
            podcast_id="feed",
            episode_title="Ep",
            prefilled_insights=[
                {"text": "", "insight_type": "claim"},
                {"text": "Valid one.", "insight_type": "fact"},
                {"insight_type": "claim"},
            ],
        )
        insights = [n for n in out["nodes"] if n["type"] == "Insight"]
        texts = [n["properties"].get("text") for n in insights]
        assert "Valid one." in texts
        assert "" not in texts

    def test_prefilled_unknown_type_normalized_to_claim(self):
        out = gi_build_artifact(
            "ep1",
            "transcript " * 20,
            podcast_id="feed",
            episode_title="Ep",
            prefilled_insights=[{"text": "X", "insight_type": "weird"}],
        )
        insights = [n for n in out["nodes"] if n["type"] == "Insight"]
        assert insights[0]["properties"].get("insight_type") == "claim"


class TestMegaBundleHelpers:
    def test_to_extraction_partial_shape(self):
        r = _make_result()
        p = r.to_extraction_partial()
        assert set(p.keys()) == {"insights", "topics", "entities"}
        assert p["insights"][0]["insight_type"] == "claim"
        assert p["topics"] == ["blockchain governance", "roman empire", "algorithmic trading"]
        assert p["entities"][0]["kind"] == "person"
