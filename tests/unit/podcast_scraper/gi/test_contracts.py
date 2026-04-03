"""Tests for GIL output contracts (RFC-050 shapes)."""

import pytest

from podcast_scraper.gi.contracts import (
    EvidenceSpan,
    GiArtifact,
    InsightSummary,
    InspectOutput,
    SupportingQuote,
)


@pytest.mark.unit
class TestGILContracts:
    """Output contracts are JSON-serializable and valid."""

    def test_evidence_span_model(self):
        """EvidenceSpan has transcript_ref, char_start, char_end, optional excerpt."""
        span = EvidenceSpan(
            transcript_ref="ep.txt",
            char_start=0,
            char_end=10,
            excerpt="Hello world",
        )
        assert span.transcript_ref == "ep.txt"
        assert span.excerpt == "Hello world"
        d = span.model_dump()
        assert d["char_start"] == 0 and d["char_end"] == 10

    def test_supporting_quote_model(self):
        """SupportingQuote includes evidence span."""
        q = SupportingQuote(
            quote_id="quote:1:0",
            text="Verbatim",
            evidence=EvidenceSpan(transcript_ref="t.txt", char_start=0, char_end=7),
        )
        assert q.quote_id == "quote:1:0"
        assert q.evidence.char_end == 7

    def test_insight_summary_model(self):
        """InsightSummary has supporting_quotes list."""
        ins = InsightSummary(
            insight_id="insight:1:0",
            text="An insight.",
            grounded=True,
            episode_id="ep:1",
            supporting_quotes=[],
        )
        assert ins.grounded is True
        assert ins.model_dump_json()

    def test_inspect_output_json_roundtrip(self):
        """InspectOutput is JSON-serializable and round-trips."""
        out = InspectOutput(
            episode_id="ep:1",
            schema_version="1.0",
            model_version="stub",
            insights=[],
            stats={"grounded_count": 0, "ungrounded_count": 0, "quote_count": 0},
        )
        js = out.model_dump_json()
        out2 = InspectOutput.model_validate_json(js)
        assert out2.episode_id == out.episode_id

    def test_gi_artifact_model_accepts_minimal_gi_json(self):
        """GiArtifact validates top-level gi.json fields (#487)."""
        g = GiArtifact(
            schema_version="1.0",
            model_version="stub",
            prompt_version="v1",
            episode_id="ep:1",
            nodes=[{"id": "n1", "type": "Insight", "properties": {}}],
            edges=[],
        )
        assert g.episode_id == "ep:1"
        assert len(g.nodes) == 1
