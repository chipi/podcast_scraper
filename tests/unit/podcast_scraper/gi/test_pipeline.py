#!/usr/bin/env python3
"""Tests for GIL pipeline build_artifact, _resolve_insight_texts, _artifact_from_multi_insight."""

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.gi import validate_artifact
from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.pipeline import (
    _artifact_from_multi_insight,
    _char_range_to_ms,
    _resolve_insight_texts,
    _speaker_id_for_char_range,
    build_artifact,
)


@pytest.mark.unit
class TestGILPipeline:
    """build_artifact produces valid minimal artifact."""

    def test_build_artifact_has_required_keys(self):
        """Output has schema_version, model_version, prompt_version, episode_id, nodes, edges."""
        out = build_artifact("episode:1", "Some transcript.")
        assert out["schema_version"] == "1.0"
        assert out["model_version"] == "stub"
        assert out["prompt_version"] == "v1"
        assert out["episode_id"] == "episode:1"
        assert isinstance(out["nodes"], list)
        assert isinstance(out["edges"], list)

    def test_build_artifact_minimal_valid(self):
        """Output passes minimal validation."""
        out = build_artifact("episode:1", "Hello.")
        validate_artifact(out, strict=False)

    def test_build_artifact_contains_episode_insight_quote(self):
        """Contains at least one Episode, one Insight, one Quote node."""
        out = build_artifact("episode:1", "Text")
        types = {n["type"] for n in out["nodes"]}
        assert "Episode" in types
        assert "Insight" in types
        assert "Quote" in types

    def test_build_artifact_has_supported_by_edge(self):
        """Contains at least one SUPPORTED_BY edge."""
        out = build_artifact("episode:abc", "")
        edge_types = [e["type"] for e in out["edges"]]
        assert "SUPPORTED_BY" in edge_types

    def test_build_artifact_quote_uses_transcript_slice(self):
        """Quote text is slice of transcript when non-empty."""
        transcript = "The quick brown fox."
        out = build_artifact("ep:1", transcript)
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(quote_nodes) == 1
        assert (
            "quick" in quote_nodes[0]["properties"]["text"]
            or "The" in quote_nodes[0]["properties"]["text"]
        )

    def test_build_artifact_optional_metadata(self):
        """Optional podcast_id, episode_title, publish_date applied."""
        out = build_artifact(
            "ep:1",
            "",
            podcast_id="podcast:xyz",
            episode_title="My Episode",
            publish_date="2025-01-15T12:00:00Z",
        )
        ep_nodes = [n for n in out["nodes"] if n["type"] == "Episode"]
        assert len(ep_nodes) == 1
        assert ep_nodes[0]["properties"]["podcast_id"] == "podcast:xyz"
        assert ep_nodes[0]["properties"]["title"] == "My Episode"
        assert ep_nodes[0]["properties"]["publish_date"] == "2025-01-15T12:00:00Z"

    def test_build_artifact_with_cfg_uses_evidence_stack_when_grounded(self):
        """With cfg and mocked grounded quote, artifact has Quote node and grounded=True."""
        cfg = MagicMock()
        cfg.generate_gi = True
        cfg.gi_require_grounding = True
        cfg.gi_insight_model = "google/flan-t5-base"
        cfg.gi_qa_model = "roberta-squad2"
        cfg.gi_nli_model = "nli-deberta-base"
        cfg.extractive_qa_device = None
        cfg.nli_device = None

        one_quote = [
            GroundedQuote(
                char_start=0,
                char_end=6,
                text="Evidence",
                qa_score=0.8,
                nli_score=0.7,
            )
        ]
        with patch("podcast_scraper.gi.grounding.find_grounded_quotes", return_value=one_quote):
            out = build_artifact(
                "ep:1",
                "Evidence here.",
                cfg=cfg,
            )
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(insight_nodes) == 1
        assert insight_nodes[0]["properties"]["grounded"] is True
        assert len(quote_nodes) == 1
        assert quote_nodes[0]["properties"]["text"] == "Evidence"
        assert len(out["edges"]) == 1
        assert out["edges"][0]["type"] == "SUPPORTED_BY"

    def test_build_artifact_with_cfg_no_grounded_quotes_yields_ungrounded_insight(self):
        """With cfg and no grounded quotes, insight has grounded=False and no Quote nodes."""
        cfg = MagicMock()
        cfg.generate_gi = True
        cfg.gi_require_grounding = True
        cfg.gi_insight_model = "google/flan-t5-base"
        cfg.gi_qa_model = "roberta-squad2"
        cfg.gi_nli_model = "nli-deberta-base"
        cfg.extractive_qa_device = None
        cfg.nli_device = None

        with patch("podcast_scraper.gi.grounding.find_grounded_quotes", return_value=[]):
            out = build_artifact("ep:1", "Some transcript.", cfg=cfg)
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(insight_nodes) == 1
        assert insight_nodes[0]["properties"]["grounded"] is False
        assert len(quote_nodes) == 0
        assert len(out["edges"]) == 0

    def test_build_artifact_with_evidence_providers_uses_provider_path(self):
        """With quote_extraction_provider and entailment_provider, uses provider path."""
        cfg = MagicMock()
        cfg.generate_gi = True
        cfg.gi_require_grounding = True
        cfg.gi_insight_model = "stub"
        cfg.gi_qa_model = "roberta-squad2"
        cfg.gi_nli_model = "nli-deberta-base"
        one_quote = [
            GroundedQuote(
                char_start=1,
                char_end=8,
                text="evidence",
                qa_score=0.9,
                nli_score=0.85,
            )
        ]
        mock_qa = MagicMock()
        mock_nli = MagicMock()
        with patch(
            "podcast_scraper.gi.grounding.find_grounded_quotes_via_providers",
            return_value=one_quote,
        ):
            out = build_artifact(
                "ep:1",
                "We have evidence here.",
                cfg=cfg,
                quote_extraction_provider=mock_qa,
                entailment_provider=mock_nli,
            )
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(insight_nodes) == 1
        assert insight_nodes[0]["properties"]["grounded"] is True
        assert len(quote_nodes) == 1
        assert quote_nodes[0]["properties"]["text"] == "evidence"
        assert quote_nodes[0]["properties"]["char_start"] == 1
        assert quote_nodes[0]["properties"]["char_end"] == 8
        assert len(out["edges"]) == 1
        assert out["edges"][0]["type"] == "SUPPORTED_BY"

    def test_build_artifact_with_mock_providers_calls_extract_quotes_and_score_entailment(self):
        """build_artifact calls extract_quotes and score_entailment (mock providers)."""
        from podcast_scraper.gi.grounding import QuoteCandidate

        cfg = MagicMock()
        cfg.generate_gi = True
        cfg.gi_require_grounding = True
        cfg.gi_insight_model = "stub"
        candidate = QuoteCandidate(char_start=2, char_end=11, text="the proof", qa_score=0.9)
        mock_qa = MagicMock()
        mock_qa.extract_quotes = MagicMock(return_value=[candidate])
        mock_nli = MagicMock()
        mock_nli.score_entailment = MagicMock(return_value=0.85)
        out = build_artifact(
            "ep:1",
            "Here is the proof we need.",
            cfg=cfg,
            insight_texts=["An insight."],
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        mock_qa.extract_quotes.assert_called_once()
        mock_nli.score_entailment.assert_called_once()
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert insight_nodes[0]["properties"]["grounded"] is True
        assert len(quote_nodes) == 1
        assert quote_nodes[0]["properties"]["text"] == "the proof"

    def test_resolve_insight_texts_non_empty_stripped_and_capped(self):
        """When insight_texts is non-empty, return stripped and capped by gi_max_insights."""
        cfg = MagicMock()
        cfg.gi_max_insights = 3
        out = _resolve_insight_texts(
            "transcript",
            cfg=cfg,
            insight_texts=["A", " B ", "C", "D", "E"],
        )
        assert out == ["A", "B", "C"]

    def test_resolve_insight_texts_empty_insight_texts_ignored(self):
        """When insight_texts is empty or all blank, fall back to stub."""
        cfg = MagicMock()
        cfg.gi_insight_source = "stub"
        cfg.gi_max_insights = 5
        out = _resolve_insight_texts("t", cfg=cfg, insight_texts=[])
        assert len(out) == 1
        assert "stub" in out[0].lower()
        out2 = _resolve_insight_texts("t", cfg=cfg, insight_texts=["  ", ""])
        assert len(out2) == 1
        assert "stub" in out2[0].lower()

    def test_resolve_insight_texts_provider_called_when_source_provider(self):
        """When gi_insight_source=provider and provider has generate_insights, use it."""
        cfg = MagicMock()
        cfg.gi_insight_source = "provider"
        cfg.gi_max_insights = 5
        provider = MagicMock()
        provider.generate_insights = MagicMock(return_value=["I1", "I2"])
        out = _resolve_insight_texts(
            "transcript here",
            cfg=cfg,
            insight_texts=None,
            insight_provider=provider,
            episode_title="Ep",
        )
        assert out == ["I1", "I2"]
        provider.generate_insights.assert_called_once()
        call_kw = provider.generate_insights.call_args[1]
        assert call_kw["text"] == "transcript here"
        assert call_kw["episode_title"] == "Ep"
        assert call_kw["max_insights"] == 5

    def test_resolve_insight_texts_provider_exception_fallback_to_stub(self):
        """When provider.generate_insights raises, return stub."""
        cfg = MagicMock()
        cfg.gi_insight_source = "provider"
        cfg.gi_max_insights = 5
        provider = MagicMock()
        provider.generate_insights = MagicMock(side_effect=RuntimeError("api error"))
        out = _resolve_insight_texts(
            "t",
            cfg=cfg,
            insight_provider=provider,
        )
        assert len(out) == 1
        assert "stub" in out[0].lower()

    def test_resolve_insight_texts_no_cfg_returns_stub(self):
        """When cfg is None, return single stub."""
        out = _resolve_insight_texts("t", cfg=None)
        assert len(out) == 1
        assert "stub" in out[0].lower()

    def test_artifact_from_multi_insight_multiple_insights_and_quotes(self):
        """Multiple insights and quote lists produce correct nodes/edges."""
        q1 = GroundedQuote(char_start=0, char_end=5, text="One", qa_score=0.9, nli_score=0.8)
        q2 = GroundedQuote(char_start=10, char_end=15, text="Two", qa_score=0.85, nli_score=0.75)
        out = _artifact_from_multi_insight(
            "ep:1",
            ["Insight A", "Insight B"],
            [[q1], [q2]],
            model_version="test",
            prompt_version="v1",
            podcast_id="pod:1",
            episode_title="Title",
            date_str="2025-01-01T00:00:00Z",
            transcript_ref="transcript.txt",
        )
        assert out["episode_id"] == "ep:1"
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(insight_nodes) == 2
        assert insight_nodes[0]["properties"]["text"] == "Insight A"
        assert insight_nodes[0]["properties"]["grounded"] is True
        assert insight_nodes[1]["properties"]["text"] == "Insight B"
        assert insight_nodes[1]["properties"]["grounded"] is True
        assert len(quote_nodes) == 2
        assert len(out["edges"]) == 2

    def test_artifact_from_multi_insight_pads_quote_lists(self):
        """When insight_quotes_list is shorter than insight_texts, pad with []."""
        out = _artifact_from_multi_insight(
            "ep:1",
            ["I1", "I2", "I3"],
            [[]],
            model_version="m",
            prompt_version="v1",
            podcast_id="p",
            episode_title="T",
            date_str="2025-01-01T00:00:00Z",
            transcript_ref="t.txt",
        )
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        assert len(insight_nodes) == 3
        assert all(n["properties"]["grounded"] is False for n in insight_nodes)

    def test_char_range_to_ms_no_segments_returns_zero(self):
        """_char_range_to_ms returns (0, 0) when segments empty or no overlap."""
        assert _char_range_to_ms("hello", 0, 5, []) == (0, 0)
        assert _char_range_to_ms("", 0, 0, [{"start": 0.0, "end": 1.0, "text": "x"}]) == (0, 0)

    def test_speaker_id_for_char_range_prefers_segment_at_char_start(self):
        """_speaker_id_for_char_range uses speaker on segment containing char_start."""
        transcript = "AA BB "
        segments = [
            {"start": 0.0, "end": 1.0, "text": "AA ", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "BB ", "speaker": "B"},
        ]
        assert _speaker_id_for_char_range(transcript, 0, 2, segments) == "A"
        assert _speaker_id_for_char_range(transcript, 3, 5, segments) == "B"

    def test_speaker_id_for_char_range_accepts_speaker_id_key(self):
        """Segment may use speaker_id instead of speaker."""
        transcript = "x"
        segments = [{"start": 0.0, "end": 1.0, "text": "x", "speaker_id": "SPK0"}]
        assert _speaker_id_for_char_range(transcript, 0, 1, segments) == "SPK0"

    def test_artifact_from_multi_insight_sets_speaker_id_from_segments(self):
        """Quote speaker_id is filled when segments carry speaker labels."""
        transcript = "First segment. Second segment."
        segments = [
            {"start": 0.0, "end": 1.5, "text": "First segment. ", "speaker": "Host"},
            {"start": 1.5, "end": 3.0, "text": "Second segment.", "speaker": "Guest"},
        ]
        gq = GroundedQuote(
            char_start=15,
            char_end=32,
            text="Second segment.",
            qa_score=0.9,
            nli_score=0.85,
        )
        out = _artifact_from_multi_insight(
            "ep:1",
            ["Insight"],
            [[gq]],
            model_version="m",
            prompt_version="v1",
            podcast_id="p",
            episode_title="T",
            date_str="2025-01-01T00:00:00Z",
            transcript_ref="t.txt",
            transcript_text=transcript,
            transcript_segments=segments,
        )
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(quote_nodes) == 1
        assert quote_nodes[0]["properties"]["speaker_id"] == "Guest"
        spoken = [e for e in out["edges"] if e["type"] == "SPOKEN_BY"]
        assert len(spoken) == 1
        assert spoken[0]["from"] == quote_nodes[0]["id"]
        spk_nodes = [n for n in out["nodes"] if n["type"] == "Speaker"]
        assert len(spk_nodes) == 1
        assert spk_nodes[0]["properties"]["name"] == "Guest"
        validate_artifact(out, strict=True)

    def test_char_range_to_ms_maps_span_to_timestamps(self):
        """_char_range_to_ms maps quote span to segment start/end in ms (FR2.2)."""
        transcript = "One two three."
        segments = [
            {"start": 0.0, "end": 1.0, "text": "One "},
            {"start": 1.0, "end": 2.0, "text": "two "},
            {"start": 2.0, "end": 3.0, "text": "three."},
        ]
        # Quote [0:4] overlaps first segment -> (0, 1000)
        start_ms, end_ms = _char_range_to_ms(transcript, 0, 4, segments)
        assert start_ms == 0
        assert end_ms == 1000
        # Quote [4:8] overlaps second segment -> (1000, 2000)
        start_ms, end_ms = _char_range_to_ms(transcript, 4, 8, segments)
        assert start_ms == 1000
        assert end_ms == 2000
        # Quote [0:14] spans all segments -> (0, 3000)
        start_ms, end_ms = _char_range_to_ms(transcript, 0, 14, segments)
        assert start_ms == 0
        assert end_ms == 3000

    def test_artifact_from_multi_insight_with_segments_fills_timestamps(self):
        """Quote nodes get timestamp_start_ms/end_ms from transcript_text/segments."""
        transcript = "First segment. Second segment."
        segments = [
            {"start": 0.0, "end": 1.5, "text": "First segment. "},
            {"start": 1.5, "end": 3.0, "text": "Second segment."},
        ]
        gq = GroundedQuote(
            char_start=0, char_end=14, text="First segment.", qa_score=0.9, nli_score=0.85
        )
        out = _artifact_from_multi_insight(
            "ep:1",
            ["Insight"],
            [[gq]],
            model_version="m",
            prompt_version="v1",
            podcast_id="p",
            episode_title="T",
            date_str="2025-01-01T00:00:00Z",
            transcript_ref="t.txt",
            transcript_text=transcript,
            transcript_segments=segments,
        )
        quote_nodes = [n for n in out["nodes"] if n["type"] == "Quote"]
        assert len(quote_nodes) == 1
        assert quote_nodes[0]["properties"]["timestamp_start_ms"] == 0
        assert quote_nodes[0]["properties"]["timestamp_end_ms"] == 1500

    def test_build_artifact_with_insight_texts_produces_multiple_insights(self):
        """build_artifact(..., insight_texts=[...]) produces one Insight per text."""
        cfg = MagicMock()
        cfg.generate_gi = True
        cfg.gi_require_grounding = False  # Skip evidence stack so no model load
        cfg.gi_insight_model = "test"
        cfg.gi_max_insights = 5  # Ensure real int for slice in _resolve_insight_texts
        out = build_artifact(
            "ep:1",
            "Transcript.",
            cfg=cfg,
            insight_texts=["Takeaway one", "Takeaway two"],
        )
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        assert len(insight_nodes) == 2
        assert insight_nodes[0]["properties"]["text"] == "Takeaway one"
        assert insight_nodes[1]["properties"]["text"] == "Takeaway two"
        validate_artifact(out, strict=False)

    def test_build_artifact_with_insight_provider_uses_generate_insights(self):
        """With insight_provider and source=provider, build_artifact calls generate_insights."""
        cfg = MagicMock()
        cfg.generate_gi = True
        cfg.gi_require_grounding = True
        cfg.gi_insight_source = "provider"
        cfg.gi_max_insights = 3
        cfg.gi_insight_model = "test"
        cfg.gi_qa_model = "roberta-squad2"
        cfg.gi_nli_model = "nli-deberta-base"
        cfg.extractive_qa_device = None
        cfg.nli_device = None
        provider = MagicMock()
        provider.generate_insights = MagicMock(return_value=["P1", "P2"])
        with patch("podcast_scraper.gi.grounding.find_grounded_quotes", return_value=[]):
            out = build_artifact(
                "ep:1",
                "Transcript.",
                cfg=cfg,
                insight_provider=provider,
            )
        provider.generate_insights.assert_called_once()
        insight_nodes = [n for n in out["nodes"] if n["type"] == "Insight"]
        assert len(insight_nodes) == 2
        assert insight_nodes[0]["properties"]["text"] == "P1"
        assert insight_nodes[1]["properties"]["text"] == "P2"
