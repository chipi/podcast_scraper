#!/usr/bin/env python3
"""Tests for GIL grounding (find_grounded_quotes with QA + NLI)."""

from unittest.mock import patch

import pytest

from podcast_scraper.gi.grounding import (
    find_grounded_quotes,
    find_grounded_quotes_via_providers,
    GroundedQuote,
    NLI_ENTAILMENT_MIN,
    QA_SCORE_MIN,
    QuoteCandidate,
    resolve_llm_quote_span,
)


@pytest.mark.unit
class TestFindGroundedQuotes:
    """find_grounded_quotes returns quotes that pass QA and NLI thresholds."""

    def test_empty_transcript_returns_empty(self):
        """Empty transcript returns no quotes."""
        result = find_grounded_quotes(
            transcript="",
            insight_text="Some insight.",
            qa_model_id="roberta-squad2",
            nli_model_id="nli-deberta-base",
        )
        assert result == []

    def test_empty_insight_returns_empty(self):
        """Empty insight text returns no quotes."""
        result = find_grounded_quotes(
            transcript="Some context here.",
            insight_text="",
            qa_model_id="roberta-squad2",
            nli_model_id="nli-deberta-base",
        )
        assert result == []

    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_qa_below_threshold_returns_empty(self, mock_qa_answer):
        """When QA score is below threshold, returns empty (NLI not called)."""
        mock_qa_answer.return_value = type(
            "QASpan", (), {"start": 0, "end": 10, "answer": "foo", "score": 0.1}
        )()
        result = find_grounded_quotes(
            transcript="Evidence here in the transcript.",
            insight_text="An insight.",
            qa_model_id="roberta-squad2",
            nli_model_id="nli-deberta-base",
            qa_score_min=QA_SCORE_MIN,
        )
        assert result == []
        mock_qa_answer.assert_called_once()

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_nli_below_threshold_returns_empty(self, mock_qa_answer, mock_nli_score):
        """When NLI score is below threshold, returns empty."""
        mock_qa_answer.return_value = type(
            "QASpan",
            (),
            {"start": 0, "end": 10, "answer": "Evidence here", "score": 0.9},
        )()
        mock_nli_score.return_value = 0.2
        result = find_grounded_quotes(
            transcript="Evidence here in the transcript.",
            insight_text="An insight.",
            qa_model_id="roberta-squad2",
            nli_model_id="nli-deberta-base",
            nli_entailment_min=NLI_ENTAILMENT_MIN,
        )
        assert result == []
        mock_nli_score.assert_called_once()

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_both_pass_returns_one_grounded_quote(self, mock_qa_answer, mock_nli_score):
        """When QA and NLI both pass, returns one GroundedQuote."""
        mock_qa_answer.return_value = type(
            "QASpan",
            (),
            {"start": 5, "end": 18, "answer": "here in the", "score": 0.85},
        )()
        mock_nli_score.return_value = 0.7
        transcript = "Some evidence here in the transcript."
        result = find_grounded_quotes(
            transcript=transcript,
            insight_text="An insight.",
            qa_model_id="roberta-squad2",
            nli_model_id="nli-deberta-base",
        )
        assert len(result) == 1
        assert isinstance(result[0], GroundedQuote)
        assert result[0].char_start == 5
        assert result[0].char_end == 18
        assert result[0].text == transcript[5:18]
        assert result[0].qa_score == 0.85
        assert result[0].nli_score == 0.7


@pytest.mark.unit
class TestFindGroundedQuotesViaProviders:
    """find_grounded_quotes_via_providers uses provider extract_quotes + score_entailment."""

    def test_empty_transcript_returns_empty(self):
        """Empty transcript returns no quotes."""
        mock_qa = type("P", (), {"extract_quotes": lambda **kw: []})()
        mock_nli = type("P", (), {"score_entailment": lambda **kw: 0.8})()
        result = find_grounded_quotes_via_providers(
            transcript="",
            insight_text="Insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        assert result == []

    def test_missing_extract_quotes_returns_empty(self):
        """When quote_extraction_provider has no extract_quotes, returns empty."""
        mock_qa = type("P", (), {})()
        mock_nli = type("P", (), {"score_entailment": lambda **kw: 0.8})()
        result = find_grounded_quotes_via_providers(
            transcript="Some text.",
            insight_text="Insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        assert result == []

    def test_missing_score_entailment_returns_empty(self):
        """When entailment_provider has no score_entailment, returns empty."""
        candidate = QuoteCandidate(char_start=0, char_end=4, text="text", qa_score=0.9)
        mock_qa = type(
            "P",
            (),
            {"extract_quotes": lambda **kw: [candidate]},
        )()
        mock_nli = type("P", (), {})()
        result = find_grounded_quotes_via_providers(
            transcript="Some text.",
            insight_text="Insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        assert result == []

    def test_extract_quotes_returns_non_list_returns_empty(self):
        """When extract_quotes returns non-list (e.g. dict or None), returns empty."""
        mock_qa = type("P", (), {"extract_quotes": lambda **kw: {"bad": "data"}})()
        mock_nli = type("P", (), {"score_entailment": lambda **kw: 0.8})()
        result = find_grounded_quotes_via_providers(
            transcript="Some text.",
            insight_text="Insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        assert result == []

    def test_provider_path_returns_grounded_quote(self):
        """When providers return one candidate and NLI passes, returns one GroundedQuote."""
        candidate = QuoteCandidate(char_start=2, char_end=9, text="evidence", qa_score=0.9)
        mock_qa = type(
            "P",
            (),
            {"extract_quotes": lambda *args, **kw: [candidate]},
        )()
        mock_nli = type(
            "P",
            (),
            {"score_entailment": lambda *args, **kw: 0.85},
        )()
        result = find_grounded_quotes_via_providers(
            transcript="We have evidence here.",
            insight_text="An insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        assert len(result) == 1
        assert isinstance(result[0], GroundedQuote)
        assert result[0].char_start == 2
        assert result[0].char_end == 9
        assert result[0].text == "evidence"
        assert result[0].qa_score == 0.9
        assert result[0].nli_score == 0.85

    def test_duck_typed_candidate_accepted(self):
        """Candidates with char_start, char_end, text, qa_score (no QuoteCandidate) work."""
        candidate = type("C", (), {"char_start": 0, "char_end": 4, "text": "ab", "qa_score": 0.8})()
        mock_qa = type(
            "P",
            (),
            {"extract_quotes": lambda *args, **kw: [candidate]},
        )()
        mock_nli = type(
            "P",
            (),
            {"score_entailment": lambda *args, **kw: 0.7},
        )()
        result = find_grounded_quotes_via_providers(
            transcript="ab",
            insight_text="I.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
        )
        assert len(result) == 1
        assert result[0].char_start == 0 and result[0].char_end == 4
        assert result[0].text == "ab" and result[0].nli_score == 0.7

    def test_qa_below_threshold_filtered_out(self):
        """Candidates with qa_score below threshold are not sent to NLI."""
        candidate = QuoteCandidate(char_start=0, char_end=5, text="low", qa_score=0.1)
        mock_qa = type(
            "P",
            (),
            {"extract_quotes": lambda *args, **kw: [candidate]},
        )()
        nli_calls = []

        def record_nli(*args, **kw):
            nli_calls.append((kw.get("premise"), kw.get("hypothesis")))
            return 0.9

        mock_nli = type("P", (), {"score_entailment": record_nli})()
        result = find_grounded_quotes_via_providers(
            transcript="low score",
            insight_text="I.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
            qa_score_min=0.5,
        )
        assert result == []
        assert len(nli_calls) == 0

    def test_pipeline_metrics_incremented_when_provided(self):
        """When pipeline_metrics is provided, evidence call counters are incremented."""
        candidate = QuoteCandidate(char_start=0, char_end=6, text="quote", qa_score=0.9)
        mock_qa = type(
            "P",
            (),
            {"extract_quotes": lambda *args, **kw: [candidate]},
        )()
        mock_nli = type("P", (), {"score_entailment": lambda *args, **kw: 0.85})()
        metrics = type(
            "M",
            (),
            {
                "gi_evidence_extract_quotes_calls": 0,
                "gi_evidence_nli_candidates_queued": 0,
                "gi_evidence_score_entailment_calls": 0,
            },
        )()
        result = find_grounded_quotes_via_providers(
            transcript="quote here",
            insight_text="Insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
            pipeline_metrics=metrics,
        )
        assert len(result) == 1
        assert metrics.gi_evidence_extract_quotes_calls == 1
        assert metrics.gi_evidence_nli_candidates_queued == 1
        assert metrics.gi_evidence_score_entailment_calls == 1

    def test_extract_retries_second_call_returns_candidate(self):
        """When first extract returns [], retry can return a candidate."""
        candidate = QuoteCandidate(char_start=0, char_end=2, text="ab", qa_score=0.9)
        calls = []

        def extract_twice(*args, **kwargs):
            calls.append(kwargs.get("insight_text", ""))
            if len(calls) == 1:
                return []
            return [candidate]

        mock_qa = type("P", (), {"extract_quotes": extract_twice})()
        mock_nli = type("P", (), {"score_entailment": lambda *a, **k: 0.8})()
        result = find_grounded_quotes_via_providers(
            transcript="ab",
            insight_text="Insight.",
            quote_extraction_provider=mock_qa,
            entailment_provider=mock_nli,
            extract_retries=1,
        )
        assert len(result) == 1
        assert len(calls) == 2
        assert "Reminder:" in calls[1]


@pytest.mark.unit
class TestResolveLlmQuoteSpan:
    """resolve_llm_quote_span maps model quote_text to transcript offsets."""

    def test_exact_substring(self):
        t = "Hello world. More text."
        r = resolve_llm_quote_span(t, "world")
        assert r is not None
        start, end, verbatim = r
        assert verbatim == "world"
        assert t[start:end] == "world"

    def test_whitespace_flexible_full_quote(self):
        t = "Prefix And in\n2003,\tthey won."
        q = "And in 2003, they won."
        r = resolve_llm_quote_span(t, q)
        assert r is not None
        _, _, verbatim = r
        assert "2003" in verbatim
        assert "they won" in verbatim

    def test_drop_leading_words_when_transcript_differs(self):
        t = "We went. And in 2003, they were able to finish the land deal quickly."
        q = "In 2003, they were able to finish the land deal quickly."
        r = resolve_llm_quote_span(t, q)
        assert r is not None
        start, end, verbatim = r
        assert verbatim == t[start:end]
        assert "2003" in verbatim
        assert verbatim.startswith("in 2003") or verbatim.startswith("In 2003")

    def test_longest_suffix_when_prefix_paraphrased(self):
        t = "They'd be luxury units with market rate. But also subsidized units here."
        q = "Most would be luxury units with market rate. But also subsidized units here."
        r = resolve_llm_quote_span(t, q)
        assert r is not None
        _, _, verbatim = r
        assert "luxury units with market rate" in verbatim
        assert "subsidized" in verbatim

    def test_no_match_returns_none(self):
        assert resolve_llm_quote_span("aaa bbb", "ccc ddd eee fff") is None

    def test_empty_returns_none(self):
        assert resolve_llm_quote_span("", "hello") is None
        assert resolve_llm_quote_span("hello", "") is None

    def test_duplicate_phrase_prefers_earlier_occurrence(self):
        t = "First alpha beta gamma stop. Middle alpha beta gamma stop."
        q = "alpha beta gamma"
        r = resolve_llm_quote_span(t, q)
        assert r is not None
        assert r[0] < t.index("Middle")

    def test_apostrophe_unicode_vs_ascii(self):
        t = "She said it\u2019s undeniable here today."
        q = "it's undeniable here"
        r = resolve_llm_quote_span(t, q)
        assert r is not None
        assert "undeniable" in r[2]
