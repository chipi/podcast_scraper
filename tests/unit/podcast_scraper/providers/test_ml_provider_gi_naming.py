"""The ML provider's GI and naming methods after their move to ``ml_provider_gi_naming`` (E04 S1):
config refusals, QA / NLI failure fallbacks and the sentence-expanded quote spans."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from podcast_scraper.exceptions import ProviderConfigError
from podcast_scraper.providers.ml.ml_provider_gi_naming import MLProviderGiNamingMethods

pytestmark = pytest.mark.unit


def _methods(**cfg: object) -> MLProviderGiNamingMethods:
    obj = MLProviderGiNamingMethods.__new__(MLProviderGiNamingMethods)
    obj.cfg = SimpleNamespace(auto_speakers=False, **cfg)  # type: ignore[attr-defined]
    obj._spacy_nlp = None  # type: ignore[attr-defined]
    return obj


def test_speaker_detection_without_auto_speakers_is_a_config_error() -> None:
    m = _methods()
    with pytest.raises(ProviderConfigError):
        m.detect_hosts("Feed", None)
    with pytest.raises(ProviderConfigError):
        m.detect_speakers("Episode", None, set())


def test_the_ml_provider_generates_no_insights() -> None:
    assert _methods().generate_insights("text") == []


def test_extract_quotes_expands_each_answer_to_its_sentence() -> None:
    transcript = "First sentence here. The answer is Codex today. Last one."
    at = transcript.index("Codex")
    spans = [
        SimpleNamespace(start=at, end=at + 5, score=0.8),
        SimpleNamespace(start=0, end=len(transcript) + 10, score=0.1),  # past the end: skipped
    ]
    with patch("podcast_scraper.providers.ml.extractive_qa.answer_candidates", return_value=spans):
        quotes = _methods().extract_quotes(transcript, "what tool?")
    assert len(quotes) == 1
    assert "Codex" in quotes[0].text and quotes[0].qa_score == 0.8
    assert quotes[0].text.strip().endswith(".")


def test_extract_quotes_returns_none_when_qa_fails_or_inputs_are_empty() -> None:
    m = _methods()
    assert m.extract_quotes("", "q") == []
    with patch(
        "podcast_scraper.providers.ml.extractive_qa.answer_candidates",
        side_effect=RuntimeError("model missing"),
    ):
        assert m.extract_quotes("Some text.", "q") == []


def test_score_entailment_falls_back_to_zero() -> None:
    m = _methods()
    assert m.score_entailment("", "h") == 0.0
    with patch(
        "podcast_scraper.providers.ml.nli_loader.entailment_score", side_effect=RuntimeError("x")
    ):
        assert m.score_entailment("premise", "hypothesis") == 0.0
    with patch("podcast_scraper.providers.ml.nli_loader.entailment_score", return_value=0.7):
        assert m.score_entailment(" premise ", " hypothesis ") == 0.7
