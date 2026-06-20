"""#1035 — unit tests for ``kg.ner_prepass.extract_kg_ner_hints``.

Exercises the deterministic NER pre-pass that seeds the KG extraction
prompt with PERSON+ORG candidate spans. Uses a fake spaCy ``Doc``/``Span``
shape so the test runs without loading any real model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from podcast_scraper.kg.ner_prepass import (
    _is_noise_candidate,
    _normalize_candidate_text,
    extract_kg_ner_hints,
)


def _make_fake_nlp(entities: List[tuple[str, str]]):
    """Build a callable that mimics a spaCy ``Language`` returning a ``Doc``
    with the given ``(text, label)`` entity tuples in ``doc.ents``.
    """
    ents = [SimpleNamespace(text=t, label_=l) for t, l in entities]
    doc = SimpleNamespace(ents=ents)
    return lambda text: doc


class TestNormalizeCandidateText:
    def test_strips_whitespace(self) -> None:
        assert _normalize_candidate_text("  Maya  ") == "Maya"

    def test_strips_trailing_punctuation(self) -> None:
        assert _normalize_candidate_text("Strava.") == "Strava"

    def test_collapses_internal_whitespace(self) -> None:
        assert _normalize_candidate_text("Cascadia   Alliance") == "Cascadia Alliance"

    def test_empty_input(self) -> None:
        assert _normalize_candidate_text("") == ""
        assert _normalize_candidate_text("   ") == ""


class TestIsNoiseCandidate:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "a",
            "J.",
            "12",
            "1995",
            "AB",  # two-letter alpha falls under length-<3 rule
        ],
    )
    def test_noise(self, text: str) -> None:
        assert _is_noise_candidate(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Maya",
            "Strava",
            "Cascadia Alliance",
            "USPS",  # 4-letter alpha — kept
            "Byrne Hobart",
        ],
    )
    def test_kept(self, text: str) -> None:
        assert _is_noise_candidate(text) is False


class TestExtractKgNerHints:
    def test_returns_empty_when_nlp_none(self) -> None:
        assert extract_kg_ner_hints("some transcript", nlp=None, max_candidates=10) == []

    def test_returns_empty_when_transcript_empty(self) -> None:
        nlp = _make_fake_nlp([("Maya", "PERSON")])
        assert extract_kg_ner_hints("", nlp=nlp, max_candidates=10) == []
        assert extract_kg_ner_hints("   ", nlp=nlp, max_candidates=10) == []

    def test_keeps_person_and_org_drops_others(self) -> None:
        nlp = _make_fake_nlp(
            [
                ("Maya", "PERSON"),
                ("Strava", "ORG"),
                ("Cascadia Alliance", "ORG"),
                ("Tuesday", "DATE"),
                ("$50", "MONEY"),
                ("Oregon", "GPE"),
                ("United Nations", "NORP"),
            ]
        )
        hints = extract_kg_ner_hints("ignored", nlp=nlp, max_candidates=10)
        labels = {h["label"] for h in hints}
        assert labels == {"PERSON", "ORG"}
        texts = {h["text"] for h in hints}
        assert texts == {"Maya", "Strava", "Cascadia Alliance"}

    def test_dedupes_case_insensitive(self) -> None:
        nlp = _make_fake_nlp(
            [
                ("Maya", "PERSON"),
                ("maya", "PERSON"),
                ("MAYA", "PERSON"),
            ]
        )
        hints = extract_kg_ner_hints("ignored", nlp=nlp, max_candidates=10)
        assert len(hints) == 1
        assert hints[0]["text"] == "Maya"  # first-occurrence spelling wins

    def test_drops_noise(self) -> None:
        nlp = _make_fake_nlp(
            [
                ("J.", "PERSON"),  # initial only — noise
                ("12", "ORG"),  # all-digit — noise
                ("Maya", "PERSON"),
                ("", "PERSON"),  # empty
            ]
        )
        hints = extract_kg_ner_hints("ignored", nlp=nlp, max_candidates=10)
        assert [h["text"] for h in hints] == ["Maya"]

    def test_caps_at_max_candidates(self) -> None:
        ents = [(f"Person{i}", "PERSON") for i in range(50)]
        nlp = _make_fake_nlp(ents)
        hints = extract_kg_ner_hints("ignored", nlp=nlp, max_candidates=10)
        assert len(hints) == 10
        # First-seen order preserved
        assert hints[0]["text"] == "Person0"
        assert hints[-1]["text"] == "Person9"

    def test_returns_empty_when_max_zero_or_negative(self) -> None:
        nlp = _make_fake_nlp([("Maya", "PERSON")])
        assert extract_kg_ner_hints("ignored", nlp=nlp, max_candidates=0) == []
        assert extract_kg_ner_hints("ignored", nlp=nlp, max_candidates=-1) == []

    def test_nlp_exception_returns_empty(self) -> None:
        def broken_nlp(text: str):
            raise RuntimeError("boom")

        assert extract_kg_ner_hints("ignored", nlp=broken_nlp, max_candidates=10) == []

    def test_known_org_injected_when_missing(self) -> None:
        nlp = _make_fake_nlp([("Maya", "PERSON")])
        hints = extract_kg_ner_hints(
            "ignored",
            nlp=nlp,
            max_candidates=10,
            known_org="Singletrack Sessions",
        )
        assert {h["text"] for h in hints} == {"Maya", "Singletrack Sessions"}
        org_entry = next(h for h in hints if h["text"] == "Singletrack Sessions")
        assert org_entry["label"] == "ORG"

    def test_known_org_skipped_when_already_present(self) -> None:
        nlp = _make_fake_nlp([("Singletrack Sessions", "ORG"), ("Maya", "PERSON")])
        hints = extract_kg_ner_hints(
            "ignored",
            nlp=nlp,
            max_candidates=10,
            known_org="Singletrack Sessions",
        )
        # Only one Singletrack Sessions entry (not duplicated by known_org injection)
        assert sum(1 for h in hints if h["text"] == "Singletrack Sessions") == 1

    def test_known_org_skipped_when_at_cap(self) -> None:
        nlp = _make_fake_nlp([("Maya", "PERSON"), ("Strava", "ORG")])
        hints = extract_kg_ner_hints(
            "ignored",
            nlp=nlp,
            max_candidates=2,
            known_org="Show Title",
        )
        assert len(hints) == 2  # cap respected
        assert "Show Title" not in {h["text"] for h in hints}
