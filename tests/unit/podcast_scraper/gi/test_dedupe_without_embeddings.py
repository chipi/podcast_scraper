"""Insight dedup must work on the image production actually runs (#1657 follow-up).

THE DEFECT
``dedupe`` was embedding-only: it imported ``sentence_transformers``, and on ImportError logged
a WARNING and returned every input unchanged. That module lives in the ``[ml]`` and ``[search]``
extras — NOT ``[llm]``, which is what the production pipeline image is built with. So in
production the deduper never ran. Every episode of the acceptance run logged::

    insight dedup unavailable (ModuleNotFoundError); keeping all 24

The redundancy it was written to remove was real and measured — 18 gemini episodes emitted 21.6
insights of which only 14.1 were distinct, 35 % restatement, including one claim emitted
verbatim twice. The fix shipped and could not execute.

THE FIX
A lexical tier that needs nothing but the standard library, so it runs everywhere, with the
embedding tier kept as an optional recall upgrade where the module exists.

WHY THE BAR IS AT 0.99
An earlier 0.85 was measured merging a claim with its OPPOSITE (see
``TestItNeverMergesOpposites``). A bag of words cannot see polarity, so any threshold loose
enough to catch a paraphrase is loose enough to merge a statement with its negation. Dropping a
distinct insight destroys knowledge; keeping a duplicate merely repeats it. The threshold goes
where the false positives stop.
"""

from __future__ import annotations

import sys
from typing import List

import pytest

from podcast_scraper.gi.chunked_extraction import (
    DEFAULT_LEXICAL_DEDUPE_THRESHOLD,
    dedupe,
)

pytestmark = [pytest.mark.unit]

PROD_THRESHOLD = 0.72  # gi_insight_dedupe_threshold on provider_chunked_gated_v25


def _dedupe(texts: List[str]) -> List[str]:
    """Exercise the path production takes: embedding tier absent, lexical tier live."""
    return dedupe(texts, threshold=PROD_THRESHOLD)


class TestItRunsWithoutTheEmbeddingModel:
    """The whole point — no sentence-transformers on the [llm] image."""

    def test_sentence_transformers_is_genuinely_absent_here(self) -> None:
        """If this ever fails, this environment stopped resembling the production image and the
        other tests in this class are no longer proving what they claim."""
        assert "sentence_transformers" not in sys.modules
        with pytest.raises(ImportError):
            __import__("sentence_transformers")

    def test_exact_duplicates_are_collapsed_anyway(self) -> None:
        claim = "Paul Tudor Jones believes the greatest challenge will be finding significance."
        assert _dedupe([claim, claim]) == [claim]

    def test_it_does_not_raise_when_the_model_is_missing(self) -> None:
        assert _dedupe(["one distinct claim.", "another distinct claim."]) == [
            "one distinct claim.",
            "another distinct claim.",
        ]

    def test_the_missing_model_is_no_longer_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It was WARNING when it meant "dedup did not happen". It now means "the optional
        higher-recall tier is unavailable", which is a DEBUG-level fact."""
        import logging

        from podcast_scraper.gi import chunked_extraction as ce

        with caplog.at_level(logging.WARNING, logger=ce.logger.name):
            _dedupe(["a claim about one thing.", "a claim about another thing."])
        assert "dedup unavailable" not in "\n".join(r.getMessage() for r in caplog.records)


class TestItCatchesTheShapesSeenInTheWild:
    def test_a_claim_emitted_verbatim_twice(self) -> None:
        c = "Founders should stay technical for as long as they possibly can."
        assert len(_dedupe([c, c, c])) == 1

    def test_punctuation_and_case_only(self) -> None:
        assert (
            len(
                _dedupe(
                    [
                        "AI will reshape hiring, says Horowitz.",
                        "ai will reshape hiring says horowitz",
                    ]
                )
            )
            == 1
        )

    def test_the_same_words_reordered(self) -> None:
        """The sim=0.96 case: identical content, different sentence construction."""
        assert (
            len(
                _dedupe(
                    [
                        "The most important thing for young people to focus on is "
                        "communication skills.",
                        "For young people, the most important thing is to focus on "
                        "communication skills.",
                    ]
                )
            )
            == 1
        )

    def test_the_first_wording_is_the_one_kept(self) -> None:
        """Order matters for provenance: the survivor should be the one extracted first, not an
        arbitrary member of the group."""
        first = "Regulation will slow deployment considerably."
        second = "regulation will slow deployment considerably"
        assert _dedupe([first, second]) == [first]


class TestItNeverMergesOpposites:
    """The false positive that set the threshold. Measured, not hypothesised.

    At 0.85 these merged: six shared tokens, one different, cosine 0.857 — and that one word
    carries the whole claim.
    """

    def test_slow_versus_accelerate(self) -> None:
        texts = [
            "Kalanick believes regulation will slow autonomous vehicle deployment.",
            "Kalanick believes regulation will accelerate autonomous vehicle deployment.",
        ]
        assert _dedupe(texts) == texts

    def test_a_negation_survives(self) -> None:
        texts = [
            "The company will ship the feature this quarter.",
            "The company will not ship the feature this quarter.",
        ]
        assert len(_dedupe(texts)) == 2

    def test_two_claims_about_one_person_both_survive(self) -> None:
        texts = [
            "Ben Horowitz argues founders should stay technical as long as possible.",
            "Ben Horowitz argues hiring executives too early destroys a startup's culture.",
        ]
        assert _dedupe(texts) == texts

    def test_different_numbers_are_different_claims(self) -> None:
        texts = [
            "Revenue grew 40 percent year over year.",
            "Revenue grew 4 percent year over year.",
        ]
        assert len(_dedupe(texts)) == 2


class TestTheContract:
    def test_fewer_than_two_inputs_pass_through(self) -> None:
        assert dedupe([], 0.72) == []
        assert dedupe(["only one"], 0.72) == ["only one"]

    def test_threshold_at_one_disables_near_duplicate_matching(self) -> None:
        """The documented off switch. Exact restatements still collapse — emitting one claim
        twice is never intended, whatever the threshold says."""
        reordered = [
            "The most important thing for young people is communication.",
            "For young people, communication is the most important thing.",
        ]
        assert len(dedupe(reordered, 1.0)) == 2
        same = "One claim."
        assert dedupe([same, same], 1.0) == [same]

    def test_the_default_bar_is_strict(self) -> None:
        assert DEFAULT_LEXICAL_DEDUPE_THRESHOLD >= 0.95, (
            "a looser lexical bar was measured merging a claim with its opposite; see "
            "TestItNeverMergesOpposites"
        )

    def test_empty_and_whitespace_texts_do_not_collapse_together(self) -> None:
        """Two content-free strings share no tokens; they must not be judged the same claim."""
        assert len(dedupe(["...", "???"], PROD_THRESHOLD)) == 2

    def test_order_is_preserved(self) -> None:
        texts = ["first claim here.", "second claim here.", "third claim here."]
        assert _dedupe(texts) == texts


class TestAgainstTheRealCorpus:
    """Every insight in the corpora is distinct. Dropping ANY of them is a false positive.

    These corpora were produced by the [llm] image, so dedup never ran on them — they are
    undeduped ground truth, which is exactly what a false-positive test needs.
    """

    def _insight_texts(self) -> List[List[str]]:
        import glob
        import json
        import os

        out: List[List[str]] = []
        for base in (
            "/Users/claude/podcast-acceptance-corpus/feeds",
            "/Users/claude/acceptance-run-2/feeds",
        ):
            if not os.path.isdir(base):
                continue
            for p in sorted(glob.glob(f"{base}/**/*.gi.json", recursive=True)):
                doc = json.load(open(p))
                texts = [
                    str((n.get("properties") or {}).get("text", "")).strip()
                    for n in doc.get("nodes") or []
                    if n.get("type") == "Insight"
                ]
                texts = [t for t in texts if t]
                if len(texts) >= 2:
                    out.append(texts)
        return out

    def test_no_real_insight_is_removed(self) -> None:
        episodes = self._insight_texts()
        if not episodes:
            pytest.skip("no local corpus available on this machine")
        removed = []
        for texts in episodes:
            kept = _dedupe(texts)
            if len(kept) != len(texts):
                dropped = [t for t in texts if t not in kept]
                removed.extend(dropped)
        assert not removed, f"{len(removed)} distinct insight(s) deleted, e.g. {removed[:2]}"
