"""Insight dedup must work where ``sentence-transformers`` is absent (#1657 follow-up).

THE DEFECT
``dedupe`` was embedding-only: it imported ``sentence_transformers``, and on ImportError logged
a WARNING and returned every input unchanged. So in any environment without that optional
dependency, no dedup happened at all::

    insight dedup unavailable (ModuleNotFoundError); keeping all 24

The redundancy it was written to remove was real and measured — 18 gemini episodes emitted 21.6
insights of which only 14.1 were distinct, 35 % restatement, including one claim emitted
verbatim twice.

WHICH ENVIRONMENTS — corrected 2026-08-16, after this file first said the wrong thing
This docstring originally claimed ``sentence-transformers`` is not in ``[llm]`` and therefore
"in production the deduper never ran". FALSE. ``docker/pipeline/Dockerfile`` builds the llm
variant with ``.[llm,search,sentry,langfuse]``; ``[search]`` pins
``sentence-transformers>=5.6.0``; the runtime stage copies the whole site-packages tree. The
production image has the embedding tier.

The log line above is real, but it came from running FROM SOURCE on a macOS x86_64 dev box,
where torch and lancedb publish no wheels and the ML extras cannot install. One machine's truth
was written up as production's.

So the defect is narrower than first stated — and still worth fixing: a dedup feature that
silently no-ops wherever an optional heavy dependency is missing is a feature with a hole in it,
and the hole is invisible (a debug line) rather than loud.

THE FIX
A lexical tier that needs nothing but the standard library, so it runs everywhere — including
that dev box and any air-gapped or minimal deployment — with the embedding tier kept as an
optional recall upgrade where the module exists. A floor, not a replacement.

WHY THE BAR IS AT 0.90
It is bounded on both sides by measurement over all 14 episodes of the 2026-08-16 acceptance
corpus (14,539 surviving pairs):

  - Not lower, because at 0.85 a claim merged with its OPPOSITE — cosine 0.857, one differing
    word carrying the whole claim (``TestItNeverMergesOpposites``). A bag of words cannot see
    polarity.
  - Not higher, because the lowest REAL duplicate measured scores 0.9375 ("was right" vs "was
    justified"). The original 0.99 was set on short episodes where no pair exceeded 0.60, and
    it missed all 5 duplicates the long episodes produced. That is what reopened #27.

Between those populations sits an empty gap — nothing scored between 0.7108 and 0.9375 — so
0.90 is not balanced on a single observation.

Duplicates DO continue below the gap (a full paraphrase at 0.6390), and this method cannot
reach them without sweeping in the 14,533 mostly-distinct pairs below 0.70. That is the honest
ceiling of a bag-of-words method and the embedding tier's job. Dropping a distinct insight
destroys knowledge; keeping a duplicate merely repeats it, so the bar stays where the false
positives stop.
"""

from __future__ import annotations

import sys
from typing import List

import pytest

from podcast_scraper.gi.chunked_extraction import (
    dedupe,
    DEFAULT_LEXICAL_DEDUPE_THRESHOLD,
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

    def test_the_default_bar_sits_in_the_measured_gap(self) -> None:
        """The bar is bounded on BOTH sides by measurement, not by caution in one direction.

        Lower bound 0.86: the synthetic antonym pair scores 0.857 and must stay unmerged, since
        a bag of words cannot see polarity (see TestItNeverMergesOpposites).

        Upper bound 0.93: the lowest REAL duplicate measured across the 14-episode acceptance
        corpus scores 0.9375 ("was right" vs "was justified"). A bar above that misses it, which
        is the regression that reopened #27 after the original 0.99 was set on short episodes
        where no pair scored above 0.60.

        Nothing was observed between 0.7108 and 0.9375, so the gap is wide and this is not a
        threshold balanced on one data point.
        """
        assert 0.86 <= DEFAULT_LEXICAL_DEDUPE_THRESHOLD <= 0.93, (
            f"threshold {DEFAULT_LEXICAL_DEDUPE_THRESHOLD} is outside the measured gap "
            f"[0.86, 0.93]: below it merges a claim with its opposite (0.857), above it misses "
            f"a real duplicate (0.9375)"
        )

    def test_the_verb_swap_duplicates_from_the_acceptance_run_are_caught(self) -> None:
        """The five real duplicates that reopened #27. Verbatim from the corpus.

        Four are pure verb swaps on an otherwise identical sentence; the fifth swaps one
        adjective. All five survived the 0.99 bar and are the reason it moved.
        """
        says_argues = [
            "Ryan Greenblatt argues that ML research is less deep than math, so there is less "
            "reliance on individual deep experts combining insights, making it more amenable "
            "to AI automation.",
            "Ryan Greenblatt says that ML research is less deep than math, so there is less "
            "reliance on individual deep experts combining insights, making it more amenable "
            "to AI automation.",
        ]
        assert len(_dedupe(says_argues)) == 1, "a verb swap is not a new claim"

        right_justified = [
            "Charity Majors argues that the industry's past skepticism about AI was right, but "
            "that the current evidence from harnesses and tooling shows the direction is clear.",
            "Charity Majors argues that the industry's past skepticism about AI was justified, "
            "but that the current evidence from harnesses and tooling shows the direction is "
            "clear.",
        ]
        assert (
            len(_dedupe(right_justified)) == 1
        ), "cos 0.9375 — the lowest real duplicate measured, and the one the 0.99 bar missed"

    def test_the_paraphrase_tier_is_still_missed_and_that_is_known(self) -> None:
        """Honesty test: the lexical ceiling is documented, so it must be OBSERVED, not claimed.

        This pair IS a duplicate (cos 0.6390, from the same acceptance corpus) and this method
        cannot catch it — reaching that low would sweep in 14,533 mostly-distinct pairs. If a
        future change makes this pass, the threshold moved into dangerous territory and the
        docstring on DEFAULT_LEXICAL_DEDUPE_THRESHOLD is now wrong.
        """
        paraphrase = [
            "Charity Majors argues that for a company's own code, telemetry should be a product "
            "decision, and the value of rich data grows combinatorially, not linearly or "
            "exponentially, because adding a new field makes every other field more valuable.",
            "Charity Majors says that the value of rich telemetry data goes up combinatorially, "
            "not linearly or even exponentially, because adding a new bit of data to a wide "
            "event or trace makes that new data more valuable.",
        ]
        assert len(_dedupe(paraphrase)) == 2, (
            "if this now merges, the bar dropped into the distinct-claim population — verify "
            "against the full corpus before accepting it"
        )

    def test_empty_and_whitespace_texts_do_not_collapse_together(self) -> None:
        """Two content-free strings share no tokens; they must not be judged the same claim."""
        assert len(dedupe(["...", "???"], PROD_THRESHOLD)) == 2

    def test_order_is_preserved(self) -> None:
        texts = ["first claim here.", "second claim here.", "third claim here."]
        assert _dedupe(texts) == texts


class TestAgainstTheRealCorpus:
    """The corpora are undeduped ground truth: anything removed must be a REAL duplicate.

    These corpora were produced by a FROM-SOURCE run on a macOS x86_64 box, where torch and
    lancedb publish no wheels, so ``sentence-transformers`` could not be installed and no
    embedding dedup ran on them. They are undeduped ground truth — exactly what a
    false-positive test needs.

    (This said "produced by the [llm] image" until 2026-08-16. Wrong: that image installs
    ``[search]``, which carries ``sentence-transformers``. The undeduped property these tests
    depend on is real, but it comes from the dev box's missing wheels, not from the image's
    extras. Stated correctly here because the property is load-bearing — if these corpora HAD
    been deduped already, every assertion below would be measuring nothing.)

    This class used to assert that NOTHING was ever removed, on the stated premise that "every
    insight in the corpora is distinct". Measuring all 14,539 pairs falsified that premise: the
    corpus holds 5 genuine duplicates (four verb swaps — "says"/"argues"/"claims" — and one
    adjective swap), which is what reopened #27. The old assertion passed only because the 0.99
    bar was too high to catch them; it was measuring the threshold's blindness, not the
    corpus's cleanliness.

    So the guarantee is now stated precisely instead of absolutely: removals are allowed, but
    ONLY of pairs that a human read and confirmed. Any removal beyond that list is a false
    positive and fails — which is the property that actually protects knowledge.
    """

    #: Content-word fragments unique to the 5 confirmed duplicate pairs. Anything else removed
    #: is unreviewed and must fail.
    _CONFIRMED_DUPLICATE_MARKERS = (
        "past skepticism about ai was justified",
        "ml research is less deep than math",
        "babies first new theory",
        "ai r&d tasks can be made verifiable",
        "ai r&d has properties similar to mathematics",
    )

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

    def test_only_confirmed_duplicates_are_removed(self) -> None:
        """No DISTINCT insight is deleted. Removals are allowed only for reviewed duplicates."""
        episodes = self._insight_texts()
        if not episodes:
            pytest.skip("no local corpus available on this machine")

        unreviewed = []
        for texts in episodes:
            kept = _dedupe(texts)
            for dropped in (t for t in texts if t not in kept):
                low = dropped.lower()
                if not any(m in low for m in self._CONFIRMED_DUPLICATE_MARKERS):
                    unreviewed.append(dropped)

        assert not unreviewed, (
            f"{len(unreviewed)} insight(s) removed that nobody confirmed are duplicates — "
            f"a false positive destroys knowledge, so read these and either add them to "
            f"_CONFIRMED_DUPLICATE_MARKERS or raise the threshold: {unreviewed[:2]}"
        )

    def test_the_known_duplicates_are_actually_caught(self) -> None:
        """The other direction: the bar must still be low enough to earn its keep.

        Without this, raising the threshold back to 0.99 would make the test above pass
        trivially — zero removals, zero unreviewed — and silently restore the #27 regression.
        """
        episodes = self._insight_texts()
        if not episodes:
            pytest.skip("no local corpus available on this machine")

        removed_count = sum(len(texts) - len(_dedupe(texts)) for texts in episodes)

        assert removed_count >= 5, (
            f"only {removed_count} duplicate(s) removed across the corpus; 5 were measured and "
            f"confirmed by reading them. A higher threshold misses real duplicates — that is "
            f"the regression #27 was reopened for."
        )
