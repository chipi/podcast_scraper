"""An episode holds a finite amount of knowledge. Saying it twice does not create more.

`dedupe()` existed — and only ever ran inside the CHUNKED extraction path. A normal episode does
not chunk, so on the path production actually takes there was NO deduplication at all, and the
value gate cannot cover for it: it grades each insight in ISOLATION, so two copies of the same
claim both score well and both survive. Redundancy is invisible to a per-item judge.

Measured on 18 episodes with the real corpus: gemini emitted 21.6 surfaceable insights per episode
and only **14.1 were distinct** — 35% redundancy. And not the subtle kind:

    sim 1.00   "Paul Tudor Jones believes that the greatest challenge in the coming years will be
                finding significance..."                              <- emitted VERBATIM twice
    sim 0.96   "...the most important thing for young people to focus on is communication..."
               "...the most important thing for young people is to focus on communication..."

That inflated gemini's apparent lead over qwen from a true 1.46x to a reported 2.06x. The judge
could not see it, and neither could I until the insights were embedded and compared.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from podcast_scraper.gi.pipeline import _dedupe_insight_specs

pytestmark = pytest.mark.integration


def _cfg(threshold: float = 0.75) -> Any:
    class C:
        gi_insight_dedupe_threshold = threshold

    return C()


# The real thing, from the Invest Like the Best episode.
PTJ_A = (
    "Paul Tudor Jones believes that the greatest challenge in the coming years will be finding "
    "significance and purpose in a world where AI has replaced many jobs."
)
PTJ_B = PTJ_A  # emitted verbatim twice
YOUNG_A = (
    "Paul Tudor Jones believes that the most important thing for young people to focus on is "
    "communication, persuasion and the ability to tell a story."
)
YOUNG_B = (
    "Paul Tudor Jones believes that the most important thing for young people is to focus on "
    "communication, persuasion and the ability to tell a story."
)
DISTINCT = (
    "The interconnectedness of global markets means a sell-off in one asset class can force "
    "liquidations in another entirely unrelated one."
)


class TestTheSameClaimTwice:
    def test_a_VERBATIM_duplicate_is_dropped(self) -> None:
        specs: List[Tuple[str, str]] = [(PTJ_A, "claim"), (PTJ_B, "claim")]
        assert _dedupe_insight_specs(specs, _cfg()) == [(PTJ_A, "claim")]

    def test_a_RESTATEMENT_is_dropped(self) -> None:
        """0.96 similar — the same sentence with two words swapped. Two insights, one claim."""
        specs = [(YOUNG_A, "claim"), (YOUNG_B, "claim")]
        assert len(_dedupe_insight_specs(specs, _cfg())) == 1

    def test_GENUINELY_DIFFERENT_knowledge_survives(self) -> None:
        """The rule must not eat the knowledge it exists to count honestly."""
        specs = [(PTJ_A, "claim"), (DISTINCT, "claim")]
        assert len(_dedupe_insight_specs(specs, _cfg())) == 2

    def test_the_FIRST_wording_is_the_one_kept(self) -> None:
        specs = [(YOUNG_A, "claim"), (YOUNG_B, "claim")]
        assert _dedupe_insight_specs(specs, _cfg())[0][0] == YOUNG_A

    def test_the_insight_TYPE_survives_deduplication(self) -> None:
        specs = [(PTJ_A, "stance"), (DISTINCT, "observation")]
        assert [k for _, k in _dedupe_insight_specs(specs, _cfg())] == ["stance", "observation"]


class TestItNeverBreaksTheEpisode:
    def test_a_threshold_of_one_disables_it(self) -> None:
        specs = [(PTJ_A, "claim"), (PTJ_B, "claim")]
        assert _dedupe_insight_specs(specs, _cfg(1.0)) == specs

    def test_a_single_insight_is_untouched(self) -> None:
        assert _dedupe_insight_specs([(PTJ_A, "claim")], _cfg()) == [(PTJ_A, "claim")]

    def test_no_insights_is_not_a_crash(self) -> None:
        assert _dedupe_insight_specs([], _cfg()) == []


def test_dedup_runs_on_the_UNCHUNKED_path_too() -> None:
    """The bug: `dedupe()` lived inside the chunked branch, and a normal episode does not chunk."""
    import inspect

    from podcast_scraper.gi import pipeline as gi_pipeline

    src = inspect.getsource(gi_pipeline._resolve_insight_specs)
    assert "_dedupe_insight_specs(" in src, (
        "insight dedup only runs when the transcript is chunked, so a normal episode ships the "
        "same claim twice — and the value gate, grading each insight alone, cannot see it"
    )
