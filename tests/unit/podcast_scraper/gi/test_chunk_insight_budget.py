"""The insight ceiling is an EPISODE budget, split across chunks — not handed whole to each.

The bug: ``generate_chunked`` passed the same ``max_insights`` to every chunk. That number comes
from ``duration_scaled_max_insights``, which has *already* scaled it by episode length, and the
chunk count is *also* derived from episode length. Duration was counted twice, so the configured
ceiling was multiplied by the chunk count:

    transcript   chunks   cap/chunk   effective
        52k         2         50          100
       120k         4        125          500
       200k         6        200         1200

Measured on the 2026-08-31 DGX batch: median 79.5 insights/episode against a configured 50,
max 157. ``gi_max_insights: 50`` meant 50 nowhere.

Why it is worth fixing beyond tidiness: insight count drives the per-insight downstream fan-out.
Over 71 episodes, insight_count correlated r=0.58 with extract_quotes CALLS, r=0.60 with quote
INPUT tokens, and r=0.76 with score_entailment calls — and extract_quotes alone is 72% of all
input tokens the pipeline spends. Over-generation is not a tidy-number problem, it is the
multiplier on the most expensive stage.

DESIGN CONSTRAINT (ADR-135:59-61): ``gi_max_insights`` and ``GI_MAX_INSIGHTS_CEILING`` are
"extraction/token-budget safety only, never a corpus cutoff". So this fix makes the BUDGET
correct; it deliberately does NOT add a final trim of the merged list down to ``max_insights``.
Trimming the merged result would be exactly the corpus cutoff the ADR forbids.
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.chunked_extraction import (
    generate_chunked,
    MAX_CHUNKS,
    MIN_CHARS_TO_CHUNK,
    per_chunk_budget,
    plan_chunks,
)


class TestPerChunkBudget:
    @pytest.mark.parametrize(
        "cap,chunks,expected",
        [
            (50, 1, 50),  # unchunked: the episode budget IS the call budget
            (50, 2, 25),
            (50, 4, 13),  # ceil(12.5) — see below
            (125, 4, 32),
            (200, 6, 34),
        ],
    )
    def test_divides_the_episode_budget(self, cap, chunks, expected):
        assert per_chunk_budget(cap, chunks) == expected

    def test_rounds_up_so_the_pieces_still_cover_the_ceiling(self):
        """A floor would quietly UNDER-run the configured budget: 50/4 -> 12*4 = 48.

        The ceiling is a limit, not a target, so a small overshoot is the correct side to err
        on — silently delivering less than the operator configured is the worse failure.
        """
        assert per_chunk_budget(50, 4) * 4 >= 50

    def test_never_returns_zero(self):
        """A chunk allowed zero insights is a pass that cannot contribute anything."""
        assert per_chunk_budget(1, 6) == 1
        assert per_chunk_budget(0, 6) == 1
        assert per_chunk_budget(3, 6) == 1

    def test_degenerate_chunk_counts_do_not_explode(self):
        assert per_chunk_budget(50, 0) == 50
        assert per_chunk_budget(50, -1) == 50


class TestGenerateChunkedPassesTheDividedBudget:
    @staticmethod
    def _recorder():
        seen: list[int] = []

        def gen(*, text, episode_title, max_insights, params, pipeline_metrics):
            seen.append(max_insights)
            return [f"insight {len(seen)}.{i}" for i in range(3)]

        return gen, seen

    def test_each_chunk_gets_a_share_not_the_whole_ceiling(self):
        """THE REGRESSION: every chunk used to receive the full episode ceiling."""
        gen, seen = self._recorder()
        text = "line one. line two.\n" * 8000  # comfortably over MIN_CHARS_TO_CHUNK

        generate_chunked(
            gen,
            text,
            episode_title="t",
            max_insights=50,
            chunk_chars=30_000,
            dedupe_threshold=0.75,
        )

        assert len(seen) > 1, "fixture must actually chunk or it proves nothing"
        assert all(c < 50 for c in seen), f"a chunk got the whole episode ceiling: {seen}"
        assert sum(seen) >= 50, "the pieces must still cover the configured budget"

    def test_the_unchunked_path_still_gets_the_full_ceiling(self):
        """A single pass over the whole transcript SHOULD get the whole budget."""
        gen, seen = self._recorder()
        generate_chunked(
            gen,
            "short transcript",
            episode_title="t",
            max_insights=50,
            chunk_chars=30_000,
            dedupe_threshold=0.75,
        )
        assert seen == [50]

    def test_effective_ceiling_no_longer_scales_with_chunk_count(self):
        """Duration was counted twice. Now the episode budget is invariant to chunking.

        Parameterised over the real transcript sizes from the batch, this is the property that
        was violated: the SUM of the per-chunk budgets must stay near the episode ceiling
        instead of growing linearly with the number of passes.
        """
        for chars in (60_000, 120_000, 200_000):
            n = plan_chunks("x" * chars, 30_000)
            cap = 50
            total = per_chunk_budget(cap, n) * n
            assert (
                cap <= total < cap + n
            ), f"{chars} chars -> {n} chunks: effective ceiling {total} should be ~{cap}"
        assert plan_chunks("x" * 10_000_000, 30_000) == MAX_CHUNKS
        assert plan_chunks("x" * (MIN_CHARS_TO_CHUNK - 1), 30_000) == 1


class TestMergedResultIsNotTruncated:
    """ADR-135:59-61 — the ceiling is token-budget safety, NEVER a corpus cutoff.

    Pinned as a test because the obvious "tidy" follow-up to the fix above is to trim the merged
    list to ``max_insights``, which would silently turn a budget into a cutoff and drop
    gated-good insights the ranking layer is supposed to order, not discard.
    """

    def test_merged_output_may_exceed_the_episode_ceiling(self):
        # 50k chars over 30k chunk_chars -> exactly 2 passes; cap 5 -> ceil(5/2) = 3 each,
        # so 6 distinct insights against an episode ceiling of 5. The overshoot must survive.
        calls = {"n": 0}
        # Lexically unrelated sentences. Templated strings differing only by an index are ~90%
        # identical and the dedupe pass eats them, which made the first version of this test
        # fail for a reason that had nothing to do with truncation.
        BANK = [
            "Interest rates fell sharply after the central bank meeting.",
            "The rover discovered frozen water beneath the southern crater.",
            "Union negotiators rejected the proposed shift schedule.",
            "Coffee exports from Colombia doubled in the second quarter.",
            "A new compiler optimisation removed most bounds checks.",
            "The museum returned seventeen artefacts to Benin.",
        ]

        def gen(*, text, episode_title, max_insights, params, pipeline_metrics):
            i = calls["n"] * max_insights
            calls["n"] += 1
            return BANK[i : i + max_insights]

        out = generate_chunked(
            gen,
            "line one. line two.\n" * 2500,
            episode_title="t",
            max_insights=5,
            chunk_chars=30_000,
            dedupe_threshold=0.99,  # near-1.0 so only near-identical text is dropped
        )
        assert calls["n"] == 2, f"fixture must produce 2 chunks, got {calls['n']}"
        assert len(out) > 5, (
            f"merged list ({len(out)}) was trimmed to the ceiling (5) — ADR-135 forbids "
            "treating the budget as a corpus cutoff"
        )


class TestFailureModesStillHold:
    def test_a_failing_chunk_does_not_cost_the_episode(self):
        calls = {"n": 0}

        def gen(*, text, episode_title, max_insights, params, pipeline_metrics):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("chunk exploded")
            return ["a good insight"]

        out = generate_chunked(
            gen,
            "line one. line two.\n" * 8000,
            episode_title="t",
            max_insights=50,
            chunk_chars=30_000,
            dedupe_threshold=0.75,
        )
        assert out, "one bad chunk must not lose the whole episode"

    def test_all_chunks_empty_falls_back_to_a_single_full_pass(self):
        seen: list[int] = []
        chunks = plan_chunks("line one. line two.\n" * 8000, 30_000)

        def gen(*, text, episode_title, max_insights, params, pipeline_metrics):
            seen.append(max_insights)
            # EVERY chunk must come back empty, or `merged` is non-empty and the fallback
            # never fires — which is how the first version of this test fooled itself.
            return [] if len(seen) <= chunks else ["recovered"]

        out = generate_chunked(
            gen,
            "line one. line two.\n" * 8000,
            episode_title="t",
            max_insights=50,
            chunk_chars=30_000,
            dedupe_threshold=0.75,
        )
        assert out == ["recovered"]
        assert seen[-1] == 50, "the whole-transcript fallback pass gets the whole budget"
