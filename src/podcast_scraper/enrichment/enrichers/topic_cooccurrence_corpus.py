"""``topic_cooccurrence_corpus`` — corpus-wide Topic-pair counts (deterministic).

Aggregates per-episode Topic-pair occurrences across the corpus. Each
pair carries two signals: **A** = ``episode_count`` (raw frequency — how
many episodes the pair co-occurs in) and **B** = ``lift`` / ``pmi``
(association strength — does the pair co-occur *more than chance*, given
each topic's own frequency). The default ordering is by ``episode_count``;
the Topic card ranks the same pairs both ways so A and B can be compared
live. The output also drives autoresearch hypothesis generation and the
dashboard "co-mentioned topics".

Reuses the same algorithm shape as the existing
``podcast_scraper.kg.corpus.topic_cooccurrence`` (corpus-scope KG
aggregator), but reads directly from the per-episode bundles instead
of from a separate ``loaded`` list — saves one IO pass when the
``topic_cooccurrence`` enricher has already run, and tolerates being
run standalone.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import load_kg, node_label, nodes_of_type
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)


#: Minimum episodes EACH topic of a pair must appear in before the pair is emitted (#1928).
#:
#: Not a floor on how often the PAIR co-occurs — that filter leaves only pairs whose topics are
#: unique to the same episodes, where every association measure saturates. This one asks whether
#: the two topics recur independently, which is what distinguishes an editorial link from one
#: conversation counted twice.
#:
#: 2 is the minimum that means anything ("appears more than once"). Measured on the 1,066-episode
#: corpus: 45,009 pairs -> 1,665, and the survivors are the readable ones (``agentic ai systems``
#: + ``enterprise ai adoption``, ``ai agents`` + ``ai regulation``) instead of seven identical
#: ``active learning`` pairs at lift 533.
_logger = logging.getLogger(__name__)

#: CORRECTION (2026-09-03) to a measurement this module's floors were reasoned from.
#:
#: af6bed32 concluded that label canonicalisation was not worth doing: "normalisation (lowercase ->
#: strip punctuation -> drop stopwords + sort words -> crude singularise) collapses only 0.8% of
#: topics — 72 of 9,263 — so a canonicalisation pass would gain under 1%". That measurement was
#: WRONG, not merely pessimistic: the normaliser stripped punctuation without treating hyphens as
#: word SEPARATORS, so ``us-china-ai-competition`` normalised to one token and could never collide
#: with ``us china ai competition``. Slug-shaped ids are the common case here, so the check was
#: blind to exactly the variants it was looking for.
#:
#: Re-measured in #1933: 66 collision families of the same concept, and merging them raises strong
#: co-occurrence pairs ~8x. So the sparsity this module compensates for is PART labelling artifact,
#: where af6bed32 said it was essentially all genuine corpus diversity. The floors below are still
#: correct — they filter on recurrence, which no relabelling invents — but "93.6% of topics appear
#: in one episode" should be read as an upper bound on true diversity, not a measurement of it.
#: Re-derive these numbers after #1933 lands rather than carrying them forward.

_DEFAULT_MIN_TOPIC_DF = 2


def _read_min_topic_df(config: dict[str, Any]) -> int:
    """Per-topic document-frequency floor (see the constant)."""
    raw = config.get("min_topic_episode_count", _DEFAULT_MIN_TOPIC_DF)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_TOPIC_DF
    return value if value >= 1 else 1


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    pair_labels: dict[tuple[str, str], tuple[str, str]] = {}
    topic_df: dict[str, int] = defaultdict(int)  # episodes each topic appears in
    bundles = all_bundles or []
    for b in bundles:
        kg = load_kg(b)
        topics = nodes_of_type(kg, "Topic")
        ids = sorted({str(t.get("id")) for t in topics if t.get("id")})
        labels = {str(t.get("id")): node_label(t) for t in topics if t.get("id")}
        for tid in ids:
            topic_df[tid] += 1
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b_ = ids[i], ids[j]
                key = (a, b_)
                pair_count[key] += 1
                pair_labels[key] = (labels.get(a, a), labels.get(b_, b_))
    n = len(bundles)

    # #1928 — require BOTH topics to recur before the pair is emitted.
    #
    # ``lift`` rewards rarity by construction, so on a corpus where 93.6% of topics appear once it
    # ranked the rarest pairs highest: measured before this, 99.4% of 45,009 pairs co-occurred in
    # exactly ONE episode, and lift's median, p90 and max were all 1066 — the corpus episode count,
    # which is what ``N / (1 x 1)`` evaluates to. Maximum-possible lift was also modal lift.
    #
    # Filtering on PAIR frequency (the obvious knob, and what the sibling enrichers use) does not
    # fix it here: at ``episode_count >= 2`` only 258 pairs survive, 257 of them at exactly 2, and
    # every one has ``df_a = df_b = 2`` — both topics appear ONLY in those two episodes, so every
    # association measure returns its maximum. NPMI, shrinkage and log-scaling were all tried and
    # all produce the same ordering, because the inputs are indistinguishable.
    #
    # What separates a real association from a coincidence is whether the topics recur
    # INDEPENDENTLY. ``agentic ai systems`` (6 episodes) with ``enterprise ai adoption`` (10) is a
    # genuine editorial link; two topics that each appear only in the same two episodes are one
    # conversation seen twice. So the floor is on per-topic document frequency.
    #
    # NOTE the honest ceiling this exposes: the highest co-occurrence anywhere in the
    # 1,066-episode corpus is THREE episodes. This filter surfaces the real associations that
    # exist; it cannot manufacture ones that do not.
    min_topic_df = _read_min_topic_df(config)
    pairs: list[dict[str, Any]] = []
    below_floor = 0
    for (a, b_), cnt in sorted(pair_count.items(), key=lambda x: (-x[1], x[0])):
        la, lb = pair_labels[(a, b_)]
        da, db = topic_df[a], topic_df[b_]
        if da < min_topic_df or db < min_topic_df:
            below_floor += 1
            continue
        # A = raw ``episode_count`` (how often the pair co-occurs). B = lift/PMI
        # (does the pair co-occur *more than chance*?). lift = P(a,b)/(P(a)·P(b))
        # = cnt·N / (df_a·df_b); >1 ⇒ more than independence predicts. PMI =
        # log2(lift). Both 0.0 when a frequency is missing. Ranking A vs B is a
        # UI concern — we emit the raw signals per pair and let the card sort.
        lift = (cnt * n / (da * db)) if (n and da and db) else 0.0
        pmi = math.log2(lift) if lift > 0 else 0.0
        # #1928 — NPMI: PMI normalised by -log2(P(a,b)), bounded to [-1, 1].
        #
        # Raw lift and PMI are unbounded and reward rarity, so on this corpus they rank the
        # thinnest evidence highest: a pair whose two topics appear ONLY in the same two episodes
        # scores lift 533 (the maximum) while ``agentic ai systems`` + ``enterprise ai adoption``
        # — 6 and 10 episodes, a real editorial link — scores 35. The per-topic floor above
        # removes the worst of that; NPMI makes the remaining values COMPARABLE, because a bounded
        # measure lets the Topic card mix association strength with raw frequency instead of
        # having to choose one. 1.0 still means "these never occur apart", which on thin evidence
        # is exactly the claim a reader should discount, so it is emitted alongside the counts
        # rather than instead of them.
        npmi = (pmi / -math.log2(cnt / n)) if (n and cnt and cnt < n and pmi) else 0.0
        pairs.append(
            {
                "topic_a_id": a,
                "topic_b_id": b_,
                "topic_a_label": la,
                "topic_b_label": lb,
                "episode_count": cnt,
                "topic_a_episode_count": da,
                "topic_b_episode_count": db,
                "lift": round(lift, 4),
                "pmi": round(pmi, 4),
                "npmi": round(npmi, 4),
            }
        )
    # #1208 no-silent-fail contract — an empty pair list has distinct causes.
    partial_reason: str | None = None
    if not bundles:
        partial_reason = "no_bundles"
    elif not pairs:
        partial_reason = "all_pairs_below_min_topic_df" if pair_count else "no_cooccurring_topics"
    if partial_reason is not None:
        _logger.warning(
            "topic_cooccurrence_corpus produced no pairs run_id=%s enricher=%s reason=%s "
            "bundles=%d raw_pairs=%d min_topic_df=%d",
            getattr(ctx, "run_id", ""),
            getattr(ctx, "enricher_id", ""),
            partial_reason,
            len(bundles),
            len(pair_count),
            min_topic_df,
        )

    return {
        "episode_count": n,
        "pairs": pairs,
        # #1928 — say what was withheld, so a short list reads as policy rather than missing data.
        "min_topic_episode_count": min_topic_df,
        "pairs_below_min_topic_df": below_floor,
        "partial_reason": partial_reason,
    }


_enrich_async = sync_enricher(_compute)


class TopicCooccurrenceCorpusEnricher:
    """Corpus-scope Topic-pair co-occurrence aggregator (ranked)."""

    manifest = EnricherManifest(
        id="topic_cooccurrence_corpus",
        version="1.2.0",  # +npmi, +per-topic df floor (#1928)
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="topic_cooccurrence_corpus.json",
        description=(
            "Corpus-wide Topic-pair co-occurrence (episode_count + lift/PMI/NPMI per "
            "pair). Pairs are emitted only when BOTH topics recur (#1928), because lift "
            "rewards rarity and 93.6% of topics appear once."
        ),
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "min_topic_episode_count": {
                    "type": "integer",
                    "minimum": 1,
                    "default": _DEFAULT_MIN_TOPIC_DF,
                    "description": (
                        "Minimum episodes EACH topic of a pair must appear in (#1928). NOT a "
                        "floor on pair frequency — that filter leaves only pairs whose topics "
                        "are unique to the same episodes, where every association measure "
                        "saturates. 1 disables."
                    ),
                },
            },
        },
        expected_duration_s=30,
    )

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Enricher.enrich impl — delegates to the sync body via @sync_enricher."""
        return await _enrich_async(bundle, corpus_root, all_bundles, config, ctx)


__all__ = ["TopicCooccurrenceCorpusEnricher"]
