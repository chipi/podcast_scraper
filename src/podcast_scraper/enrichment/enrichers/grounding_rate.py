"""``grounding_rate`` — per-EPISODE share of insights backed by a supporting quote.

For each episode, computes:

* ``total_insights`` — every Insight node in the episode's GI artifact
* ``grounded_insights`` — those with ``properties.grounded == true``
* ``rate`` — grounded / total

Answers "which episodes produced insights we could not ground", i.e. extraction quality. Reads
``*.gi.json`` only; deterministic.

WHY THIS IS NOT PER-PERSON ANY MORE (#1927)
-------------------------------------------
It used to attribute insights to the Person who spoke their supporting quote, via
SPOKEN_BY → SUPPORTED_BY. On the 1,066-episode corpus that returned **exactly 1.0 for all 689
people** — a "credibility" signal on which everyone scored perfectly.

The cause is structural, not a traversal bug. Measured across 5,111 insights:

    grounded=False, speaker=False : 101
    grounded=True,  speaker=True  : 5010
    grounded=False, speaker=True  :   0      <- never happens

An insight is grounded *exactly when* a supporting quote was found, and the quote is what carries
the speaker. No quote means no grounding AND no speaker, so the ungrounded insights belong to
nobody and any speaker-keyed denominator reproduces ``total == grounded``. Re-attributing via
``Insight.speaker`` was tried and gives the identical result (43/43 speakers at 1.0). **You cannot
attribute an unattributed insight** — the per-person question is unanswerable, not under-computed.

The episode is the smallest scope where both terms are observable, and there the signal is real:
measured range 0.800–1.000, 20 of 77 episodes below perfect, worst at 36/45 grounded. That is a
usable corpus-QA signal about which episodes extracted badly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

from podcast_scraper.enrichment.enrichers._loaders import (
    load_gi,
    nodes_of_type,
)
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    episodes_out: list[dict[str, Any]] = []
    corpus_total = 0
    corpus_grounded = 0

    bundles = all_bundles or []
    for b in bundles:
        gi = load_gi(b)
        insights = nodes_of_type(gi, "Insight")
        total = len(insights)
        if not total:
            continue
        grounded = sum(
            1 for node in insights if bool((node.get("properties") or {}).get("grounded", False))
        )
        corpus_total += total
        corpus_grounded += grounded
        episodes_out.append(
            {
                "episode_id": b.episode_id,
                "total_insights": total,
                "grounded_insights": grounded,
                "ungrounded_insights": total - grounded,
                "rate": round(grounded / total, 4),
            }
        )

    # Worst first — the point of the signal is finding episodes that extracted badly, so the
    # consumer should not have to sort to see them.
    episodes_out.sort(key=lambda r: (r["rate"], -r["total_insights"], r["episode_id"]))

    # #1208 — no-silent-fail contract; see temporal_velocity for rationale.
    partial_reason: str | None = None
    if len(bundles) == 0:
        partial_reason = "no_bundles"
    elif not episodes_out:
        partial_reason = "no_episodes_with_insights"
    if partial_reason is not None:
        _logger.warning(
            "grounding_rate produced empty output run_id=%s enricher=%s reason=%s bundles=%d",
            ctx.run_id,
            ctx.enricher_id,
            partial_reason,
            len(bundles),
        )

    return {
        "episodes": episodes_out,
        "episode_count": len(bundles),
        # Corpus-wide roll-up, so a consumer wanting one number does not have to re-aggregate.
        "corpus_total_insights": corpus_total,
        "corpus_grounded_insights": corpus_grounded,
        "corpus_rate": round(corpus_grounded / corpus_total, 4) if corpus_total else 0.0,
        "partial_reason": partial_reason,
    }


_enrich_async = sync_enricher(_compute)


class GroundingRateEnricher:
    """Corpus-scope per-EPISODE grounded-insight ratio (#1927; was per-Person)."""

    manifest = EnricherManifest(
        id="grounding_rate",
        version="2.0.0",  # per-EPISODE, not per-person (#1927) — breaking shape change
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".gi.json"],
        writes="grounding_rate.json",
        description=(
            "Per-EPISODE ratio of Insights backed by a supporting quote — extraction "
            "quality, worst episodes first. Was per-Person until #1927, which returned "
            "1.0 for everyone because an ungrounded insight has no speaker."
        ),
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


__all__ = ["GroundingRateEnricher"]
