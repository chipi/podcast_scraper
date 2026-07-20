"""``grounding_rate`` — per-Person grounded-insight ratio (deterministic).

For each Person appearing as a Quote speaker across the corpus, computes:

* ``total_insights`` — Insights they support via SPOKEN_BY → SUPPORTED_BY chain
* ``grounded_insights`` — Insights with ``properties.grounded == true``
* ``rate`` — grounded / total (0.0 when total == 0)

Output drives "speaker credibility / rigor" dashboards. Reads
``*.gi.json`` only (quotes + insights + edges); deterministic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

from podcast_scraper.enrichment.enrichers._loaders import (
    edges_of_type,
    is_unresolved_speaker_placeholder,
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
    per_person_total: dict[str, int] = defaultdict(int)
    per_person_grounded: dict[str, int] = defaultdict(int)
    person_labels: dict[str, str] = {}

    bundles = all_bundles or []
    for b in bundles:
        gi = load_gi(b)
        # Person id -> Quote ids the person spoke
        quote_to_speaker: dict[str, str] = {}
        for edge in edges_of_type(gi, "SPOKEN_BY"):
            q = str(edge.get("from") or "")
            p = str(edge.get("to") or "")
            if q and p:
                quote_to_speaker[q] = p
        # Insight grounded flag
        insight_grounded: dict[str, bool] = {}
        for node in nodes_of_type(gi, "Insight"):
            iid = str(node.get("id") or "")
            if not iid:
                continue
            grounded = bool((node.get("properties") or {}).get("grounded", False))
            insight_grounded[iid] = grounded
        # Person label
        for node in nodes_of_type(gi, "Person"):
            pid = str(node.get("id") or "")
            if not pid:
                continue
            person_labels[pid] = str((node.get("properties") or {}).get("name") or pid)
        # Insight -> Quote(s) via SUPPORTED_BY
        for edge in edges_of_type(gi, "SUPPORTED_BY"):
            insight_id = str(edge.get("from") or "")
            quote_id = str(edge.get("to") or "")
            speaker = quote_to_speaker.get(quote_id)
            if not insight_id or not speaker:
                continue
            per_person_total[speaker] += 1
            if insight_grounded.get(insight_id):
                per_person_grounded[speaker] += 1

    persons_out: list[dict[str, Any]] = []
    for pid, total in per_person_total.items():
        name = person_labels.get(pid, pid)
        # Drop unresolved diarization placeholders — each episode's
        # ``SPEAKER_NN`` is a label-local id, so aggregating their
        # grounding rate across episodes is meaningless.
        if is_unresolved_speaker_placeholder(pid, name):
            continue
        grounded_count = per_person_grounded.get(pid, 0)
        persons_out.append(
            {
                "person_id": pid,
                "person_name": name,
                "total_insights": total,
                "grounded_insights": grounded_count,
                "rate": round(grounded_count / total, 4) if total else 0.0,
            }
        )
    persons_out.sort(key=lambda r: (-r["rate"], -r["total_insights"], r["person_id"]))

    # #1208 — no-silent-fail contract; see temporal_velocity for rationale.
    partial_reason: str | None = None
    if len(bundles) == 0:
        partial_reason = "no_bundles"
    elif not persons_out:
        partial_reason = "no_persons_with_insights"
    if partial_reason is not None:
        _logger.warning(
            "grounding_rate produced empty output run_id=%s enricher=%s reason=%s bundles=%d",
            ctx.run_id,
            ctx.enricher_id,
            partial_reason,
            len(bundles),
        )

    return {"persons": persons_out, "episode_count": len(bundles), "partial_reason": partial_reason}


_enrich_async = sync_enricher(_compute)


class GroundingRateEnricher:
    """Corpus-scope per-Person grounded-insight ratio."""

    manifest = EnricherManifest(
        id="grounding_rate",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".gi.json"],
        writes="grounding_rate.json",
        description="Per-Person ratio of grounded Insights they support across the corpus.",
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
