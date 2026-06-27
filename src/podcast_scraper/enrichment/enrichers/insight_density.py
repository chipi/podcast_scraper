"""``insight_density`` — Insight count per segment (early/mid/late) (deterministic).

For each episode, partitions Quotes by their start timestamp into
thirds (early/mid/late) of the episode duration, then counts how many
Insights are supported by quotes in each third.

Reads ``*.gi.json`` (Quote + Insight + SUPPORTED_BY edges with quote
``properties.start_s`` or ``start_seconds``) and ``*.metadata.json``
for ``duration_seconds``. Tolerates missing duration / timestamps by
falling back to even quote-count splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import (
    edges_of_type,
    load_gi,
    load_metadata,
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


def _quote_start_s(node: dict[str, Any]) -> float | None:
    props = node.get("properties") or {}
    for key in ("start_s", "start_seconds", "start"):
        val = props.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _segment_of(start_s: float, total_s: float) -> str:
    if total_s <= 0:
        return "unknown"
    frac = start_s / total_s
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    if bundle is None:
        from podcast_scraper.enrichment.resilience import BadInputError

        raise BadInputError("insight_density is EPISODE scope and requires a bundle; got None")
    gi = load_gi(bundle)
    meta = load_metadata(bundle)
    duration = float(meta.get("duration_seconds") or 0.0)

    # Quote id → start_s
    quote_start: dict[str, float | None] = {}
    for node in nodes_of_type(gi, "Quote"):
        qid = str(node.get("id") or "")
        if qid:
            quote_start[qid] = _quote_start_s(node)
    # Insight id → Quote ids (via SUPPORTED_BY)
    insight_quotes: dict[str, list[str]] = {}
    for edge in edges_of_type(gi, "SUPPORTED_BY"):
        iid = str(edge.get("from") or "")
        qid = str(edge.get("to") or "")
        if not iid or not qid:
            continue
        insight_quotes.setdefault(iid, []).append(qid)

    # Decide split mode.
    has_timing = duration > 0 and any(quote_start.get(q) is not None for q in quote_start)
    counts = {"early": 0, "mid": 0, "late": 0, "unknown": 0}
    insight_segments: list[dict[str, Any]] = []
    if has_timing:
        for iid, qids in insight_quotes.items():
            # Insight segment = segment of its earliest quote with timing.
            starts = [quote_start.get(q) for q in qids]
            valid = [s for s in starts if s is not None]
            if not valid:
                counts["unknown"] += 1
                insight_segments.append({"insight_id": iid, "segment": "unknown"})
                continue
            seg = _segment_of(min(valid), duration)
            counts[seg] += 1
            insight_segments.append({"insight_id": iid, "segment": seg})
    else:
        # Even split by insight order.
        insight_ids = list(insight_quotes.keys())
        n = len(insight_ids)
        third = max(1, n // 3) if n else 1
        for idx, iid in enumerate(insight_ids):
            if idx < third:
                seg = "early"
            elif idx < 2 * third:
                seg = "mid"
            else:
                seg = "late"
            counts[seg] += 1
            insight_segments.append({"insight_id": iid, "segment": seg})

    total_insights = sum(counts.values())
    return {
        "episode_id": bundle.episode_id,
        "duration_seconds": duration,
        "has_timing": has_timing,
        "counts": counts,
        "total_insights": total_insights,
        "insight_segments": insight_segments,
    }


_enrich_async = sync_enricher(_compute)


class InsightDensityEnricher:
    """Episode-scope Insight count per (early/mid/late) third."""

    manifest = EnricherManifest(
        id="insight_density",
        version="1.0.0",
        scope=EnricherScope.EPISODE,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".gi.json", ".metadata.json"],
        writes="insight_density.json",
        description="Per-episode Insight density by (early / mid / late) third of duration.",
        expected_duration_s=5,
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


__all__ = ["InsightDensityEnricher"]
