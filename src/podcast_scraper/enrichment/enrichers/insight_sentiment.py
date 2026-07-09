"""``insight_sentiment`` — per-Insight VADER sentiment (deterministic).

Scores every Insight's text with VADER (a pure-Python lexicon sentiment analyzer that
bundles its own lexicon — no model download, no network) and labels it
negative / neutral / positive via VADER's standard ±0.05 compound thresholds.

This is the **colour layer** for the conversation-timeline surfaces (per-person position
arc + global topic timeline): each insight gets a stable sentiment tint. Unlike a stance
*score*, sentiment is a decoration — a neutral insight (the common case for factual
summaries) is a perfectly fine grey, so a mostly-neutral corpus is not a failure.

Deterministic tier: the lexicon is fixed, so the same insight always scores the same →
no accuracy gate, always admitted. Reads ``*.gi.json`` (Insight nodes); writes
``metadata/enrichments/{stem}.insight_sentiment.json``.

**Consumption is indirect — do not expect a frontend reader.** No viewer/player
component reads this artifact directly. The read-time CIL layer joins it:
``cil_queries._attach_sentiment`` (``server/cil_queries.py``) reads the
``{stem}.insight_sentiment.json`` sidecar and tags each Insight node with
``sentiment: {compound, label}`` so the conversation-arc / position-arc responses
carry the tint. So the enricher → server-join → arc-tint chain is easy to break
silently in a refactor (deleting the artifact or renaming ``label`` / ``compound``
de-colours the arcs with no failing frontend read). See the player walkthrough v3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import load_gi, nodes_of_type
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)

# VADER's canonical decision thresholds on the compound score.
_POS_CUTOFF = 0.05
_NEG_CUTOFF = -0.05

_ANALYZER: Any = None


def _analyzer() -> Any:
    """Lazy singleton VADER analyzer (loads the bundled lexicon once per process)."""
    global _ANALYZER
    if _ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def _label(compound: float) -> str:
    if compound >= _POS_CUTOFF:
        return "positive"
    if compound <= _NEG_CUTOFF:
        return "negative"
    return "neutral"


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    if bundle is None:
        from podcast_scraper.enrichment.resilience import BadInputError

        raise BadInputError("insight_sentiment is EPISODE scope and requires a bundle; got None")
    gi = load_gi(bundle)
    sia = _analyzer()
    insights: list[dict[str, Any]] = []
    counts = {"negative": 0, "neutral": 0, "positive": 0}
    for node in nodes_of_type(gi, "Insight"):
        iid = str(node.get("id") or "")
        text = str((node.get("properties") or {}).get("text") or "").strip()
        if not iid or not text:
            continue
        compound = round(float(sia.polarity_scores(text)["compound"]), 4)
        label = _label(compound)
        counts[label] += 1
        insights.append({"insight_id": iid, "compound": compound, "label": label})
    return {
        "episode_id": bundle.episode_id,
        "counts": counts,
        "total_insights": len(insights),
        "insights": insights,
    }


_enrich_async = sync_enricher(_compute)


class InsightSentimentEnricher:
    """Episode-scope per-Insight VADER sentiment (compound + neg/neu/pos label)."""

    manifest = EnricherManifest(
        id="insight_sentiment",
        version="1.0.0",
        scope=EnricherScope.EPISODE,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".gi.json"],
        writes="insight_sentiment.json",
        description="Per-Insight VADER sentiment (compound + neg/neu/pos) — timeline colour layer.",
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
        """Score each Insight's text with VADER → compound + label."""
        return await _enrich_async(
            bundle=bundle,
            corpus_root=corpus_root,
            all_bundles=all_bundles,
            config=config,
            ctx=ctx,
        )


__all__ = ["InsightSentimentEnricher"]
