"""``nli_contradiction`` — cross-Person Insight contradictions per Topic (ml tier).

For every Topic, the enricher collects Insights that:

1. mention that topic (via ``ABOUT`` edge in the GI), AND
2. are spoken by *different* Persons (via SUPPORTED_BY → SPOKEN_BY).

It then asks the injected ``NliScorer`` to compute the contradiction
probability for each cross-Person Insight pair on that topic. Pairs
with score ≥ ``threshold`` (default 0.5) land in the output.

Inherited resilience (ML tier policy): 2 retries, 60s max backoff,
circuit at 3 consecutive failures, auto-disable at 2 failed runs.
``opt_in_flags`` are NOT required for this enricher — it's CPU-local,
not LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import (
    edges_of_type,
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
    STATUS_CANCELLED,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.protocol import NliScorer


def _episode_topic_insight_speaker_index(
    bundles: list[EpisodeArtifactBundle],
) -> tuple[dict[str, list[tuple[str, str, str]]], dict[str, str]]:
    """Build the index `(topic_id → [(insight_id, person_id, insight_text)])`.

    Walks every bundle's GI; resolves Insight → Quote (SUPPORTED_BY)
    → Person (SPOKEN_BY) and Insight → Topic (ABOUT). Returns the
    index plus a person_id → name label map.
    """
    by_topic: dict[str, list[tuple[str, str, str]]] = {}
    person_label: dict[str, str] = {}
    for b in bundles:
        gi = load_gi(b)
        # Insight text (best-effort: properties.text or properties.title).
        insight_text: dict[str, str] = {}
        for node in nodes_of_type(gi, "Insight"):
            iid = str(node.get("id") or "")
            if not iid:
                continue
            props = node.get("properties") or {}
            insight_text[iid] = str(props.get("text") or props.get("title") or "")
        # Quote → speaker.
        quote_speaker: dict[str, str] = {}
        for edge in edges_of_type(gi, "SPOKEN_BY"):
            q = str(edge.get("from") or "")
            p = str(edge.get("to") or "")
            if q and p:
                quote_speaker[q] = p
        for node in nodes_of_type(gi, "Person"):
            pid = str(node.get("id") or "")
            if pid:
                person_label[pid] = str((node.get("properties") or {}).get("name") or pid)
        # Insight → speaker (any quote that supports it).
        insight_speaker: dict[str, str] = {}
        for edge in edges_of_type(gi, "SUPPORTED_BY"):
            iid = str(edge.get("from") or "")
            qid = str(edge.get("to") or "")
            spk = quote_speaker.get(qid)
            if iid and spk:
                insight_speaker.setdefault(iid, spk)
        # Insight → topics (ABOUT).
        for edge in edges_of_type(gi, "ABOUT"):
            iid = str(edge.get("from") or "")
            tid = str(edge.get("to") or "")
            if not (iid and tid):
                continue
            spk = insight_speaker.get(iid)
            if not spk:
                continue
            by_topic.setdefault(tid, []).append((iid, spk, insight_text.get(iid, "")))
    return by_topic, person_label


class NliContradictionEnricher:
    """Corpus-scope cross-Person Insight contradictions per Topic (ml tier)."""

    manifest = EnricherManifest(
        id="nli_contradiction",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json"],
        writes="nli_contradiction.json",
        description=(
            "Cross-Person Insight contradiction pairs per Topic via NliScorer "
            "(default threshold 0.5)."
        ),
        expected_duration_s=300,
    )

    def __init__(
        self,
        scorer: NliScorer,
        *,
        model_id: str = "cross-encoder/nli-deberta-v3-small",
        model_version: str = "v1",
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._scorer = scorer
        self._model_id = model_id
        self._model_version = model_version
        self._threshold = threshold

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        threshold = float(config.get("threshold", self._threshold))
        model_id = str(config.get("model_id", self._model_id))
        model_version = str(config.get("model_version", self._model_version))
        bundles = all_bundles or []
        by_topic, person_label = _episode_topic_insight_speaker_index(bundles)

        contradictions: list[dict[str, Any]] = []
        pairs_scored = 0
        for tid, entries in by_topic.items():
            # Only cross-Person pairs.
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    if ctx.cancel_event.is_set():
                        return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                    iid_a, pid_a, text_a = entries[i]
                    iid_b, pid_b, text_b = entries[j]
                    if pid_a == pid_b:
                        continue
                    score = await self._scorer.score(text_a, text_b)
                    pairs_scored += 1
                    if score.contradiction >= threshold:
                        contradictions.append(
                            {
                                "topic_id": tid,
                                "person_a_id": pid_a,
                                "person_a_name": person_label.get(pid_a, pid_a),
                                "person_b_id": pid_b,
                                "person_b_name": person_label.get(pid_b, pid_b),
                                "insight_a_id": iid_a,
                                "insight_b_id": iid_b,
                                "contradiction_score": round(score.contradiction, 6),
                                "model_id": model_id,
                                "model_version": model_version,
                            }
                        )

        contradictions.sort(
            key=lambda r: (-r["contradiction_score"], r["topic_id"], r["insight_a_id"])
        )
        return EnricherResult(
            status=STATUS_OK,
            data={
                "model_id": model_id,
                "model_version": model_version,
                "threshold": threshold,
                "pairs_scored": pairs_scored,
                "contradictions": contradictions,
            },
        )


__all__ = ["NliContradictionEnricher"]
