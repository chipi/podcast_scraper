"""Reference scorer — ``topic_similarity`` (ranking / top-K shape).

Grades the per-topic top-K neighbours against authored expected neighbours,
macro-averaged over the gold topics. Emits ``precision_at_k`` / ``recall_at_k``
/ ``f1_at_k``. This is the template every *ranking* enricher scorer follows
(topic_theme_clusters neighbours, query relatedness, …).
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.eval.protocol import ScoreResult, ScorerManifest


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


class TopicSimilarityScorer:
    """Ranking accuracy scorer for the ``topic_similarity`` enricher."""

    manifest = ScorerManifest(
        enricher_id="topic_similarity",
        version="1.0.0",
        metrics=("precision_at_k", "recall_at_k", "f1_at_k"),
        description="Per-topic top-K neighbours vs expected, macro-averaged.",
        gold_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "expected_neighbours": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "string"}},
                }
            },
            "required": ["expected_neighbours"],
        },
    )

    def score(
        self,
        *,
        output: dict[str, Any],
        gold: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Macro-average precision/recall@k of emitted vs expected neighbours."""
        expected = gold.get("expected_neighbours")
        topics = output.get("topics")
        if not isinstance(expected, dict) or not isinstance(topics, list):
            return ScoreResult(
                enricher_id="topic_similarity",
                skipped=True,
                notes="missing topics output or expected_neighbours gold",
            )
        # Grade at the K the enricher ran with when supplied, else the emitted length.
        cfg_k = None
        if config and isinstance(config.get("top_k"), int):
            cfg_k = int(config["top_k"])

        emitted: dict[str, list[str]] = {}
        for t in topics:
            tid = str(t.get("topic_id") or "")
            if not tid:
                continue
            neighbours = [str(n.get("topic_id") or "") for n in t.get("top_k", [])]
            emitted[tid] = [n for n in neighbours if n]

        precisions: list[float] = []
        recalls: list[float] = []
        graded = 0
        for tid, exp_list in expected.items():
            exp = {str(x) for x in exp_list if x}
            if not exp or tid not in emitted:
                continue
            pred_list = emitted[tid][:cfg_k] if cfg_k else emitted[tid]
            pred = set(pred_list)
            hit = len(pred & exp)
            precisions.append(hit / len(pred) if pred else 0.0)
            recalls.append(hit / len(exp))
            graded += 1

        if graded == 0:
            return ScoreResult(
                enricher_id="topic_similarity",
                skipped=True,
                notes="no gold topic overlapped the emitted topics",
            )
        precision = sum(precisions) / graded
        recall = sum(recalls) / graded
        return ScoreResult(
            enricher_id="topic_similarity",
            metrics={
                "precision_at_k": round(precision, 4),
                "recall_at_k": round(recall, 4),
                "f1_at_k": round(_f1(precision, recall), 4),
            },
            sample_count=graded,
        )


__all__ = ["TopicSimilarityScorer"]
