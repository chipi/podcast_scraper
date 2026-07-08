"""Reference scorer — ``guest_coappearance`` (set / unordered-pairs shape).

Grades the emitted co-appearance pairs against authored expected pairs as
unordered sets. Emits ``precision`` / ``recall`` / ``f1``. This is the template
every *set* enricher scorer follows (topic_cooccurrence_corpus, contradiction
pairs, …).
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.eval.protocol import ScoreResult, ScorerManifest


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def _pair_key(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


class GuestCoappearanceScorer:
    """Set accuracy scorer for the ``guest_coappearance`` enricher."""

    manifest = ScorerManifest(
        enricher_id="guest_coappearance",
        version="1.0.0",
        metrics=("precision", "recall", "f1"),
        description="Emitted co-appearance pairs vs expected, as unordered sets.",
        gold_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "expected_pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                }
            },
            "required": ["expected_pairs"],
        },
    )

    def score(
        self,
        *,
        output: dict[str, Any],
        gold: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Precision/recall/F1 of emitted vs expected unordered person pairs."""
        pairs = output.get("pairs")
        expected_raw = gold.get("expected_pairs")
        if not isinstance(pairs, list) or not isinstance(expected_raw, list):
            return ScoreResult(
                enricher_id="guest_coappearance",
                skipped=True,
                notes="missing pairs output or expected_pairs gold",
            )
        pred = {
            _pair_key(str(p.get("person_a_id") or ""), str(p.get("person_b_id") or ""))
            for p in pairs
            if p.get("person_a_id") and p.get("person_b_id")
        }
        exp = {
            _pair_key(str(e[0]), str(e[1]))
            for e in expected_raw
            if isinstance(e, (list, tuple)) and len(e) == 2
        }
        if not exp:
            return ScoreResult(
                enricher_id="guest_coappearance", skipped=True, notes="empty expected_pairs gold"
            )
        hit = len(pred & exp)
        precision = hit / len(pred) if pred else 0.0
        recall = hit / len(exp)
        return ScoreResult(
            enricher_id="guest_coappearance",
            metrics={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(_f1(precision, recall), 4),
            },
            sample_count=len(exp),
        )


__all__ = ["GuestCoappearanceScorer"]
