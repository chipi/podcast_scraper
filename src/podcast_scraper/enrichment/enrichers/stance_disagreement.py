"""``stance_disagreement`` — cross-Person *stance* disagreement per Topic (ml tier, #1144).

Where ``nli_contradiction`` scores *atomic-insight* pairs and over-fires on merely
topic-adjacent insights (0% precision, #1106), this enricher works at the **stance**
level: per Topic it aggregates each speaker's Insights into one position and scores that
*pair once* — coarser and more robust than cherry-picked atomic pairs.

Deliberately **no LLM** (#1144, no-LLM path): it reuses the CPU DeBERTa ``NliScorer`` the
project already loads for ``nli_contradiction`` — zero marginal cost, deterministic, runs
on ``.[dev]``. Whether stance-level NLI is precise enough was answered by **measurement,
not assertion** — and the answer is **no**. ``disagreement_stance_eval_v1.py`` scored this
scorer against ``gold_v1.jsonl`` (40 prod-v2 pairs + the designed v3 Cho-vs-Bessent
disagreement + a Cho-vs-Fischer topic-adjacent hard negative):

* stance-aggregate symmetric contradiction → **0% precision, 0% recall** (the real
  disagreement dilutes to 0.30, below threshold; ``no_shared_question`` negatives over-fire),
* atomic-max → catches the positive (0.999) but **10% precision** — the ``no_shared_question``
  negatives *also* score 0.96–0.999, so DeBERTa cannot separate genuine opposition from mere
  topic-adjacency. This is the same failure that gave ``nli_contradiction`` 0% precision (#1106).

The shared-question gate (does A engage B's *proposition*, not just B's *topic*?) is a
semantic judgment DeBERTa structurally lacks; it needs an LLM, which #1144 rules out
(operator constraint: no LLM-dependent enrichers). So this enricher ships **gated dark**:
its manifest ``accuracy_gate`` (precision ≥ 0.5) + the measured ``gate_metrics.json``
(precision 0.0) exclude it from the registry / profiles / UI. It is retained as a wired,
gate-guarded **framework placeholder** — if a future *non-LLM* scorer ever clears the gold,
it auto-promotes with no code edit; until then ``perspectives`` (#1146) is the live no-LLM
multi-speaker surface (it shows both stances without claiming "disagreement", so it cannot
fabricate). ``gold_v1.jsonl`` is the durable regression bar any future detector must beat.

Tests inject a mock ``NliScorer`` (the machinery is correct even though the real scorer's
precision is not); the eval scores the real DeBERTa output against the gold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers.nli_contradiction import (
    _episode_topic_insight_speaker_index,
)
from podcast_scraper.enrichment.protocol import (
    AccuracyGateRule,
    AccuracyGateSpec,
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    ProviderRequirement,
    RunContext,
    STATUS_CANCELLED,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.protocol import NliScorer


def _topic_speaker_stances(
    by_topic: dict[str, list[tuple[str, str, str]]],
) -> dict[str, dict[str, list[str]]]:
    """``topic_id → {person_id → [insight_text, ...]}`` from the nli index."""
    out: dict[str, dict[str, list[str]]] = {}
    for tid, entries in by_topic.items():
        per_person: dict[str, list[str]] = {}
        for _iid, pid, text in entries:
            if text.strip():
                per_person.setdefault(pid, []).append(text.strip())
        out[tid] = per_person
    return out


class StanceDisagreementEnricher:
    """Corpus-scope cross-Person *stance* disagreement per Topic (ml tier, #1144).

    No LLM: an injected :class:`NliScorer` (CPU DeBERTa) scores the two aggregated
    stances. A pair is emitted only when the contradiction probability clears the
    threshold **in both directions** (A-vs-B *and* B-vs-A) — a symmetry check that
    filters one-directional "A negates a clause of B" noise. Measured insufficient as a
    shared-question gate (see module docstring): retained for the framework/eval, gated
    dark by the accuracy gate.
    """

    manifest = EnricherManifest(
        id="stance_disagreement",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json"],
        writes="stance_disagreement.json",
        description=(
            "Cross-Person stance disagreement per Topic via stance-level NLI "
            "(no LLM, #1144). Emits symmetric contradictions above threshold."
        ),
        expected_duration_s=120,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "min_insights": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2,
                    "description": "Min insights a speaker needs on a topic to have a stance.",
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.6,
                    "description": "Min (symmetric) contradiction probability to emit a pair.",
                },
            },
        },
        provider_requirement=ProviderRequirement(
            protocol="NliScorer",
            description="NLI scorer (DeBERTa local, scripted fixture, or future hosted NLI).",
        ),
        # Data-driven admission (RFC-088 accuracy-gate): stays out of the registry /
        # profiles / UI until an eval scored against the #1144 Opus silver records
        # precision >= 0.5. This is exactly what decides "is DeBERTa-stance enough, or do
        # we need the LLM?" — no code edit, no premature LLM spend.
        accuracy_gate=AccuracyGateSpec(
            rules=(AccuracyGateRule(metric_name="precision", min_value=0.5),),
            on_missing_data="reject",
        ),
    )

    def __init__(
        self,
        scorer: NliScorer,
        *,
        model_id: str = "cross-encoder/nli-deberta-v3-small",
        model_version: str = "v1",
        min_insights: int = 2,
        threshold: float = 0.6,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._scorer = scorer
        self._model_id = model_id
        self._model_version = model_version
        self._min_insights = min_insights
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
        """Score aggregated cross-Person stances per Topic; emit symmetric contradictions."""
        min_insights = int(config.get("min_insights", self._min_insights))
        threshold = float(config.get("threshold", self._threshold))
        bundles = all_bundles or []
        by_topic, person_label = _episode_topic_insight_speaker_index(bundles)
        stances = _topic_speaker_stances(by_topic)

        disagreements: list[dict[str, Any]] = []
        pairs_scored = 0
        for tid, per_person in stances.items():
            rich = sorted(p for p, txts in per_person.items() if len(txts) >= min_insights)
            for i in range(len(rich)):
                for j in range(i + 1, len(rich)):
                    if ctx.cancel_event.is_set():
                        return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                    pid_a, pid_b = rich[i], rich[j]
                    stance_a = " ".join(per_person[pid_a])
                    stance_b = " ".join(per_person[pid_b])
                    ab = await self._scorer.score(stance_a, stance_b)
                    ba = await self._scorer.score(stance_b, stance_a)
                    pairs_scored += 1
                    # Symmetry: both directions must read as contradiction. One-sided
                    # scores are the topic-adjacency noise that sank nli_contradiction.
                    symmetric = min(ab.contradiction, ba.contradiction)
                    if symmetric < threshold:
                        continue
                    disagreements.append(
                        {
                            "topic_id": tid,
                            "person_a_id": pid_a,
                            "person_a_name": person_label.get(pid_a, pid_a),
                            "person_b_id": pid_b,
                            "person_b_name": person_label.get(pid_b, pid_b),
                            "a_stance": stance_a,
                            "b_stance": stance_b,
                            "contradiction_score": round(symmetric, 6),
                            "model_id": self._model_id,
                            "model_version": self._model_version,
                        }
                    )

        disagreements.sort(
            key=lambda r: (-r["contradiction_score"], r["topic_id"], r["person_a_id"])
        )
        return EnricherResult(
            status=STATUS_OK,
            data={
                "model_id": self._model_id,
                "model_version": self._model_version,
                "min_insights": min_insights,
                "threshold": threshold,
                "pairs_scored": pairs_scored,
                "disagreements": disagreements,
            },
            records_written=len(disagreements),
        )


__all__ = ["StanceDisagreementEnricher"]
