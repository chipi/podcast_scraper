"""``topic_consensus`` — cross-Person *corroboration* per Topic (ml tier, ADR-108).

The reimagining of ``nli_contradiction``. The contradiction detector hit 0% precision because
sentence-pair NLI can't tell "same contested *proposition*" from "same *topic*" (the
shared-question gate). This enricher flips to the robust **entailment** side of NLI and detects
**agreement** — "what the corpus corroborates" — which is more useful and far more separable from
topic-adjacency.

**Shared-question gate, no LLM:** a pair is emitted only on **symmetric entailment** — A entails B
*and* B entails A above threshold. Mutual entailment (each is a paraphrase/consequence of the other)
can't be mere topic-adjacency, so symmetry *is* the gate: two speakers making mutually-entailing
claims on a topic are corroborating the same proposition.

Reuses the CPU DeBERTa ``NliScorer`` the project already loads (it already returns ``entailment``);
no scorer change, no LLM. Gated by the data-driven accuracy gate until an eval clears
``precision ≥ 0.5`` against the corroboration gold.
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


def _topic_insight_speaker_index(
    bundles: list[EpisodeArtifactBundle],
) -> tuple[dict[str, list[tuple[str, str, str]]], dict[str, str]]:
    """``topic_id → [(insight_id, person_id, insight_text)]`` + a person label map."""
    by_topic: dict[str, list[tuple[str, str, str]]] = {}
    person_label: dict[str, str] = {}
    for b in bundles:
        gi = load_gi(b)
        insight_text = {
            str(n.get("id") or ""): str((n.get("properties") or {}).get("text") or "")
            for n in nodes_of_type(gi, "Insight")
            if n.get("id")
        }
        for n in nodes_of_type(gi, "Person"):
            pid = str(n.get("id") or "")
            if pid:
                person_label[pid] = str((n.get("properties") or {}).get("name") or pid)
        quote_speaker = {
            str(e.get("from") or ""): str(e.get("to") or "") for e in edges_of_type(gi, "SPOKEN_BY")
        }
        insight_speaker: dict[str, str] = {}
        for e in edges_of_type(gi, "SUPPORTED_BY"):
            spk = quote_speaker.get(str(e.get("to") or ""))
            if str(e.get("from") or "") and spk:
                insight_speaker.setdefault(str(e.get("from")), spk)
        for e in edges_of_type(gi, "ABOUT"):
            iid, tid = str(e.get("from") or ""), str(e.get("to") or "")
            spk = insight_speaker.get(iid)
            if iid and tid and spk:
                by_topic.setdefault(tid, []).append((iid, spk, insight_text.get(iid, "")))
    return by_topic, person_label


class TopicConsensusEnricher:
    """Corpus-scope cross-Person corroboration per Topic via symmetric NLI-entailment."""

    manifest = EnricherManifest(
        id="topic_consensus",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json"],
        writes="topic_consensus.json",
        description=(
            "Cross-Person corroboration per Topic via symmetric NLI-entailment (ADR-108). "
            "Emits agreeing (mutually-entailing) speaker pairs. No LLM."
        ),
        expected_duration_s=120,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.6,
                    "description": "Min symmetric entailment probability to emit a consensus pair.",
                },
            },
        },
        provider_requirement=ProviderRequirement(
            protocol="NliScorer",
            description="NLI scorer (DeBERTa local, scripted fixture, or future hosted NLI).",
        ),
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
        threshold: float = 0.6,
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
        """Emit cross-Person pairs whose insights mutually entail on a shared Topic."""
        threshold = float(config.get("threshold", self._threshold))
        by_topic, person_label = _topic_insight_speaker_index(all_bundles or [])

        consensus: list[dict[str, Any]] = []
        pairs_scored = 0
        for tid, entries in sorted(by_topic.items()):
            usable = [(iid, pid, txt) for iid, pid, txt in entries if txt.strip()]
            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    iid_a, pid_a, txt_a = usable[i]
                    iid_b, pid_b, txt_b = usable[j]
                    if pid_a == pid_b:  # same speaker → not cross-Person corroboration
                        continue
                    if ctx.cancel_event.is_set():
                        return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                    ab = await self._scorer.score(txt_a, txt_b)
                    ba = await self._scorer.score(txt_b, txt_a)
                    pairs_scored += 1
                    # Symmetric entailment IS the shared-question gate: mutual paraphrase can't be
                    # mere topic-adjacency. One-directional entailment is dropped as too weak.
                    symmetric = min(ab.entailment, ba.entailment)
                    if symmetric < threshold:
                        continue
                    consensus.append(
                        {
                            "topic_id": tid,
                            "person_a_id": pid_a,
                            "person_a_name": person_label.get(pid_a, pid_a),
                            "person_b_id": pid_b,
                            "person_b_name": person_label.get(pid_b, pid_b),
                            "insight_a_id": iid_a,
                            "insight_a_text": txt_a,
                            "insight_b_id": iid_b,
                            "insight_b_text": txt_b,
                            "consensus_score": round(symmetric, 6),
                            "model_id": self._model_id,
                            "model_version": self._model_version,
                        }
                    )

        consensus.sort(key=lambda r: (-r["consensus_score"], r["topic_id"], r["insight_a_id"]))
        return EnricherResult(
            status=STATUS_OK,
            data={
                "model_id": self._model_id,
                "model_version": self._model_version,
                "threshold": threshold,
                "pairs_scored": pairs_scored,
                "consensus": consensus,
            },
            records_written=len(consensus),
        )


__all__ = ["TopicConsensusEnricher"]
