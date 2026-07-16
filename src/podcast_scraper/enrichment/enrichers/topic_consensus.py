"""``topic_consensus`` — cross-Person *corroboration* per Topic (ml tier, ADR-108).

The reimagining of ``nli_contradiction``. The contradiction detector hit 0% precision because
sentence-pair NLI can't tell "same contested *proposition*" from "same *topic*" (the
shared-question gate). This enricher detects **agreement** instead — "what the corpus corroborates".

**The signal (from real-corpus eval, docs/wip/ADR-108-REAL-CORPUS-EVAL-2026-07.md):** an early
version gated on *symmetric NLI entailment* and found almost nothing — genuine agreement between two
speakers is expressed in different words, so mutual entailment is near-zero (1 pair / 2903 on
prod-v2). The signal that actually recalls agreement is a **composite**:

* **embedding cosine** ≥ ``cos_threshold`` — the *shared-question* gate (are the two insights about
  the same proposition), and
* **NLI contradiction** ≤ ``contra_threshold`` — the *direction* gate (they don't disagree),

which filters the similar-but-opposite pairs embedding proximity alone admits. On prod-v2 this hits
precision ~0.91 with ~22 pairs. Both models are CPU-local (MiniLM + DeBERTa) via the injected
:class:`ConsensusScorer` — still no LLM. Gated by the data-driven accuracy gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import (
    edges_of_type,
    is_unresolved_speaker_placeholder,
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
from podcast_scraper.enrichment.scorers.protocol import ConsensusScorer


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
            if not (iid and tid and spk):
                continue
            # An unresolved diarization voice is not a real person — excluding it
            # stops cross-episode SPEAKER_NN coincidences counting as consensus (#1167).
            if is_unresolved_speaker_placeholder(spk, person_label.get(spk)):
                continue
            by_topic.setdefault(tid, []).append((iid, spk, insight_text.get(iid, "")))
    return by_topic, person_label


class TopicConsensusEnricher:
    """Corpus-scope cross-Person corroboration per Topic (embedding cosine + low contradiction)."""

    manifest = EnricherManifest(
        id="topic_consensus",
        version="2.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json"],
        writes="topic_consensus.json",
        description=(
            "Cross-Person corroboration per Topic — embedding cosine (shared-question gate) + low "
            "NLI contradiction (they don't disagree). ADR-108 composite. No LLM."
        ),
        expected_duration_s=180,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cos_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.70,
                    "description": "Min embedding cosine — the shared-question gate.",
                },
                "contra_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                    "description": "Max NLI contradiction (either direction) — the direction gate.",
                },
            },
        },
        provider_requirement=ProviderRequirement(
            protocol="ConsensusScorer",
            description="Composite consensus scorer (consensus_local MiniLM+DeBERTa, or fixture).",
        ),
        accuracy_gate=AccuracyGateSpec(
            rules=(AccuracyGateRule(metric_name="precision", min_value=0.5),),
            on_missing_data="reject",
        ),
    )

    def __init__(
        self,
        scorer: ConsensusScorer,
        *,
        model_id: str = "all-MiniLM-L6-v2+deberta-v3-small",
        model_version: str = "v2",
        cos_threshold: float = 0.70,
        contra_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= cos_threshold <= 1.0:
            raise ValueError("cos_threshold must be in [0, 1]")
        if not 0.0 <= contra_threshold <= 1.0:
            raise ValueError("contra_threshold must be in [0, 1]")
        self._scorer = scorer
        self._model_id = model_id
        self._model_version = model_version
        self._cos_threshold = cos_threshold
        self._contra_threshold = contra_threshold

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Emit cross-Person pairs that are semantically close AND non-contradictory on a Topic."""
        cos_threshold = float(config.get("cos_threshold", self._cos_threshold))
        contra_threshold = float(config.get("contra_threshold", self._contra_threshold))
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
                    sig = await self._scorer.score(txt_a, txt_b)
                    pairs_scored += 1
                    # Composite gate: embedding proximity (same proposition) AND low contradiction
                    # (they don't disagree). Cosine alone admits similar-but-opposite pairs; the
                    # contradiction filter removes them (ADR-108 eval).
                    if sig.cosine < cos_threshold or sig.contradiction > contra_threshold:
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
                            "consensus_score": round(sig.cosine, 6),
                            "cosine": round(sig.cosine, 6),
                            "contradiction": round(sig.contradiction, 6),
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
                "cos_threshold": cos_threshold,
                "contra_threshold": contra_threshold,
                "pairs_scored": pairs_scored,
                "consensus": consensus,
            },
            records_written=len(consensus),
        )


__all__ = ["TopicConsensusEnricher"]
