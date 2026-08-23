"""``topic_consensus`` — cross-Person *corroboration* per Topic (ml tier, ADR-108).

The reimagining of ``nli_contradiction``. The contradiction detector hit 0% precision because
sentence-pair NLI can't tell "same contested *proposition*" from "same *topic*" (the
shared-question gate). This enricher detects **agreement** instead — "what the corpus corroborates".

**The signal (from real-corpus eval,
docs/adr/ADR-108-nli-disagreement-enrichers-gated-dark.md):** an early
version gated on *symmetric NLI entailment* and found almost nothing — genuine agreement between two
speakers is expressed in different words, so mutual entailment is near-zero (1 pair / 2903 on
prod-v2). The signal that actually recalls agreement is a **composite**:

* **embedding cosine** ≥ ``cos_threshold`` — the *shared-question* gate (are the two insights about
  the same proposition), and
* **NLI contradiction** ≤ ``contra_threshold`` — the *direction* gate (they don't disagree),

which filters the similar-but-opposite pairs embedding proximity alone admits. On prod-v2 this hits
precision ~0.91 with ~22 pairs. Both models are CPU-local (MiniLM + DeBERTa) via the injected
:class:`ConsensusScorer` — still no LLM. Gated by the data-driven accuracy gate.

**Incremental (RFC-118):** the pairwise NLI is the entire wall-clock (O(pairs) at ~ms/pair; a
1-episode repair drove a ~28-min full-corpus pass that timed out). Raw ``(cosine, contradiction)``
scores are cached per ``(insight_a_id, insight_b_id)`` with the endpoint episode ids;
``enrich_incremental`` re-scores only pairs with an endpoint in the delta and reuses the rest.
The thresholds re-apply from raw scores every run, so a threshold change re-filters without
re-scoring. Full and incremental share one ``_compute`` kernel — full IS incremental with an
empty reusable set — so the two paths cannot diverge in scoring or filtering logic (§7 gate).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

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

if TYPE_CHECKING:
    from podcast_scraper.corpus_delta import CorpusDelta

_logger = logging.getLogger(__name__)

PAIR_CACHE_SCHEMA_VERSION = 1
PAIR_CACHE_FILENAME = "topic_consensus.pairs_cache.json"

_KEY_SEP = "\x1f"


def _topic_insight_speaker_index(
    bundles: list[EpisodeArtifactBundle],
) -> tuple[dict[str, list[tuple[str, str, str, str]]], dict[str, str]]:
    """``topic_id → [(insight_id, person_id, insight_text, episode_id)]`` + a person label map.

    ``episode_id`` (the bundle's) rides along so the RFC-118 pair cache can judge
    validity — a cached pair is reusable only when NEITHER endpoint episode changed.
    """
    by_topic: dict[str, list[tuple[str, str, str, str]]] = {}
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
            by_topic.setdefault(tid, []).append((iid, spk, insight_text.get(iid, ""), b.episode_id))
    return by_topic, person_label


def pair_cache_path(corpus_root: Path) -> Path:
    """RFC-118 raw-score cache sidecar, next to the enricher's output."""
    return Path(corpus_root) / "enrichments" / PAIR_CACHE_FILENAME


def _load_pair_cache(corpus_root: Path, *, model_id: str, model_version: str) -> dict[str, Any]:
    """Load ``{pair_key: {c, x, ea, eb}}``; ``{}`` on absence, corruption, schema or
    model mismatch — any bump to model id/version discards the cache wholesale."""
    try:
        raw = json.loads(pair_cache_path(corpus_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != PAIR_CACHE_SCHEMA_VERSION
        or raw.get("model_id") != model_id
        or raw.get("model_version") != model_version
    ):
        return {}
    pairs = raw.get("pairs")
    return pairs if isinstance(pairs, dict) else {}


def _write_pair_cache(
    corpus_root: Path,
    pairs: dict[str, Any],
    *,
    model_id: str,
    model_version: str,
) -> None:
    """Atomically persist the raw-score cache. Non-fatal on failure (it is a cache)."""
    p = pair_cache_path(corpus_root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "schema": PAIR_CACHE_SCHEMA_VERSION,
                    "model_id": model_id,
                    "model_version": model_version,
                    "pairs": pairs,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError as exc:
        _logger.warning("topic_consensus: could not write pair cache %s: %s", p, exc)


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
        # Corpus-scope + TWO local models (MiniLM embed + deberta-v3-small NLI), the NLI
        # scored pairwise within each topic — heavier than topic_similarity. Primarily the
        # hard ``asyncio.wait_for`` cap; 180s killed the run. It also feeds the heartbeat
        # stall-warning threshold (``is_stalled`` is checked post-completion; enrichers never
        # call ``record_heartbeat``), so raising it raises that warning too — fine, it's a
        # post-hoc log, not a live watchdog. Sized for compute + a run-1 cold MiniLM+DeBERTa
        # download; HF cache warms runs 2+. Advisor 2026-08-23.
        expected_duration_s=600,
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
        supports_incremental=True,
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

    async def _compute(
        self,
        *,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle],
        config: dict[str, Any],
        ctx: RunContext,
        reusable: dict[str, tuple[float, float]],
    ) -> EnricherResult:
        """The shared full/incremental kernel (RFC-118 §4.1).

        ``reusable`` maps ``pair_key → (cosine, contradiction)`` raw scores the caller
        validated for reuse. Full == this kernel with an empty map, so the two paths
        share every line of candidate selection, scoring, threshold filtering, and
        ordering — they can only differ in which pairs hit the model.
        ``pairs_scored`` counts candidate pairs EVALUATED (reused or re-scored), so
        the output stays byte-identical between the paths (§7).
        """
        cos_threshold = float(config.get("cos_threshold", self._cos_threshold))
        contra_threshold = float(config.get("contra_threshold", self._contra_threshold))
        by_topic, person_label = _topic_insight_speaker_index(all_bundles)

        consensus: list[dict[str, Any]] = []
        pairs_scored = 0
        pairs_rescored = 0
        fresh_cache: dict[str, Any] = {}
        for tid, entries in sorted(by_topic.items()):
            usable = [(iid, pid, txt, eid) for iid, pid, txt, eid in entries if txt.strip()]
            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    iid_a, pid_a, txt_a, ep_a = usable[i]
                    iid_b, pid_b, txt_b, ep_b = usable[j]
                    if pid_a == pid_b:  # same speaker → not cross-Person corroboration
                        continue
                    if ctx.cancel_event.is_set():
                        return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                    key = f"{iid_a}{_KEY_SEP}{iid_b}"
                    hit = reusable.get(key)
                    if hit is not None:
                        cosine, contradiction = hit
                    else:
                        sig = await self._scorer.score(txt_a, txt_b)
                        cosine, contradiction = float(sig.cosine), float(sig.contradiction)
                        pairs_rescored += 1
                    pairs_scored += 1
                    fresh_cache[key] = {"c": cosine, "x": contradiction, "ea": ep_a, "eb": ep_b}
                    # Composite gate: embedding proximity (same proposition) AND low contradiction
                    # (they don't disagree). Cosine alone admits similar-but-opposite pairs; the
                    # contradiction filter removes them (ADR-108 eval).
                    if cosine < cos_threshold or contradiction > contra_threshold:
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
                            "consensus_score": round(cosine, 6),
                            "cosine": round(cosine, 6),
                            "contradiction": round(contradiction, 6),
                            "model_id": self._model_id,
                            "model_version": self._model_version,
                        }
                    )

        consensus.sort(key=lambda r: (-r["consensus_score"], r["topic_id"], r["insight_a_id"]))

        # #1208 — no-silent-fail contract; see temporal_velocity for rationale.
        partial_reason: str | None = None
        if not all_bundles:
            partial_reason = "no_bundles"
        elif pairs_scored == 0:
            partial_reason = "no_scoreable_pairs"
        elif not consensus:
            partial_reason = "no_pairs_above_threshold"
        if partial_reason is not None:
            _logger.warning(
                "topic_consensus empty output run_id=%s enricher=%s " "reason=%s pairs_scored=%d",
                ctx.run_id,
                ctx.enricher_id,
                partial_reason,
                pairs_scored,
            )
        if reusable:
            _logger.info(
                "topic_consensus incremental: %d/%d pairs re-scored (%d reused) run_id=%s",
                pairs_rescored,
                pairs_scored,
                pairs_scored - pairs_rescored,
                ctx.run_id,
            )

        # Persist raw scores for the next incremental pass. Only pairs present in the
        # CURRENT corpus are carried, so stale insight/episode ids age out naturally.
        _write_pair_cache(
            corpus_root,
            fresh_cache,
            model_id=self._model_id,
            model_version=self._model_version,
        )

        return EnricherResult(
            status=STATUS_OK,
            data={
                "model_id": self._model_id,
                "model_version": self._model_version,
                "cos_threshold": cos_threshold,
                "contra_threshold": contra_threshold,
                "pairs_scored": pairs_scored,
                "consensus": consensus,
                "partial_reason": partial_reason,
            },
            records_written=len(consensus),
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
        """Full pass: every candidate pair hits the model; the raw-score cache is rebuilt."""
        return await self._compute(
            corpus_root=corpus_root,
            all_bundles=all_bundles or [],
            config=config,
            ctx=ctx,
            reusable={},
        )

    async def enrich_incremental(
        self,
        *,
        delta: "CorpusDelta",
        prior_output: dict[str, Any] | None,
        corpus_root: Path,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Delta pass: reuse cached raw scores for pairs untouched by the delta.

        A cached pair is valid iff NEITHER endpoint episode is in
        ``changed_ids ∪ removed_ids`` (RFC-118 §4.3). The output is rebuilt from raw
        scores every time — ``prior_output`` is deliberately unused, which is what
        makes full and incremental structurally identical: only the set of model
        invocations differs, never the merge/filter logic.
        """
        invalid = set(delta.changed_ids) | set(delta.removed_ids)
        reusable: dict[str, tuple[float, float]] = {}
        if not delta.forced:
            cached = _load_pair_cache(
                corpus_root, model_id=self._model_id, model_version=self._model_version
            )
            for key, entry in cached.items():
                if not isinstance(entry, dict):
                    continue
                ea, eb = str(entry.get("ea") or ""), str(entry.get("eb") or "")
                if not ea or not eb or ea in invalid or eb in invalid:
                    continue
                try:
                    reusable[key] = (float(entry["c"]), float(entry["x"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return await self._compute(
            corpus_root=corpus_root,
            all_bundles=list(delta.all_bundles),
            config=config,
            ctx=ctx,
            reusable=reusable,
        )


__all__ = ["TopicConsensusEnricher"]
