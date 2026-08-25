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
        # 3.0.0 (#1817): batched/budgeted redesign — bulk embeddings + matrix cosine
        # gate + ONE batched NLI pass over gate-passing pairs bounded by per-topic and
        # per-run budgets, with partial cache flushes so interruption never discards
        # completed work. Selection semantics changed (budget can drop lowest-cosine
        # tail pairs), hence the major bump; model_version v2 -> v3 discards the old
        # pair cache wholesale.
        version="3.0.0",
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
                "max_nli_pairs_per_topic": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 200,
                    "description": (
                        "#1817 NLI budget: at most this many gate-passing pairs per topic "
                        "reach the cross-encoder, highest cosine first. Dropped pairs are "
                        "counted in pairs_nli_budget_dropped, never silent."
                    ),
                },
                "max_nli_pairs_per_run": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 20000,
                    "description": (
                        "#1817 global NLI budget per run — the hard bound that keeps the "
                        "baseline O(topics), not O(pairs^2), at any corpus size."
                    ),
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
        model_version: str = "v3",
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
        reusable: dict[str, tuple[float, float | None]],
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
        max_nli_per_topic = int(config.get("max_nli_pairs_per_topic", 200))
        max_nli_per_run = int(config.get("max_nli_pairs_per_run", 20000))
        by_topic, person_label = _topic_insight_speaker_index(all_bundles)

        # ---- Candidate enumeration (shared by both scoring paths) -------------------
        # One flat, deterministically-ordered list; the 2026-08-24 incident measured
        # 467,835 candidates at 678 episodes, so everything downstream must be
        # batched and budgeted, never per-pair model calls (#1817).
        candidates: list[tuple[str, str, tuple[str, str, str, str], tuple[str, str, str, str]]] = []
        for tid, entries in sorted(by_topic.items()):
            usable = [(iid, pid, txt, eid) for iid, pid, txt, eid in entries if txt.strip()]
            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    if usable[i][1] == usable[j][1]:  # same speaker → not cross-Person
                        continue
                    key = f"{usable[i][0]}{_KEY_SEP}{usable[j][0]}"
                    candidates.append((tid, key, usable[i], usable[j]))

        consensus: list[dict[str, Any]] = []
        pairs_scored = 0
        pairs_rescored = 0
        pairs_nli_budget_dropped = 0
        fresh_cache: dict[str, Any] = {}
        # pair_key -> (cosine, contradiction|None). None = cosine-gated out, NLI never
        # ran; reusable ONLY while the pair stays below the current cosine gate.
        scored: dict[str, tuple[float, float | None]] = {}

        batch_capable = bool(getattr(self._scorer, "supports_batch", False))
        to_score = []
        for cand in candidates:
            hit = reusable.get(cand[1])
            if hit is None:
                to_score.append(cand)
                continue
            # A cached (cosine, None) means NLI never ran because the pair sat below
            # the cosine gate (or was budget-dropped) at cache time. That reuse is
            # only valid while the pair STILL fails the current gate — if the
            # operator lowered cos_threshold since, the pair now needs real NLI.
            if hit[1] is None and hit[0] >= cos_threshold:
                to_score.append(cand)
                continue
            scored[cand[1]] = hit
        if to_score and batch_capable:
            dropped = await self._score_batched(
                to_score=to_score,
                candidates=candidates,
                scored=scored,
                cos_threshold=cos_threshold,
                max_nli_per_topic=max_nli_per_topic,
                max_nli_per_run=max_nli_per_run,
                corpus_root=corpus_root,
                ctx=ctx,
            )
            if dropped is None:
                return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
            pairs_nli_budget_dropped = dropped
            pairs_rescored = len(to_score)
        elif to_score:
            # ---- Legacy path: score()-only providers (fixtures, custom backends) ----
            for cand in to_score:
                if ctx.cancel_event.is_set():
                    return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                sig = await self._scorer.score(cand[2][2], cand[3][2])
                scored[cand[1]] = (float(sig.cosine), float(sig.contradiction))
                pairs_rescored += 1

        # ---- Shared gating + output (identical for full/incremental — §7) -----------
        for cand in candidates:
            tid, key, (iid_a, pid_a, txt_a, ep_a), (iid_b, pid_b, txt_b, ep_b) = cand
            entry = scored.get(key)
            if entry is None:
                continue
            cosine, contradiction = entry
            pairs_scored += 1
            fresh_cache[key] = {"c": cosine, "x": contradiction, "ea": ep_a, "eb": ep_b}
            # Composite gate: embedding proximity (same proposition) AND low contradiction
            # (they don't disagree). Cosine alone admits similar-but-opposite pairs; the
            # contradiction filter removes them (ADR-108 eval). contradiction=None means
            # the pair never reached NLI (cosine-gated or budget-dropped) → not consensus.
            if contradiction is None or cosine < cos_threshold or contradiction > contra_threshold:
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
        if pairs_nli_budget_dropped:
            _logger.warning(
                "topic_consensus: NLI budget dropped %d gate-passing pair(s) this run "
                "(max_nli_pairs_per_topic/max_nli_pairs_per_run) — raise the budgets to "
                "cover them; they are cached with contradiction=None, run_id=%s",
                pairs_nli_budget_dropped,
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
                # §7 identity-safe: derived from the FINAL cache state (identical for
                # full and incremental), NOT from this run's to_score set. A pair with
                # cosine above the gate but no contradiction means NLI never ran for
                # it — cosine-budget-dropped. per-run rescored counts are LOG-ONLY
                # (they differ between the paths by design).
                "pairs_nli_pending": sum(
                    1 for v in fresh_cache.values() if v["x"] is None and v["c"] >= cos_threshold
                ),
                "consensus": consensus,
                "partial_reason": partial_reason,
            },
            records_written=len(consensus),
        )

    async def _score_batched(
        self,
        *,
        to_score: list[Any],
        candidates: list[Any],
        scored: dict[str, tuple[float, float | None]],
        cos_threshold: float,
        max_nli_per_topic: int,
        max_nli_per_run: int,
        corpus_root: Path,
        ctx: RunContext,
    ) -> int | None:
        """Batch path (#1817): bulk embed -> matrix cosine -> budgeted batched NLI.

        Mutates ``scored`` in place; returns the budget-dropped count, or None on
        cancellation. Determinism: budget selection ranks by (-cosine, pair_key),
        so full and incremental runs pick identical NLI sets (§7).
        """
        texts = sorted({e[2] for c in to_score for e in (c[2], c[3])})
        if ctx.cancel_event.is_set():
            return None
        vec_map = await self._scorer.embed_texts_batch(texts)  # type: ignore[attr-defined]
        import numpy as np

        order = {t: i for i, t in enumerate(texts)}
        dim = max((len(vec_map.get(t) or []) for t in texts), default=1)
        mat = np.zeros((len(texts), dim), dtype=np.float64)
        for t, i in order.items():
            v = vec_map.get(t) or []
            if len(v) == dim:  # ragged/failed embeddings stay zero → cosine 0.0
                mat[i] = v
        norms = np.linalg.norm(mat, axis=1)
        safe = norms.copy()
        safe[safe == 0.0] = 1.0
        normed = mat / safe[:, None]
        normed[norms == 0.0] = 0.0
        ia = np.asarray([order[c[2][2]] for c in to_score])
        ib = np.asarray([order[c[3][2]] for c in to_score])
        cosines = np.einsum("ij,ij->i", normed[ia], normed[ib])
        if ctx.cancel_event.is_set():
            return None

        # Budget the NLI set deterministically: gate by cosine, rank per topic by
        # (-cosine, key), cap per topic then globally. Dropped pairs are counted
        # loudly and cached with contradiction=None (no silent truncation).
        dropped = 0
        per_topic: dict[str, list[int]] = {}
        for idx, cand in enumerate(to_score):
            if float(cosines[idx]) >= cos_threshold:
                per_topic.setdefault(cand[0], []).append(idx)
        nli_idx: list[int] = []
        for tid in sorted(per_topic):
            ranked = sorted(per_topic[tid], key=lambda k: (-float(cosines[k]), to_score[k][1]))
            kept = ranked[:max_nli_per_topic]
            dropped += len(ranked) - len(kept)
            nli_idx.extend(kept)
        if len(nli_idx) > max_nli_per_run:
            nli_idx.sort(key=lambda k: (-float(cosines[k]), to_score[k][1]))
            dropped += len(nli_idx) - max_nli_per_run
            nli_idx = nli_idx[:max_nli_per_run]
        nli_idx.sort()

        _NLI_CHUNK = 512
        contradiction_by_idx: dict[int, float] = {}
        for start in range(0, len(nli_idx), _NLI_CHUNK):
            if ctx.cancel_event.is_set():
                return None
            chunk = nli_idx[start : start + _NLI_CHUNK]
            pairs_txt = [(to_score[k][2][2], to_score[k][3][2]) for k in chunk]
            contras = await self._scorer.contradictions_batch(  # type: ignore[attr-defined]
                pairs_txt
            )
            for k, x in zip(chunk, contras):
                contradiction_by_idx[k] = float(x)
            # Partial persistence (#1817): a timeout/crash must never discard a
            # completed prefix again — flush the raw scores accumulated so far.
            for k in chunk:
                cand = to_score[k]
                scored[cand[1]] = (float(cosines[k]), contradiction_by_idx[k])
            self._flush_partial_cache(corpus_root, candidates, scored)
        for idx, cand in enumerate(to_score):
            if cand[1] in scored:
                continue
            scored[cand[1]] = (float(cosines[idx]), contradiction_by_idx.get(idx))
        return dropped

    def _flush_partial_cache(
        self,
        corpus_root: Path,
        candidates: list[Any],
        scored: dict[str, tuple[float, float | None]],
    ) -> None:
        """#1817 partial persistence — flush raw scores accumulated so far.

        Every prior timeout discarded 100%% of completed work (three times on
        2026-08-24 prod, ~6 CPU-hours for zero bytes) because the cache wrote only
        at the end. Flushing between NLI chunks caps the loss at one chunk. Same
        atomic tmp+rename writer; failure is non-fatal (it is a cache).
        """
        partial = {
            cand[1]: {
                "c": scored[cand[1]][0],
                "x": scored[cand[1]][1],
                "ea": cand[2][3],
                "eb": cand[3][3],
            }
            for cand in candidates
            if cand[1] in scored
        }
        _write_pair_cache(
            corpus_root,
            partial,
            model_id=self._model_id,
            model_version=self._model_version,
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
        reusable: dict[str, tuple[float, float | None]] = {}
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
                    # ``x`` may be None (#1817): the pair sat below the cosine gate at
                    # cache time so NLI never ran. The kernel re-validates that reuse
                    # against the CURRENT threshold and re-scores when it no longer holds.
                    x_raw = entry["x"]
                    reusable[key] = (
                        float(entry["c"]),
                        None if x_raw is None else float(x_raw),
                    )
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
