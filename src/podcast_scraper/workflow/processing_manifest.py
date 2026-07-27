"""Per-episode processing manifest (RFC-109 / ADR-130).

A ``<transcript-base>.manifest.json`` sidecar: the source of truth for **how** an episode was
processed — per-stage provenance, quality metrics, method versions, cost, and rework flags. It
complements ``metadata.json`` (the **product** record, ADR-131), which says what the episode *is*.

The honesty rule (ADR-130 §Convention 1): **each stage writes its own block from its own result,
never from ``cfg``.** A field no stage owns is not added — that is the ``whisper_model``-from-config
rot this artifact exists to prevent.

Versioning is layered (ADR-130 §Convention 2): ``git_sha`` (+ ``git_dirty``) is the exact-code
ground-truth backstop; ``pipeline_composition_version`` is a hash of *which stages ran* (the
stage-graph shape — a re-wire like moving the ASR gate downstream of diarization changes it); and
each stage's ``method_version`` (inside its block) is the reprocess **query key** — bumped when that
stage's logic changes, so "reprocess exactly the episodes produced by the old naming logic" is a
query, not a guess.

Stages run sequentially within one episode's worker, so the writer is a read-modify-write on the
sidecar: load-or-init, set ``stages[name]``, recompute the roll-ups, write back. Episodes run in
parallel but each writes its own file, so there is no cross-episode contention.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .run_manifest import _get_git_info

logger = logging.getLogger(__name__)

# Bumped only on a breaking change to the top-level manifest shape.
MANIFEST_SCHEMA_VERSION = 1

# The reprocess query key, one per stage. Bump when a STAGE'S LOGIC changes (not its config). One
# edit here is greppable and is what makes "reprocess episodes below naming-3" expressible.
METHOD_VERSIONS: Dict[str, str] = {
    "asr": "asr-gate-1",  # ADR-129 speech-normalized coverage gate + failover
    "diarization": "diar-1",
    "naming": "naming-3",  # ADR-128 metadata name recovery + audit 2a/3
    "summary": "summary-1",
    "gi": "gi-1",
    "kg": "kg-1",
}

# Canonical stage order for the composition hash — the order stages run in the pipeline. The
# composition version is derived from the SUBSET that actually ran, in this order.
CANONICAL_STAGE_ORDER = ("asr", "diarization", "naming", "summary", "gi", "kg")

# Closed vocabulary of rework signals so the corpus ledger can GROUP BY them (ADR-130).
QUALITY_FLAGS = frozenset(
    {
        "asr_failover",
        "asr_speech_coverage_low",
        "unnamed_dominant_voice",
        "guest_in_title_not_placed",
        "empty_host_anchor",
        "gi_all_gated",
    }
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeCostProbe:
    """Per-episode cost capture for the GI/KG stages, for the processing manifest (RFC-109).

    GI/KG cost is recorded by the LLM providers (gemini/deepseek/grok/anthropic/mistral, all via the
    same ``record_llm_gi_call`` / ``record_llm_kg_call`` / ``record_llm_gi_evidence_stage_call``
    hooks) onto **run-level** accumulators on ``pipeline_metrics`` — which is shared across parallel
    episodes, so a before/after delta on it is racy. This probe wraps ``pipeline_metrics`` for the
    duration of ONE episode's GI/KG build: it forwards every attribute and method to the real object
    (so run totals and all other counters stay correct) while capturing that episode's GI/KG cost in
    isolation. Provider-agnostic — it hooks the recorder methods every provider funnels cost
    through, not any one provider.
    """

    __slots__ = ("_inner", "gi_cost_usd", "kg_cost_usd")

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "gi_cost_usd", 0.0)
        object.__setattr__(self, "kg_cost_usd", 0.0)

    def record_llm_gi_call(
        self, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        if cost_usd:
            object.__setattr__(self, "gi_cost_usd", self.gi_cost_usd + float(cost_usd))
        return self._inner.record_llm_gi_call(input_tokens, output_tokens, cost_usd=cost_usd)

    def record_llm_gi_evidence_stage_call(
        self, stage: str, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        # Captured here because the inner impl routes its own record_llm_gi_call to ``inner`` (not
        # this probe), so a single provider call hits exactly one probe method — no double count.
        if cost_usd:
            object.__setattr__(self, "gi_cost_usd", self.gi_cost_usd + float(cost_usd))
        return self._inner.record_llm_gi_evidence_stage_call(
            stage, input_tokens, output_tokens, cost_usd=cost_usd
        )

    def record_llm_kg_call(
        self, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        if cost_usd:
            object.__setattr__(self, "kg_cost_usd", self.kg_cost_usd + float(cost_usd))
        return self._inner.record_llm_kg_call(input_tokens, output_tokens, cost_usd=cost_usd)

    def __getattr__(self, name: str) -> Any:
        # Everything not overridden (counters, other recorders, attributes) forwards to the real
        # pipeline_metrics so run-level accounting is unchanged.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Attribute writes (e.g. ``pm.gi_evidence_extract_quotes_calls += 1`` inside the builders)
        # must land on the real object, not this wrapper.
        setattr(object.__getattribute__(self, "_inner"), name, value)


def manifest_path(effective_output_dir: str, rel_transcript_path: str) -> str:
    """``<transcript-base>.manifest.json`` next to the transcript."""
    full = os.path.join(effective_output_dir, rel_transcript_path)
    base, _ = os.path.splitext(full)
    return base + ".manifest.json"


def git_ground_truth() -> Dict[str, Any]:
    """The exact-code backstop: short git SHA + dirty flag (ADR-130), via the run-manifest probe."""
    commit_sha, _branch, dirty = _get_git_info()
    return {"git_sha": (commit_sha[:7] if commit_sha else None), "git_dirty": bool(dirty)}


def pipeline_composition_version(stage_names: Iterable[str]) -> str:
    """Short deterministic hash of *which stages ran*, in canonical order — the stage-graph shape.

    Changes when a stage is added/removed/re-wired, which no single stage's ``method_version``
    captures. Independent of ``git_sha`` (which moves on every commit, including docs).
    """
    present = [s for s in CANONICAL_STAGE_ORDER if s in set(stage_names)]
    digest = hashlib.sha1(",".join(present).encode("utf-8")).hexdigest()[:8]
    return f"pc-{digest}"


def stage_block(
    *,
    ran: bool,
    method: Optional[str] = None,
    model: Optional[str] = None,
    method_version: Optional[str] = None,
    duration_s: Optional[float] = None,
    cost_usd: Optional[float] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    failover: Any = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build one stage's block in the common shape, dropping keys the stage did not set.

    ``ran`` is always present; every other key is included only when the owning stage supplies it,
    so absence is honest (nobody owns it) rather than a config-derived default.
    """
    block: Dict[str, Any] = {"ran": bool(ran)}
    if method is not None:
        block["method"] = method
    if model is not None:
        block["model"] = model
    if method_version is not None:
        block["method_version"] = method_version
    if duration_s is not None:
        block["duration_s"] = round(float(duration_s), 3)
    if cost_usd is not None:
        block["cost_usd"] = round(float(cost_usd), 6)
    if metrics:
        block["metrics"] = {k: v for k, v in metrics.items() if v is not None}
    if failover is not None:
        block["failover"] = failover
    if warnings:
        block["warnings"] = list(warnings)
    return block


def _sum_cost(stages: Mapping[str, Any]) -> float:
    total = 0.0
    for blk in stages.values():
        if isinstance(blk, dict) and isinstance(blk.get("cost_usd"), (int, float)):
            total += float(blk["cost_usd"])
    return round(total, 6)


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _init_manifest(
    episode_id: Optional[str], feed_id: Optional[str], run_id: Optional[str]
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "feed_id": feed_id,
        "run_id": run_id,
        "generated_at": _utcnow(),
        "stages": {},
        "quality_flags": [],
        "cost_usd_total": 0.0,
    }
    manifest.update(git_ground_truth())
    return manifest


def update_stage(
    effective_output_dir: str,
    rel_transcript_path: str,
    stage: str,
    block: Mapping[str, Any],
    *,
    episode_id: Optional[str] = None,
    feed_id: Optional[str] = None,
    run_id: Optional[str] = None,
    quality_flags: Sequence[str] = (),
) -> Optional[str]:
    """Read-modify-write the episode manifest: set ``stages[stage]``, merge flags, roll up cost.

    Called by the stage that produced ``block`` (stage ownership). Never raises into the pipeline —
    a manifest write failing must not lose the episode. Returns the path written, or ``None``.
    """
    if not rel_transcript_path or not stage:
        return None
    path = manifest_path(effective_output_dir, rel_transcript_path)
    data = _load(path) or _init_manifest(episode_id, feed_id, run_id)
    # Backfill identity if a later stage learns it and an earlier init left it blank.
    for key, val in (("episode_id", episode_id), ("feed_id", feed_id), ("run_id", run_id)):
        if val and not data.get(key):
            data[key] = val
    stages = data.setdefault("stages", {})
    stages[stage] = dict(block)
    if quality_flags:
        existing = data.setdefault("quality_flags", [])
        for flag in quality_flags:
            if flag and flag not in existing:
                if flag not in QUALITY_FLAGS:
                    logger.debug("processing_manifest: out-of-vocab quality flag %r", flag)
                existing.append(flag)
    data["cost_usd_total"] = _sum_cost(stages)
    data["pipeline_composition_version"] = pipeline_composition_version(stages.keys())
    data["updated_at"] = _utcnow()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False, sort_keys=False)
        logger.debug("processing_manifest: wrote %s (stage=%s)", path, stage)
        return path
    except OSError as exc:
        logger.warning("processing_manifest: could not write %s: %s", path, exc)
        return None
