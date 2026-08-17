"""Per-episode processing manifest (RFC-109 / ADR-132).

A ``<transcript-base>.manifest.json`` sidecar: the source of truth for **how** an episode was
processed — per-stage provenance, quality metrics, method versions, cost, and rework flags. It
complements ``metadata.json`` (the **product** record, ADR-133), which says what the episode *is*.

The honesty rule (ADR-132 §Convention 1): **each stage writes its own block from its own result,
never from ``cfg``.** A field no stage owns is not added — that is the ``whisper_model``-from-config
rot this artifact exists to prevent.

Versioning is layered (ADR-132 §Convention 2): ``git_sha`` (+ ``git_dirty``) is the exact-code
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
    "asr": "asr-gate-1",  # ADR-131 speech-normalized coverage gate + failover
    "diarization": "diar-1",
    "naming": "naming-4",  # ADR-139 text-normalization contract: narrated-desk cue vocab +
    # case-blind metadata-anchored self-intro + nickname/ASR-fuzzy binding + org-form reject + "my
    # name is" discovery + Pattern-B (bounded unknown-vs-tape classification, defect-share alarm)
    "summary": "summary-1",
    "gi": "gi-1",
    "kg": "kg-1",
}

# Canonical stage order for the composition hash — the order stages run in the pipeline. The
# composition version is derived from the SUBSET that actually ran, in this order.
CANONICAL_STAGE_ORDER = ("asr", "diarization", "naming", "summary", "gi", "kg")

# Closed vocabulary of rework signals so the corpus ledger can GROUP BY them (ADR-132).
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
    """Per-episode cost capture for the summary/GI/KG stages, for the processing manifest (RFC-109).

    Summary/GI/KG cost is recorded by the LLM providers (gemini/deepseek/grok/anthropic/mistral, all
    via the same ``record_llm_summarization_call`` / ``record_llm_gi_call`` / ``record_llm_kg_call``
    / ``record_llm_gi_evidence_stage_call`` hooks) onto **run-level** accumulators on
    ``pipeline_metrics`` — which is shared across parallel episodes, so a before/after delta on it
    is racy. This probe wraps ``pipeline_metrics`` for the duration of ONE episode's LLM stages: it
    forwards every attribute and method to the real object (so run totals and all other counters
    stay correct) while capturing that episode's summary/GI/KG cost in isolation. Provider-agnostic:
    hooks the recorder methods every provider funnels cost through, not any one provider.
    """

    __slots__ = (
        "_inner",
        "summary_cost_usd",
        "gi_cost_usd",
        "kg_cost_usd",
        "speaker_detection_cost_usd",
    )

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "summary_cost_usd", 0.0)
        object.__setattr__(self, "gi_cost_usd", 0.0)
        object.__setattr__(self, "kg_cost_usd", 0.0)
        object.__setattr__(self, "speaker_detection_cost_usd", 0.0)

    def record_llm_speaker_detection_call(
        self, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        """Accumulate this episode's speaker-detection cost, then forward to the real recorder.

        Added after the #1657 acceptance run showed ``naming.cost_usd: null`` on every episode
        while ``llm_speaker_detection_cost_usd`` was accruing at run level. Speaker naming is
        NOT always free — ``speaker_detector_provider: litellm`` in ``cloud_balanced`` resolves
        voices with a real LLM call — so the manifest was under-reporting every episode's true
        cost by exactly that amount, and ``cost_usd_total`` inherited the gap.
        """
        if cost_usd:
            object.__setattr__(
                self,
                "speaker_detection_cost_usd",
                self.speaker_detection_cost_usd + float(cost_usd),
            )
        return self._inner.record_llm_speaker_detection_call(
            input_tokens, output_tokens, cost_usd=cost_usd
        )

    def record_llm_summarization_call(
        self, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        """Accumulate this episode's summary cost, then forward to the real recorder."""
        # Chunked summarization makes several calls per episode; each provider funnels through here,
        # so accumulating captures the full per-episode summary cost (the ProviderCallMetrics object
        # only carries token counts, never the priced cost — the block was None without it).
        if cost_usd:
            object.__setattr__(self, "summary_cost_usd", self.summary_cost_usd + float(cost_usd))
        return self._inner.record_llm_summarization_call(
            input_tokens, output_tokens, cost_usd=cost_usd
        )

    def record_llm_gi_call(
        self, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        """Accumulate this episode's GI cost, then forward to the real recorder."""
        if cost_usd:
            object.__setattr__(self, "gi_cost_usd", self.gi_cost_usd + float(cost_usd))
        return self._inner.record_llm_gi_call(input_tokens, output_tokens, cost_usd=cost_usd)

    def record_llm_gi_evidence_stage_call(
        self, stage: str, input_tokens: int, output_tokens: int, cost_usd: Optional[float] = None
    ) -> Any:
        """Accumulate this episode's GI evidence-stage cost, then forward to the real recorder."""
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
        """Accumulate this episode's KG cost, then forward to the real recorder."""
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
    """The exact-code backstop: short git SHA + dirty flag (ADR-132), via the run-manifest probe.

    The probe captures once per process, so every episode in a run records the commit that was
    on disk when the process started — not whatever HEAD drifted to by the time this particular
    manifest got written. See ``_get_git_info`` for why that distinction matters.
    """
    commit_sha, _branch, dirty = _get_git_info()
    return {"git_sha": (commit_sha[:7] if commit_sha else None), "git_dirty": bool(dirty)}


def pipeline_composition_version(stage_names: Iterable[str]) -> str:
    """Short deterministic hash of *which stages ran*, in canonical order — the stage-graph shape.

    Changes when a stage is added/removed/re-wired, which no single stage's ``method_version``
    captures. Independent of ``git_sha`` (which moves on every commit, including docs).
    """
    present = [s for s in CANONICAL_STAGE_ORDER if s in set(stage_names)]
    # Not a security hash — a short stable fingerprint of the stage-graph shape.
    digest = hashlib.sha1(",".join(present).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
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

    ``cost_usd`` is the exception and is ALWAYS present, because cost has three states and the
    other two are already spoken for: ``0.0`` means measured and free, a number means measured,
    and ``null`` means not measured. Dropping the key added a fourth, silent state that readers
    could only guess at — and it read as "free" to anyone summing the blocks. That is exactly how
    naming's cost went unnoticed on every episode of the acceptance run while the run-level
    counter kept climbing, and how ``cost_usd_total`` inherited the gap.
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
    block["cost_usd"] = None if cost_usd is None else round(float(cost_usd), 6)
    if metrics:
        block["metrics"] = {k: v for k, v in metrics.items() if v is not None}
    if failover is not None:
        block["failover"] = failover
    if warnings:
        block["warnings"] = list(warnings)
    return block


#: Transcription providers that run on hardware we own — no invoice, so a measured zero.
LOCAL_TRANSCRIPTION_PROVIDERS = frozenset(
    {"whisper", "local", "tailnet_dgx_whisper", "dgx", "moss", "faster_whisper"}
)
#: Diarization providers that run locally — same reasoning.
LOCAL_DIARIZATION_PROVIDERS = frozenset({"local", "pyannote", "tailnet_dgx", "dgx", "moss"})


def measured_or_unmeasured(
    measured: Optional[float], provider: Optional[str], local_providers: "frozenset[str]"
) -> Optional[float]:
    """Resolve a stage's ``cost_usd`` under one rule, applied identically everywhere.

    **``0.0`` means measured and free. ``None`` means nobody measured it.** They are different
    facts and the manifest must not blur them — a fabricated zero on an uninstrumented stage is
    how a cost roll-up silently under-reports, and a ``null`` on a stage that genuinely cost
    nothing reads as "we don't know" when we do.

    Before this, the same situation produced different answers: a locally-diarized episode
    recorded ``diarization.cost_usd: 0.0`` while ``naming.cost_usd`` was ``null``, though both
    ran locally and both were free. ADR-132 had specified ``None`` for local diarization; the
    code did the opposite. The rule below is the corrected intent — measured-and-free wins,
    because it carries information, and the ``null`` that remains then means something.
    """
    if measured is not None:
        return float(measured)
    if (provider or "").strip().lower() in local_providers:
        return 0.0
    return None


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
    # git_sha is the exact-code backstop for the code that PRODUCED the manifest's current state
    # (ADR-132). On a reprocess/relabel the manifest file already exists, so _init_manifest is
    # skipped and the loaded git_sha is the ORIGINAL build's — which then rides into every
    # pipeline_stage event (o11y showed the ancestor sha, not the re-running commit). Refresh it to
    # the code writing NOW; per-stage code provenance is carried by each stage's method_version.
    data.update(git_ground_truth())
    # OVERWRITE identity when the caller supplies it (not backfill): a re-run over an existing
    # corpus must not inherit the PREVIOUS run's run_id from the stale manifest file (advisor #3).
    for key, val in (("episode_id", episode_id), ("feed_id", feed_id), ("run_id", run_id)):
        if val:
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
    except OSError as exc:
        logger.warning("processing_manifest: could not write %s: %s", path, exc)
        return None
    _emit_stage_event(stage, dict(block), data, list(quality_flags or ()))
    return path


def _emit_stage_event(
    stage: str, block: Dict[str, Any], data: Mapping[str, Any], flags: Sequence[str]
) -> None:
    """Emit one canonical ``pipeline_stage`` observability event per stage write (RFC-109 §4).

    The manifest is a per-episode FILE; this is its backend sink. Each stage write emits a
    ``pipeline_stage`` event through the vendor-neutral ``emit_event`` (``sink="log"`` → stdout →
    Alloy → VictoriaLogs), correlated by ``episode_id``/``run_id`` so the per-episode quality + cost
    signal is queryable in Grafana alongside logs/traces/cost — not a dead sidecar. Best-effort:
    ``emit_event`` never raises, and this whole call is additionally guarded.
    """
    try:
        from ..obs.events import emit_event

        emit_event(
            "pipeline_stage",
            stage=stage,
            episode_id=data.get("episode_id"),
            feed_id=data.get("feed_id"),
            run_id=data.get("run_id"),
            git_sha=data.get("git_sha"),
            # NB: pipeline_composition_version is deliberately NOT emitted per-stage — it is
            # recomputed on every RMW, so it would ship a different (partial) value on each of an
            # episode's stage events and poison any GROUP BY (advisor #7). File-only.
            ran=block.get("ran"),
            method=block.get("method"),
            model=block.get("model"),
            method_version=block.get("method_version"),
            duration_s=block.get("duration_s"),
            cost_usd=block.get("cost_usd"),
            metrics=block.get("metrics") or None,
            quality_flags=list(flags) or None,
        )
    except Exception:  # noqa: BLE001 — telemetry must never break the manifest write
        logger.debug("processing_manifest: pipeline_stage emit failed", exc_info=True)
