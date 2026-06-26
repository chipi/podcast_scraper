"""Enricher protocol — the canonical interface for the enrichment layer.

Amends RFC-088 §Enricher Protocol (was: sync ``def enrich(...) -> dict``):
the protocol is now ``async``, returns a typed ``EnricherResult``, and
receives a ``RunContext`` correlation envelope. Sync deterministic bodies
use the ``@sync_enricher`` decorator to run in the default thread executor.

See ``docs/rfc/RFC-088-enrichment-layer-architecture.md`` §Enricher Protocol
for the amended protocol shape. Audit cross-ref:
``docs/wip/RFC-088-CHUNK1-LOCK-AUDIT.md`` §B1 (async amendment) + §B2
(EnricherResult definition).
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable


class EnricherScope(Enum):
    EPISODE = "episode"
    CORPUS = "corpus"


class EnricherTier(Enum):
    DETERMINISTIC = "deterministic"
    EMBEDDING = "embedding"
    ML = "ml"
    LLM = "llm"


# Canonical status vocabulary for ``EnricherResult.status``.
# Kept as string constants (not an enum) so JSONL events and metrics
# record stable strings without enum serialization concerns.
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_QUARANTINED = "quarantined"
STATUS_CANCELLED = "cancelled"
STATUS_SKIPPED = "skipped"

ALL_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_OK,
        STATUS_FAILED,
        STATUS_TIMEOUT,
        STATUS_QUARANTINED,
        STATUS_CANCELLED,
        STATUS_SKIPPED,
    }
)


@dataclass(frozen=True)
class EpisodeArtifactBundle:
    """Resolved paths to core artifacts for one episode."""

    metadata_path: Path  # e.g. metadata/0001 - ep.metadata.json
    gi_path: Path | None  # e.g. metadata/0001 - ep.gi.json
    kg_path: Path | None  # e.g. metadata/0001 - ep.kg.json
    bridge_path: Path | None  # e.g. metadata/0001 - ep.bridge.json
    episode_id: str  # raw GUID or sha256:... (no "episode:" prefix)
    stem: str  # shared filename stem, e.g. "0001 - ep"


@dataclass(frozen=True)
class EnricherManifest:
    """Declares an enricher's inputs, outputs, tier, scope, and cost caps."""

    id: str  # e.g. "topic_cooccurrence"
    version: str  # semver e.g. "1.0.0"
    scope: EnricherScope
    tier: EnricherTier
    reads: list[str]  # artifact suffixes consumed, e.g. [".kg.json", ".bridge.json"]
    writes: str  # output filename, e.g. "topic_cooccurrence.json"
    description: str
    requires_opt_in: bool = False  # True forces double-opt-in for LLM tier
    # Cost-cap fields (O1 decision): per-enricher soft budget; exceeded
    # → quarantine that enricher only. ``None`` means unbounded.
    max_cost_usd_per_run: float | None = None
    # Expected wall-clock seconds (per episode for EPISODE scope, per
    # corpus for CORPUS scope). Used by the heartbeat watchdog to
    # detect stalls; ``None`` disables.
    expected_duration_s: int | None = None


@dataclass(frozen=True)
class RunContext:
    """Correlation envelope passed into every ``enrich()`` call.

    Threads ``run_id`` / ``enricher_id`` / ``tier`` / ``attempt`` through
    every emit surface (Sentry tags, Langfuse metadata, Loki structured
    extras, JSONL events, run_summary, status, health) so
    ``prod_correlate(run_id)`` returns one consistent story across
    pipeline + enrichment + LLM calls.
    """

    run_id: str
    parent_run_id: str | None  # pipeline run_id if attached; None standalone
    enricher_id: str
    enricher_version: str
    tier: str  # serialized ``EnricherTier``
    attempt: int  # 1-indexed; bumps on retry
    job_id: str  # jobs-API record id (== run_id for standalone enrichment)
    # Cooperative cancel signal shared across parallel enrichers in
    # the same run. Long-running enricher bodies check
    # ``ctx.cancel_event.is_set()`` between batches and bail with
    # status == "cancelled" + partial-output preservation.
    cancel_event: asyncio.Event


@dataclass(frozen=True)
class EnricherResult:
    """Canonical return shape — ``enrich()`` never raises out of itself.

    The executor reads ``status`` to drive the state machine; the
    envelope writer reads ``data`` to populate the on-disk file when
    ``status == "ok"``.
    """

    status: str
    data: dict[str, Any] | None = None
    error: str | None = None
    error_class: str | None = None
    retry_count: int = 0
    circuit_state: str | None = None
    duration_ms: int = 0
    records_written: int = 0

    def __post_init__(self) -> None:
        if self.status not in ALL_STATUSES:
            raise ValueError(
                f"EnricherResult.status must be one of {sorted(ALL_STATUSES)}, "
                f"got {self.status!r}"
            )
        if self.status == STATUS_OK and self.data is None:
            raise ValueError("EnricherResult.status=='ok' requires non-None data")


@runtime_checkable
class Enricher(Protocol):
    """The canonical async enricher protocol.

    PEP 544 structural typing + ``@runtime_checkable``; consistent with
    the existing provider protocols (ADR-020). Single-method, stateless
    units with no lifecycle.
    """

    @property
    def manifest(self) -> EnricherManifest:
        """Declare inputs, outputs, tier, and scope."""
        ...

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,  # set for EPISODE scope
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,  # set for CORPUS scope
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Compute enrichment. Return an ``EnricherResult`` — never raise.

        Catch exceptions internally and return ``result.status ==
        "failed"`` with ``error`` / ``error_class`` set. Never modify
        any core artifact file.

        Long-running bodies MUST check ``ctx.cancel_event.is_set()``
        between batches and bail cleanly with ``status == "cancelled"``
        + partial-output preservation.

        For sync deterministic bodies, decorate the implementation with
        ``@sync_enricher`` — the decorator runs the function in the
        default thread executor and wraps the return in
        ``EnricherResult``.
        """
        ...


def sync_enricher(
    func: Callable[..., Any],
) -> Callable[..., Coroutine[Any, Any, EnricherResult]]:
    """Decorator: wrap a sync enricher body to satisfy the async protocol.

    Runs the sync body via ``asyncio.to_thread`` so the executor's
    async machinery flows uninterrupted. Sync deterministic enrichers
    use this so authors don't pay the async ceremony tax.

    Behaviour:

    * If the wrapped function returns a ``dict``, the decorator wraps
      it in ``EnricherResult(status="ok", data=...)``.
    * If the wrapped function returns an ``EnricherResult``, it's
      passed through unchanged.
    * If the wrapped function raises, the decorator returns
      ``EnricherResult(status="failed", error=str(exc),
      error_class=type(exc).__name__)`` — matches the executor's
      safety-net contract.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> EnricherResult:
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            return EnricherResult(
                status=STATUS_FAILED,
                error=str(exc),
                error_class=type(exc).__name__,
            )
        if isinstance(result, EnricherResult):
            return result
        if isinstance(result, dict):
            return EnricherResult(status=STATUS_OK, data=result)
        raise TypeError(
            "@sync_enricher: function must return dict or EnricherResult, "
            f"got {type(result).__name__}"
        )

    return wrapper


@dataclass
class EnricherSet:
    """Minimal stub — chunk 7 extends with profile-preset wiring.

    Defines which enrichers are active for a given run and their
    per-enricher config overrides. Chunk 1 ships this stub so the
    executor has something to consume via the no-op path; chunk 7
    wires ``ProfilePreset`` → ``EnricherSet`` construction with the
    profile-preset matrix from RFC-088 plan §"Profile-preset matrix".
    """

    enabled_enrichers: list[str] = field(default_factory=list)
    per_enricher_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    opt_in_flags: dict[str, bool] = field(default_factory=dict)

    def is_enabled(self, enricher_id: str) -> bool:
        """True when this enricher should run in this set."""
        return enricher_id in self.enabled_enrichers

    def get_config(self, enricher_id: str) -> dict[str, Any]:
        """Per-enricher config override (empty dict when no override)."""
        return self.per_enricher_config.get(enricher_id, {})

    def has_opt_in(self, enricher_id: str) -> bool:
        """True when the LLM-tier opt-in flag is set for this enricher."""
        return bool(self.opt_in_flags.get(enricher_id, False))
