"""Enrichment executor — asyncio two-phase pass with the full resilience model.

Ties together every other module:

* ``protocol`` — Enricher / RunContext / EnricherResult.
* ``registry`` — which enrichers to run.
* ``resilience`` — retry, circuit, cost-cap, failure taxonomy.
* ``health`` — cross-run state + auto-disable gating.
* ``metrics`` — per-enricher counter recording.
* ``status`` — live ``.viewer/enrichment_status.json`` writer.
* ``events`` — JSONL event emission to ``enrichments/run.jsonl``.
* ``run_summary`` — final ``enrichments/run_summary.json``.
* ``envelope`` — wrap each result into the on-disk envelope.
* ``paths`` — multi-feed-aware output paths.
* ``observability`` — Sentry breadcrumbs + Langfuse helpers.

Phase 1 runs episode-scope enrichers concurrent up to the tier
concurrency cap. Phase 2 runs corpus-scope enrichers (typically
sequentially for ml-tier; concurrent for deterministic). Each enricher
call goes through a retry-with-backoff loop, classified by the
``resilience.classify_failure`` taxonomy. The executor enforces
per-enricher and run-wide cost caps (REPLAN-O7).

Failure outcomes never raise out of an enricher — every call returns
an ``EnricherResult``. The executor's own bugs are caught by the
safety net (``try/except Exception``) and turned into
``status="failed"`` results with ``error_class="<ExecutorBug>"`` so
the pipeline-attached path stays robust.

See ``docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md``
§"Resilience model" and §"Async execution + concurrency caps".
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.envelope import build_envelope, utc_iso_now
from podcast_scraper.enrichment.events import (
    append_event,
    build_auto_disabled,
    build_cancelled,
    build_circuit_opened,
    build_enricher_completed,
    build_enricher_retry,
    build_enricher_started,
    build_run_completed,
    build_run_skipped,
    build_run_started,
    build_stall_warning,
)
from podcast_scraper.enrichment.health import HealthRegistry
from podcast_scraper.enrichment.metrics import EnrichmentMetrics, new_metrics_for
from podcast_scraper.enrichment.observability import (
    breadcrumb_circuit_opened,
    message_auto_disabled,
    stamp_sentry_correlation,
)
from podcast_scraper.enrichment.paths import (
    corpus_enrichment_path,
    ensure_directory,
    episode_enrichment_path,
)
from podcast_scraper.enrichment.protocol import (
    Enricher,
    EnricherResult,
    EnricherScope,
    EnricherSet,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_QUARANTINED,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.resilience import (
    classify_failure,
    compute_backoff,
    CostCapState,
    EnricherCircuitState,
    HeartbeatWatchdog,
    policy_for,
    RetryClass,
    status_for_exception,
)
from podcast_scraper.enrichment.run_summary import build_run_summary, write_run_summary
from podcast_scraper.enrichment.status import write_idle, write_status

logger = logging.getLogger(__name__)


# Default timeouts per scope (per chunk-1 plan body).
DEFAULT_EPISODE_TIMEOUT_S = 60
DEFAULT_CORPUS_TIMEOUT_S = 600


@dataclass
class ExecutorOptions:
    """Operator-facing knobs for one executor invocation."""

    only: list[str] | None = None  # ``--only id,id`` CLI filter
    skip: list[str] | None = None  # ``--skip id,id`` CLI filter
    corpus_only: bool = False  # Skip episode-scope enrichers
    # Cost-cap fields (O1).
    max_total_cost_usd_per_run: float | None = None
    fail_on_run_cost_cap: bool = True
    # Output paths (when None, defaults derived from corpus_root).
    jsonl_events_path: Path | None = None
    enricher_schema_version: str = "1.0"
    profile: str | None = None


@dataclass
class EnrichmentRunResult:
    """Summary returned by ``EnrichmentExecutor.run``."""

    run_id: str
    status: str  # "ok" | "failed" | "cancelled" | "skipped"
    duration_ms: int
    per_enricher_metrics: dict[str, EnrichmentMetrics] = field(default_factory=dict)
    run_summary: dict[str, Any] = field(default_factory=dict)


class EnrichmentExecutor:
    """Runs an ``EnricherSet`` over a corpus — async, resilient, observed.

    Construction is cheap (no I/O). ``run(...)`` opens the JSONL file,
    loads health, schedules the two phases, writes the run-summary,
    and persists health at the end.

    The executor is **stateless** between runs — every ``run(...)``
    call creates fresh per-enricher metrics and circuit state.
    """

    def __init__(
        self,
        *,
        corpus_root: Path,
        registry: EnricherRegistry,
        enricher_set: EnricherSet,
    ) -> None:
        self._corpus_root = corpus_root
        self._registry = registry
        self._enricher_set = enricher_set

    # ------------------------------------------------------------------ run

    async def run(
        self,
        *,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        episode_bundles: list[EpisodeArtifactBundle] | None = None,
        options: ExecutorOptions | None = None,
    ) -> EnrichmentRunResult:
        """Execute the enrichment pass.

        ``run_id`` defaults to a fresh UUID if not provided (standalone
        run). ``parent_run_id`` is set by the pipeline-attached path.
        ``episode_bundles`` is the list of per-episode artifact bundles
        (or ``None`` when ``options.corpus_only=True`` / there are no
        episodes).
        """
        opts = options or ExecutorOptions()
        run_id = run_id or str(uuid.uuid4())
        started_at_iso = utc_iso_now()
        started_at_s = time.monotonic()

        # Build the active-enricher list filtered by EnricherSet,
        # health (auto-disabled / cooldown), and CLI filters.
        health = HealthRegistry(self._corpus_root)
        health.load()
        active = self._resolve_active(opts, health)
        active_ids = [e.manifest.id for e in active]

        # Per-enricher metrics + circuit state for this run.
        metrics_by_id: dict[str, EnrichmentMetrics] = {
            e.manifest.id: new_metrics_for(
                enricher_id=e.manifest.id,
                enricher_version=e.manifest.version,
                scope=e.manifest.scope.value,
                tier=e.manifest.tier.value,
            )
            for e in active
        }
        circuit_by_id: dict[str, EnricherCircuitState] = {
            eid: EnricherCircuitState() for eid in active_ids
        }
        cost_state = CostCapState()
        cancel_event = asyncio.Event()

        jsonl_path = opts.jsonl_events_path or (self._corpus_root / "enrichments" / "run.jsonl")

        # JSONL: run started.
        self._safe_append_event(
            jsonl_path,
            build_run_started(
                run_id=run_id,
                parent_run_id=parent_run_id,
                profile=opts.profile,
                enricher_set=active_ids,
            ),
        )

        # Live status: starting.
        write_status(
            self._corpus_root,
            run_id=run_id,
            started_at=started_at_iso,
            profile=opts.profile,
            current_enricher=None,
            queue=active_ids,
            completed=[],
        )

        completed: list[dict[str, Any]] = []
        run_status = STATUS_OK
        try:
            # Phase 1 — episode-scope.
            if not opts.corpus_only and episode_bundles:
                episode_active = [e for e in active if e.manifest.scope is EnricherScope.EPISODE]
                await self._run_phase(
                    enrichers=episode_active,
                    bundles=episode_bundles,
                    metrics_by_id=metrics_by_id,
                    circuit_by_id=circuit_by_id,
                    cost_state=cost_state,
                    cancel_event=cancel_event,
                    parent_run_id=parent_run_id,
                    run_id=run_id,
                    jsonl_path=jsonl_path,
                    completed=completed,
                    schema_version=opts.enricher_schema_version,
                    cost_opts=opts,
                )

            # Phase 2 — corpus-scope.
            if not cancel_event.is_set():
                corpus_active = [e for e in active if e.manifest.scope is EnricherScope.CORPUS]
                await self._run_phase(
                    enrichers=corpus_active,
                    bundles=None,
                    metrics_by_id=metrics_by_id,
                    circuit_by_id=circuit_by_id,
                    cost_state=cost_state,
                    cancel_event=cancel_event,
                    parent_run_id=parent_run_id,
                    run_id=run_id,
                    jsonl_path=jsonl_path,
                    completed=completed,
                    schema_version=opts.enricher_schema_version,
                    cost_opts=opts,
                    all_bundles=episode_bundles,
                )

        except asyncio.CancelledError:
            run_status = STATUS_CANCELLED
            raise
        except Exception as exc:  # pragma: no cover — defensive safety net
            logger.exception("enrichment executor crashed; marking run failed (%s)", exc)
            run_status = STATUS_FAILED

        # Aggregate run-level outcome from per-enricher results.
        if run_status == STATUS_OK:
            run_status = self._aggregate_run_status(metrics_by_id, cancel_event)

        finished_at_iso = utc_iso_now()
        duration_ms = int((time.monotonic() - started_at_s) * 1000)

        # JSONL: run completed.
        per_enricher_totals = {
            eid: {
                "runs_total": m.runs_total,
                "runs_ok": m.runs_ok,
                "runs_failed": m.runs_failed,
                "runs_timeout": m.runs_timeout,
                "runs_quarantined": m.runs_quarantined,
                "runs_cancelled": m.runs_cancelled,
                "runs_skipped": m.runs_skipped,
                "retries": m.retries_total,
            }
            for eid, m in metrics_by_id.items()
        }
        self._safe_append_event(
            jsonl_path,
            build_run_completed(
                run_id=run_id,
                parent_run_id=parent_run_id,
                duration_ms=duration_ms,
                per_enricher_totals=per_enricher_totals,
            ),
        )

        # Update + persist health. Capture pre-transition state so we can
        # detect cross-run auto-disable transitions and emit the JSONL +
        # Sentry message via emit_auto_disable_event (chunk-1 wiring gap).
        for eid, m in metrics_by_id.items():
            enr = self._registry.get(eid)
            policy = policy_for(enr.manifest.tier)
            was_auto_disabled = health.get(eid).auto_disabled
            updated = health.update_after_run(
                eid,
                run_id=run_id,
                status=m.last_run_status or "skipped",
                policy=policy,
                circuit_state=circuit_by_id[eid].status.value,
            )
            if updated.auto_disabled and not was_auto_disabled:
                emit_auto_disable_event(
                    jsonl_path=jsonl_path,
                    ctx=RunContext(
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        enricher_id=eid,
                        enricher_version=enr.manifest.version,
                        tier=enr.manifest.tier.value,
                        attempt=1,
                        job_id=run_id,
                        cancel_event=cancel_event,
                    ),
                    consecutive_failed_runs=updated.consecutive_failures,
                    reason=updated.auto_disabled_reason or "auto_disable_threshold reached",
                )
        health.save()

        # Build + write run summary.
        run_summary = build_run_summary(
            run_id=run_id,
            parent_run_id=parent_run_id,
            profile=opts.profile,
            started_at=started_at_iso,
            finished_at=finished_at_iso,
            duration_ms=duration_ms,
            status=run_status,
            per_enricher=metrics_by_id,
        )
        try:
            write_run_summary(self._corpus_root, run_summary)
        except OSError as exc:
            logger.warning("enrichment run_summary write failed: %s", exc)

        # Final live status: idle.
        try:
            write_idle(self._corpus_root)
        except OSError as exc:
            logger.warning("enrichment status idle-write failed: %s", exc)

        return EnrichmentRunResult(
            run_id=run_id,
            status=run_status,
            duration_ms=duration_ms,
            per_enricher_metrics=metrics_by_id,
            run_summary=run_summary,
        )

    # ------------------------------------------------------------------ skip path

    async def run_skipped(
        self,
        *,
        run_id: str | None = None,
        reason: str,
    ) -> EnrichmentRunResult:
        """Emit a ``enrichment.run.skipped`` event (no work performed).

        Pipeline-attached path uses this when the core pipeline failed
        (per chunk-1 lock audit §B4); operator gets a deterministic
        JSONL marker without the executor scheduling any work.
        """
        run_id = run_id or str(uuid.uuid4())
        jsonl_path = self._corpus_root / "enrichments" / "run.jsonl"
        self._safe_append_event(jsonl_path, build_run_skipped(run_id=run_id, reason=reason))
        return EnrichmentRunResult(run_id=run_id, status=STATUS_SKIPPED, duration_ms=0)

    # ------------------------------------------------------------------ phase

    async def _run_phase(
        self,
        *,
        enrichers: list[Enricher],
        bundles: list[EpisodeArtifactBundle] | None,
        metrics_by_id: dict[str, EnrichmentMetrics],
        circuit_by_id: dict[str, EnricherCircuitState],
        cost_state: CostCapState,
        cancel_event: asyncio.Event,
        parent_run_id: str | None,
        run_id: str,
        jsonl_path: Path,
        completed: list[dict[str, Any]],
        schema_version: str,
        cost_opts: ExecutorOptions,
        all_bundles: list[EpisodeArtifactBundle] | None = None,
    ) -> None:
        """Run one phase (episode-scope OR corpus-scope).

        Concurrency is enforced via per-tier semaphores; each enricher
        in the phase reserves a slot of its tier's semaphore. Cost-cap
        violations abort the phase mid-flight.
        """
        if not enrichers:
            return

        # Per-tier semaphores.
        semaphores: dict[EnricherTier, asyncio.Semaphore] = {
            tier: asyncio.Semaphore(policy_for(tier).concurrency) for tier in EnricherTier
        }

        async def _run_one_enricher(enricher: Enricher) -> None:
            # Run-wide cost cap check (REPLAN-O7).
            if cost_state.run_wide_cap_exceeded(cost_opts.max_total_cost_usd_per_run):
                self._mark_skipped(
                    metrics_by_id[enricher.manifest.id],
                    reason="run_cost_cap_exceeded",
                )
                return
            if cancel_event.is_set():
                return
            tier = enricher.manifest.tier
            async with semaphores[tier]:
                # Episode-scope: run once per bundle. Corpus-scope: run once.
                if enricher.manifest.scope is EnricherScope.EPISODE:
                    assert bundles is not None
                    for bundle in bundles:
                        if cancel_event.is_set():
                            return
                        await self._execute_with_resilience(
                            enricher=enricher,
                            bundle=bundle,
                            all_bundles=all_bundles,
                            metrics=metrics_by_id[enricher.manifest.id],
                            circuit=circuit_by_id[enricher.manifest.id],
                            cost_state=cost_state,
                            cancel_event=cancel_event,
                            parent_run_id=parent_run_id,
                            run_id=run_id,
                            jsonl_path=jsonl_path,
                            completed=completed,
                            schema_version=schema_version,
                            cost_opts=cost_opts,
                        )
                else:
                    await self._execute_with_resilience(
                        enricher=enricher,
                        bundle=None,
                        all_bundles=all_bundles,
                        metrics=metrics_by_id[enricher.manifest.id],
                        circuit=circuit_by_id[enricher.manifest.id],
                        cost_state=cost_state,
                        cancel_event=cancel_event,
                        parent_run_id=parent_run_id,
                        run_id=run_id,
                        jsonl_path=jsonl_path,
                        completed=completed,
                        schema_version=schema_version,
                        cost_opts=cost_opts,
                    )

        await asyncio.gather(*(_run_one_enricher(e) for e in enrichers))

    # ------------------------------------------------------------------ single enricher

    async def _execute_with_resilience(
        self,
        *,
        enricher: Enricher,
        bundle: EpisodeArtifactBundle | None,
        all_bundles: list[EpisodeArtifactBundle] | None,
        metrics: EnrichmentMetrics,
        circuit: EnricherCircuitState,
        cost_state: CostCapState,
        cancel_event: asyncio.Event,
        parent_run_id: str | None,
        run_id: str,
        jsonl_path: Path,
        completed: list[dict[str, Any]],
        schema_version: str,
        cost_opts: ExecutorOptions,
    ) -> None:
        """Run one ``enricher.enrich(...)`` with the full resilience
        envelope: retry + backoff + circuit + heartbeat + cost-cap +
        observability.
        """
        manifest = enricher.manifest
        eid = manifest.id

        # Per-enricher quarantine check.
        if circuit.is_open:
            self._mark_quarantined(metrics, reason="circuit_open_at_run_start")
            return

        policy = policy_for(manifest.tier)
        max_retries = policy.max_retries
        attempt = 1
        last_status = STATUS_FAILED
        final_result: EnricherResult | None = None
        config = self._enricher_set.get_config(eid)

        # Timeout selection. ``expected_duration_s`` doubles as both the
        # heartbeat-stall threshold (soft) and the hard wait_for cap.
        timeout_s = (
            manifest.expected_duration_s
            if manifest.expected_duration_s is not None
            else (
                DEFAULT_CORPUS_TIMEOUT_S
                if manifest.scope is EnricherScope.CORPUS
                else DEFAULT_EPISODE_TIMEOUT_S
            )
        )
        # Heartbeat watchdog — soft stall detection at expected_duration_s.
        # Reset per attempt so retries get a fresh window.
        watchdog = HeartbeatWatchdog(enricher_id=eid, expected_interval_s=float(timeout_s))

        while True:
            if cancel_event.is_set():
                final_result = EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                break

            ctx = RunContext(
                run_id=run_id,
                parent_run_id=parent_run_id,
                enricher_id=eid,
                enricher_version=manifest.version,
                tier=manifest.tier.value,
                attempt=attempt,
                job_id=run_id,
                cancel_event=cancel_event,
            )
            stamp_sentry_correlation(ctx)

            started_at = utc_iso_now()
            self._safe_append_event(
                jsonl_path,
                build_enricher_started(ctx, scope=manifest.scope.value),
            )
            t_start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    enricher.enrich(
                        bundle=bundle,
                        corpus_root=self._corpus_root,
                        all_bundles=all_bundles,
                        config=config,
                        ctx=ctx,
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                # Hard timeout — non-retryable.
                duration_ms = int((time.monotonic() - t_start) * 1000)
                final_result = EnricherResult(
                    status=STATUS_TIMEOUT,
                    error=f"hard timeout ({timeout_s}s)",
                    error_class="RunTimeoutError",
                    duration_ms=duration_ms,
                    retry_count=attempt - 1,
                )
                break
            except asyncio.CancelledError:
                final_result = EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                break
            except Exception as exc:  # pylint: disable=broad-except
                duration_ms = int((time.monotonic() - t_start) * 1000)
                cls = classify_failure(exc)
                error_class = type(exc).__name__
                if cls == RetryClass.NON_RETRYABLE or attempt > max_retries:
                    final_result = EnricherResult(
                        status=status_for_exception(exc),
                        error=str(exc),
                        error_class=error_class,
                        duration_ms=duration_ms,
                        retry_count=attempt - 1,
                    )
                    break
                if cls == RetryClass.RETRYABLE_ONCE and attempt >= 2:
                    final_result = EnricherResult(
                        status=STATUS_FAILED,
                        error=str(exc),
                        error_class=error_class,
                        duration_ms=duration_ms,
                        retry_count=attempt - 1,
                    )
                    break
                # Retry with backoff.
                backoff = compute_backoff(attempt, policy)
                self._safe_append_event(
                    jsonl_path,
                    build_enricher_retry(
                        ctx,
                        backoff_s=backoff,
                        reason="transient_error",
                        error_class=error_class,
                    ),
                )
                circuit.record_failure(policy)
                metrics.record_circuit_transition(
                    from_state="closed", to_state=circuit.status.value
                )
                if circuit.is_open:
                    self._emit_circuit_opened(ctx, circuit, jsonl_path, parent_run_id)
                    metrics.record_circuit_transition(from_state="closed", to_state="open")
                    final_result = EnricherResult(
                        status=STATUS_QUARANTINED,
                        error=f"circuit_open_after_attempt_{attempt}",
                        error_class="CircuitOpen",
                        duration_ms=duration_ms,
                        retry_count=attempt - 1,
                    )
                    break
                await asyncio.sleep(backoff)
                attempt += 1
                continue

            # Success path (or terminal result from enricher). Reset the circuit's
            # consecutive-failure counter — without this, spread-out transient
            # failures across bundles accumulate monotonically and falsely
            # quarantine a healthy enricher (review 2026-07-17 H9).
            circuit.record_success()
            final_result = result
            break

        assert final_result is not None
        finished_at = utc_iso_now()
        duration_ms = int((time.monotonic() - t_start) * 1000) if "t_start" in locals() else 0
        # Soft stall detection: if the enricher ran longer than the
        # heartbeat watchdog's window without calling record_heartbeat,
        # emit stall_warning. (Hard timeout is already handled by
        # asyncio.wait_for upstream; this catches the slow-but-finishing
        # case operators want to know about.)
        #
        # factor=1.2 gives 20% slack so enrichers that finish *right* at
        # expected_duration_s don't trigger spurious warnings — the hard
        # timeout still fires at 1.0× (via asyncio.wait_for above).
        if watchdog.is_stalled(factor=1.2):
            ctx_stall = RunContext(
                run_id=run_id,
                parent_run_id=parent_run_id,
                enricher_id=eid,
                enricher_version=manifest.version,
                tier=manifest.tier.value,
                attempt=attempt,
                job_id=run_id,
                cancel_event=cancel_event,
            )
            self._safe_append_event(
                jsonl_path,
                build_stall_warning(
                    ctx_stall,
                    last_heartbeat_at=utc_iso_now(),
                    expected_interval_s=watchdog.expected_interval_s,
                ),
            )
        metrics.record_result(
            final_result,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_ms / 1000.0,
        )

        # Cost accounting (if the enricher set EnricherResult fields).
        # For sub-5 we honour any cost the enricher returned via the
        # result's records_written for record counting; future LLM-tier
        # enrichers will inject cost via the metrics record_cost API
        # (chunk 5 LLM enrichers). The cost_state is consulted between
        # enrichers in _run_one_enricher.

        # Per-enricher cost cap check — BEFORE writing the envelope, so a
        # quarantined enricher never leaves a valid-looking output file on disk
        # for consumers to read (review 2026-07-17 M30).
        #
        # NOTE: this cap is currently INERT — CostCapState.record_cost() is not
        # wired yet (deferred to the chunk-5 LLM enrichers), so it never fires.
        # It is kept here as the enforcement point but must NOT be mistaken for
        # active cost enforcement until record_cost() is called (review M13).
        if cost_state.per_enricher_cap_exceeded(eid, manifest):
            self._mark_quarantined(metrics, reason="per_enricher_cost_cap")
            self._safe_append_event(
                jsonl_path,
                build_enricher_completed(
                    RunContext(
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        enricher_id=eid,
                        enricher_version=manifest.version,
                        tier=manifest.tier.value,
                        attempt=attempt,
                        job_id=run_id,
                        cancel_event=cancel_event,
                    ),
                    status=STATUS_QUARANTINED,
                    duration_ms=duration_ms,
                    records_written=0,
                    retries=attempt - 1,
                ),
            )
            return

        # Write envelope to disk on success (only after the per-enricher cap
        # check passed — see M30 above).
        if final_result.status == STATUS_OK:
            self._write_envelope(
                enricher=enricher,
                bundle=bundle,
                result=final_result,
                schema_version=schema_version,
            )

        # Run-wide cap check after this enricher's contribution.
        if cost_state.run_wide_cap_exceeded(cost_opts.max_total_cost_usd_per_run):
            if cost_opts.fail_on_run_cost_cap:
                cancel_event.set()
            self._mark_skipped(metrics, reason="run_cost_cap_exceeded")
            return

        # JSONL: enricher completed.
        ctx_done = RunContext(
            run_id=run_id,
            parent_run_id=parent_run_id,
            enricher_id=eid,
            enricher_version=manifest.version,
            tier=manifest.tier.value,
            attempt=attempt,
            job_id=run_id,
            cancel_event=cancel_event,
        )
        self._safe_append_event(
            jsonl_path,
            build_enricher_completed(
                ctx_done,
                status=final_result.status,
                duration_ms=duration_ms,
                records_written=final_result.records_written,
                retries=attempt - 1,
            ),
        )
        # Cancelled outcome: emit cancel event (in addition to completed).
        if final_result.status == STATUS_CANCELLED:
            self._safe_append_event(
                jsonl_path,
                build_cancelled(
                    ctx_done,
                    reason="cancel_requested",
                    partial_records_written=final_result.records_written,
                ),
            )
        last_status = final_result.status
        completed.append(
            {
                "enricher_id": eid,
                "status": last_status,
                "duration_ms": duration_ms,
            }
        )

    # ------------------------------------------------------------------ helpers

    def _resolve_active(self, opts: ExecutorOptions, health: HealthRegistry) -> list[Enricher]:
        """Filter the registry to enrichers active for this run.

        Combines: EnricherSet membership, registry presence + LLM
        opt-in (via ``registry.list_enabled``), CLI ``--only`` /
        ``--skip`` filters, and health auto-disable / cooldown gating.
        """
        from_set = self._registry.list_enabled(self._enricher_set)
        result: list[Enricher] = []
        only = set(opts.only or [])
        skip = set(opts.skip or [])
        for enr in from_set:
            eid = enr.manifest.id
            if only and eid not in only:
                continue
            if eid in skip:
                continue
            if not health.is_active(eid):
                logger.info(
                    "enrichment: skipping %r (health inactive — auto_disabled or "
                    "cooldown_active)",
                    eid,
                )
                continue
            result.append(enr)
        return result

    def _safe_append_event(self, path: Path, payload: dict[str, Any]) -> None:
        """JSONL append wrapper that downgrades I/O errors to WARNING."""
        try:
            append_event(path, payload)
        except OSError as exc:
            logger.warning(
                "enrichment event append failed: %s (event=%s)",
                exc,
                payload.get("event_type"),
            )

    def _emit_circuit_opened(
        self,
        ctx: RunContext,
        circuit: EnricherCircuitState,
        jsonl_path: Path,
        parent_run_id: str | None,
    ) -> None:
        """Fire JSONL + Sentry breadcrumb for circuit-opened."""
        self._safe_append_event(
            jsonl_path,
            build_circuit_opened(
                ctx,
                consecutive_failures=circuit.consecutive_failures,
                cooldown_until=None,
            ),
        )
        breadcrumb_circuit_opened(
            ctx,
            consecutive_failures=circuit.consecutive_failures,
            cooldown_until=None,
        )

    def _write_envelope(
        self,
        *,
        enricher: Enricher,
        bundle: EpisodeArtifactBundle | None,
        result: EnricherResult,
        schema_version: str,
    ) -> None:
        """Build + write the on-disk envelope for a successful result.

        ``records_written`` accounting lives on ``EnrichmentMetrics``
        (the upstream ``record_result`` call already counted it from
        ``result.records_written``) — this method is purely the
        on-disk write.
        """
        manifest = enricher.manifest
        envelope = build_envelope(
            result=result,
            enricher_id=manifest.id,
            enricher_version=manifest.version,
            schema_version=schema_version,
        )
        if manifest.scope is EnricherScope.EPISODE:
            assert bundle is not None
            path = episode_enrichment_path(bundle, manifest.writes)
        else:
            path = corpus_enrichment_path(self._corpus_root, manifest.writes)
        try:
            ensure_directory(path.parent)
            path.write_text(json.dumps(envelope.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "enrichment envelope write failed: %s (enricher=%s, path=%s)",
                exc,
                manifest.id,
                path,
            )

    @staticmethod
    def _mark_quarantined(metrics: EnrichmentMetrics, *, reason: str) -> None:
        metrics.record_result(
            EnricherResult(status=STATUS_QUARANTINED, error=reason, error_class="Quarantine"),
            started_at=utc_iso_now(),
            finished_at=utc_iso_now(),
        )

    @staticmethod
    def _mark_skipped(metrics: EnrichmentMetrics, *, reason: str) -> None:
        metrics.record_result(
            EnricherResult(status=STATUS_SKIPPED, error=reason, error_class="Skip"),
            started_at=utc_iso_now(),
            finished_at=utc_iso_now(),
        )

    @staticmethod
    def _aggregate_run_status(
        metrics_by_id: dict[str, EnrichmentMetrics], cancel_event: asyncio.Event
    ) -> str:
        """Aggregate per-enricher outcomes into a run-level status.

        * Any cancelled marker → ``cancelled`` for the run.
        * Else any failed / timeout / quarantined → ``failed`` for the
          run.
        * Else ``ok``.
        """
        if cancel_event.is_set():
            return STATUS_CANCELLED
        any_fail = False
        for m in metrics_by_id.values():
            if m.runs_cancelled > 0:
                return STATUS_CANCELLED
            if m.runs_failed > 0 or m.runs_timeout > 0 or m.runs_quarantined > 0:
                any_fail = True
        return STATUS_FAILED if any_fail else STATUS_OK


# Convenience to satisfy mypy on safety-net mark_quarantined for
# circuit-open at run start (``status`` requires non-OK + error).
_QUARANTINED_AT_START_REASON = "circuit_open_at_run_start"


# ---------------------------------------------------------------------------
# Auto-disable + Sentry message helpers (consumed by the JSON Schema CLI
# layer; surface here for testability).
# ---------------------------------------------------------------------------


def emit_auto_disable_event(
    *,
    jsonl_path: Path,
    ctx: RunContext,
    consecutive_failed_runs: int,
    reason: str,
) -> None:
    """Fire JSONL + Sentry message for an auto-disable event.

    Called by the health-update boundary at end of run when the
    threshold is reached (see ``HealthRegistry.update_after_run`` for
    the counter; this just fires the o11y surfaces).
    """
    try:
        append_event(
            jsonl_path,
            build_auto_disabled(
                ctx,
                consecutive_failed_runs=consecutive_failed_runs,
                reason=reason,
            ),
        )
    except OSError as exc:
        logger.warning("enrichment auto_disabled event append failed: %s", exc)
    message_auto_disabled(ctx, consecutive_failed_runs=consecutive_failed_runs, reason=reason)
