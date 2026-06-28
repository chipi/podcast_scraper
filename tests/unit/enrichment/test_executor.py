"""Unit tests for ``enrichment.executor`` — async two-phase pass + resilience.

These tests use in-process fake enrichers that conform to the
``Enricher`` protocol. The resilience scenarios (flaky / OOM /
timeout / stall) land in a later sub-commit alongside the
scorer-driven scenario fixtures; here we cover the executor's own
state machine + the integration of every chunk-1 module.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from podcast_scraper.enrichment.executor import (
    EnrichmentExecutor,
    EnrichmentRunResult,
    ExecutorOptions,
)
from podcast_scraper.enrichment.health import HealthRegistry
from podcast_scraper.enrichment.paths import (
    enrichment_health_path,
    enrichment_run_summary_path,
    enrichment_status_path,
)
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherSet,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
    STATUS_SKIPPED,
)
from podcast_scraper.enrichment.registry import EnricherRegistry

# ---------------------------------------------------------------------------
# Fixtures + fake enrichers
# ---------------------------------------------------------------------------


def _manifest(
    eid: str,
    *,
    tier: EnricherTier = EnricherTier.DETERMINISTIC,
    scope: EnricherScope = EnricherScope.CORPUS,
    writes: str | None = None,
    requires_opt_in: bool = False,
) -> EnricherManifest:
    return EnricherManifest(
        id=eid,
        version="1.0.0",
        scope=scope,
        tier=tier,
        reads=[],
        writes=writes or f"{eid}.json",
        description="test",
        requires_opt_in=requires_opt_in,
    )


class _OkEnricher:
    def __init__(self, manifest: EnricherManifest, *, records: int = 1) -> None:
        self._manifest = manifest
        self._records = records

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles,
        config: dict,
        ctx: RunContext,
    ) -> EnricherResult:
        return EnricherResult(
            status=STATUS_OK,
            data={"hello": "world"},
            records_written=self._records,
        )


class _RaiserEnricher:
    """Raises a non-retryable exception once per call."""

    def __init__(self, manifest: EnricherManifest) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(self, **_kw) -> EnricherResult:
        raise RuntimeError("boom")


class _FlakyEnricher:
    """Raises retryable errors first N times, then succeeds."""

    def __init__(self, manifest: EnricherManifest, *, fail_count: int = 2) -> None:
        self._manifest = manifest
        self._fail_count = fail_count
        self._calls = 0

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(self, **_kw) -> EnricherResult:
        self._calls += 1
        if self._calls <= self._fail_count:
            from podcast_scraper.enrichment.resilience import DependencyAccessError

            raise DependencyAccessError("transient lock")
        return EnricherResult(status=STATUS_OK, data={"x": 1}, records_written=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_executor(
    tmp_path: Path, *, enrichers: list, enabled_ids: list[str] | None = None
) -> EnrichmentExecutor:
    registry = EnricherRegistry()
    for e in enrichers:
        registry.register(e)
    if enabled_ids is None:
        enabled_ids = [e.manifest.id for e in enrichers]
    enricher_set = EnricherSet(enabled_enrichers=enabled_ids)
    return EnrichmentExecutor(corpus_root=tmp_path, registry=registry, enricher_set=enricher_set)


# ---------------------------------------------------------------------------
# No-op path: empty registry / no active enrichers
# ---------------------------------------------------------------------------


def test_run_with_empty_set_completes_ok(tmp_path: Path) -> None:
    """No enrichers in the EnricherSet → run completes with status `ok`."""
    executor = _build_executor(tmp_path, enrichers=[])
    result = asyncio.run(executor.run())
    assert isinstance(result, EnrichmentRunResult)
    assert result.status == STATUS_OK
    assert result.per_enricher_metrics == {}


def test_run_writes_idle_status_after_completion(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path, enrichers=[])
    asyncio.run(executor.run())
    payload = json.loads(enrichment_status_path(tmp_path).read_text())
    assert payload["current_enricher"] is None
    assert payload["queue"] == []


def test_run_writes_run_summary_file(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path, enrichers=[])
    asyncio.run(executor.run())
    payload = json.loads(enrichment_run_summary_path(tmp_path).read_text())
    assert payload["status"] == STATUS_OK
    assert "per_enricher" in payload


def test_run_emits_run_started_and_completed_jsonl(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path, enrichers=[])
    asyncio.run(executor.run())
    jsonl = tmp_path / "enrichments" / "run.jsonl"
    assert jsonl.is_file()
    events = [json.loads(line) for line in jsonl.read_text().strip().splitlines()]
    types = [e["event_type"] for e in events]
    assert "enrichment.run.started" in types
    assert "enrichment.run.completed" in types


# ---------------------------------------------------------------------------
# Single deterministic ok enricher (corpus scope)
# ---------------------------------------------------------------------------


def test_run_single_ok_enricher_writes_envelope(tmp_path: Path) -> None:
    enricher = _OkEnricher(_manifest("topic_cooccurrence_corpus"), records=5)
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    assert result.status == STATUS_OK
    # Output envelope was written.
    out_path = tmp_path / "enrichments" / "topic_cooccurrence_corpus.json"
    assert out_path.is_file()
    envelope = json.loads(out_path.read_text())
    assert envelope["derived"] is True
    assert envelope["status"] == STATUS_OK
    assert envelope["data"] == {"hello": "world"}
    assert envelope["enricher_id"] == "topic_cooccurrence_corpus"


def test_run_updates_per_enricher_metrics(tmp_path: Path) -> None:
    enricher = _OkEnricher(_manifest("x"), records=2)
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    m = result.per_enricher_metrics["x"]
    assert m.runs_total == 1
    assert m.runs_ok == 1
    assert m.output_records_total == 2
    assert m.last_run_status == STATUS_OK


def test_run_persists_health_after_run(tmp_path: Path) -> None:
    enricher = _OkEnricher(_manifest("x"))
    executor = _build_executor(tmp_path, enrichers=[enricher])
    asyncio.run(executor.run())
    health_payload = json.loads(enrichment_health_path(tmp_path).read_text())
    assert "x" in health_payload["enrichers"]
    assert health_payload["enrichers"]["x"]["last_status"] == STATUS_OK
    assert health_payload["enrichers"]["x"]["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# Episode-scope phase
# ---------------------------------------------------------------------------


def _bundle(tmp_path: Path, stem: str = "0001 - ep") -> EpisodeArtifactBundle:
    md = tmp_path / "metadata" / f"{stem}.metadata.json"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("{}", encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=md,
        gi_path=None,
        kg_path=None,
        bridge_path=None,
        episode_id="guid",
        stem=stem,
    )


def test_run_episode_scope_enricher_writes_per_episode_envelopes(
    tmp_path: Path,
) -> None:
    enricher = _OkEnricher(_manifest("episode_x", scope=EnricherScope.EPISODE, writes="ep.json"))
    executor = _build_executor(tmp_path, enrichers=[enricher])
    bundles = [_bundle(tmp_path, stem="0001 - ep"), _bundle(tmp_path, stem="0002 - ep")]
    result = asyncio.run(executor.run(episode_bundles=bundles))
    assert result.status == STATUS_OK
    # Both episode envelopes written.
    out1 = tmp_path / "metadata" / "enrichments" / "0001 - ep.ep.json"
    out2 = tmp_path / "metadata" / "enrichments" / "0002 - ep.ep.json"
    assert out1.is_file()
    assert out2.is_file()
    # Both runs counted on the same EnrichmentMetrics record.
    assert result.per_enricher_metrics["episode_x"].runs_total == 2


def test_corpus_only_skips_episode_scope_phase(tmp_path: Path) -> None:
    ep_enricher = _OkEnricher(_manifest("ep", scope=EnricherScope.EPISODE, writes="ep.json"))
    corpus_enricher = _OkEnricher(_manifest("corpus", scope=EnricherScope.CORPUS))
    executor = _build_executor(tmp_path, enrichers=[ep_enricher, corpus_enricher])
    bundles = [_bundle(tmp_path)]
    result = asyncio.run(
        executor.run(
            episode_bundles=bundles,
            options=ExecutorOptions(corpus_only=True),
        )
    )
    # Episode enricher never ran.
    assert result.per_enricher_metrics["ep"].runs_total == 0
    # Corpus enricher did run.
    assert result.per_enricher_metrics["corpus"].runs_total == 1


# ---------------------------------------------------------------------------
# Filters: --only / --skip
# ---------------------------------------------------------------------------


def test_only_filter_includes_only_named_enrichers(tmp_path: Path) -> None:
    a = _OkEnricher(_manifest("a"))
    b = _OkEnricher(_manifest("b"))
    executor = _build_executor(tmp_path, enrichers=[a, b])
    result = asyncio.run(executor.run(options=ExecutorOptions(only=["a"])))
    assert result.per_enricher_metrics["a"].runs_total == 1
    assert "b" not in result.per_enricher_metrics


def test_skip_filter_excludes_named_enrichers(tmp_path: Path) -> None:
    a = _OkEnricher(_manifest("a"))
    b = _OkEnricher(_manifest("b"))
    executor = _build_executor(tmp_path, enrichers=[a, b])
    result = asyncio.run(executor.run(options=ExecutorOptions(skip=["a"])))
    assert "a" not in result.per_enricher_metrics
    assert result.per_enricher_metrics["b"].runs_total == 1


# ---------------------------------------------------------------------------
# Health gating
# ---------------------------------------------------------------------------


def test_auto_disabled_enricher_does_not_run(tmp_path: Path) -> None:
    enricher = _OkEnricher(_manifest("x"))
    # Pre-populate health with auto_disabled.
    health = HealthRegistry(tmp_path)
    h = health.get("x")
    h.auto_disabled = True
    h.auto_disabled_reason = "previous test forced disable"
    health.save()

    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    # Enricher never ran — it was filtered out of the active set.
    assert "x" not in result.per_enricher_metrics


# ---------------------------------------------------------------------------
# Resilience: non-retryable failure → status: failed
# ---------------------------------------------------------------------------


def test_non_retryable_failure_records_failed_outcome(tmp_path: Path) -> None:
    enricher = _RaiserEnricher(_manifest("raiser"))
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    m = result.per_enricher_metrics["raiser"]
    assert m.runs_failed == 1
    assert m.last_run_status == "failed"
    # Run-level status reflects the failed enricher.
    assert result.status == "failed"


def test_non_retryable_failure_pushes_error_sample(tmp_path: Path) -> None:
    enricher = _RaiserEnricher(_manifest("raiser"))
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    samples = result.per_enricher_metrics["raiser"].error_samples
    assert len(samples) == 1
    assert samples[0]["error_class"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Resilience: retryable + recovers
# ---------------------------------------------------------------------------


def test_flaky_enricher_recovers_via_retry(tmp_path: Path) -> None:
    """Two retryable failures then success — only on a tier that allows retries.

    Deterministic tier has max_retries=0. We use EMBEDDING tier (max 3
    retries) so the retry path executes.
    """
    enricher = _FlakyEnricher(_manifest("flaky", tier=EnricherTier.EMBEDDING), fail_count=2)
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    m = result.per_enricher_metrics["flaky"]
    assert m.runs_ok == 1
    assert m.last_run_status == STATUS_OK


# ---------------------------------------------------------------------------
# run_skipped path
# ---------------------------------------------------------------------------


def test_run_skipped_emits_event_and_returns_skipped_result(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path, enrichers=[])
    result = asyncio.run(executor.run_skipped(reason="core_pipeline_failed"))
    assert result.status == STATUS_SKIPPED
    jsonl = tmp_path / "enrichments" / "run.jsonl"
    events = [json.loads(line) for line in jsonl.read_text().strip().splitlines()]
    assert events[-1]["event_type"] == "enrichment.run.skipped"
    assert events[-1]["reason"] == "core_pipeline_failed"


# ---------------------------------------------------------------------------
# Hard timeout
# ---------------------------------------------------------------------------


class _SlowEnricher:
    """Sleeps longer than the timeout — triggers the hard-timeout branch."""

    def __init__(self, manifest: EnricherManifest, *, sleep_s: float = 2.0) -> None:
        self._manifest = manifest
        self._sleep_s = sleep_s

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(self, **_kw) -> EnricherResult:
        await asyncio.sleep(self._sleep_s)
        return EnricherResult(status=STATUS_OK, data={})


def test_hard_timeout_marks_status_timeout(tmp_path: Path) -> None:
    """Hard timeout via expected_duration_s on the manifest."""

    enricher = _SlowEnricher(
        EnricherManifest(
            id="slow",
            version="1.0.0",
            scope=EnricherScope.CORPUS,
            tier=EnricherTier.DETERMINISTIC,
            reads=[],
            writes="slow.json",
            description="x",
            expected_duration_s=1,  # 1 second hard timeout
        ),
        sleep_s=3.0,
    )
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    m = result.per_enricher_metrics["slow"]
    assert m.runs_timeout == 1
    assert m.last_run_status == "timeout"


# ---------------------------------------------------------------------------
# Cancellation via cancel_event (cooperative)
# ---------------------------------------------------------------------------


class _CancelObservingEnricher:
    """Returns cancelled when the cancel_event fires."""

    def __init__(self, manifest: EnricherManifest, *, set_cancel: bool = True) -> None:
        self._manifest = manifest
        self._set_cancel = set_cancel

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(self, *, ctx: RunContext, **_kw) -> EnricherResult:
        if self._set_cancel:
            ctx.cancel_event.set()
        return EnricherResult(
            status="cancelled",
            error="cancel_requested",
            records_written=0,
        )


def test_enricher_marking_cancelled_flips_run_status(tmp_path: Path) -> None:
    """When the enricher sets cancel_event AND returns cancelled, run flips."""
    enricher = _CancelObservingEnricher(_manifest("c"))
    executor = _build_executor(tmp_path, enrichers=[enricher])
    result = asyncio.run(executor.run())
    assert result.status == "cancelled"
