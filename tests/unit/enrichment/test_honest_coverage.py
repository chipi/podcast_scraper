"""E1 — honest enrichment coverage.

Tests that an enabled enricher that cannot run (not_registered or
requires_opt_in) leaves an explicit ``not_run`` row in the run summary
and causes ``status == "partial"``, not ``"ok"``.

Covers the prod incident where a reenrich ran 7 deterministic enrichers,
silently skipped ``topic_similarity`` + ``topic_consensus`` (enabled-but-
not-registered), and reported ``status=ok``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from podcast_scraper.enrichment.executor import EnrichmentExecutor
from podcast_scraper.enrichment.metrics import new_metrics_for
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherSet,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
    STATUS_PARTIAL,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.run_summary import build_run_summary

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _manifest(
    eid: str,
    *,
    tier: EnricherTier = EnricherTier.DETERMINISTIC,
    scope: EnricherScope = EnricherScope.CORPUS,
    requires_opt_in: bool = False,
) -> EnricherManifest:
    return EnricherManifest(
        id=eid,
        version="1.0.0",
        scope=scope,
        tier=tier,
        reads=[],
        writes=f"{eid}.json",
        description="test",
        requires_opt_in=requires_opt_in,
    )


class _OkEnricher:
    def __init__(self, manifest: EnricherManifest) -> None:
        self._manifest = manifest

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
        return EnricherResult(status=STATUS_OK, data={"x": 1}, records_written=1)


def _build_executor(
    tmp_path: Path,
    *,
    registered: list,
    enabled_ids: list[str],
    opt_in_flags: dict[str, bool] | None = None,
) -> EnrichmentExecutor:
    registry = EnricherRegistry()
    for e in registered:
        registry.register(e)
    enricher_set = EnricherSet(
        enabled_enrichers=enabled_ids,
        opt_in_flags=opt_in_flags or {},
    )
    return EnrichmentExecutor(corpus_root=tmp_path, registry=registry, enricher_set=enricher_set)


# ---------------------------------------------------------------------------
# (a) unregistered enabled enricher → not_run row + status partial
# ---------------------------------------------------------------------------


def test_unregistered_enabled_enricher_appears_as_not_run_row(tmp_path: Path) -> None:
    """An enabled-but-unregistered id must produce a ``not_run`` row in run_summary.

    Reproduces the prod incident: ``topic_similarity`` + ``topic_consensus``
    were in enabled_enrichers but not registered; run reported ``status=ok``
    with no row for either.
    """
    # One registered enricher + two enabled-but-unregistered ids (the prod pattern).
    executor = _build_executor(
        tmp_path,
        registered=[_OkEnricher(_manifest("grounding_rate"))],
        enabled_ids=["grounding_rate", "topic_similarity", "topic_consensus"],
    )
    result = asyncio.run(executor.run())

    assert result.status == STATUS_PARTIAL, (
        f"expected 'partial' but got {result.status!r} — "
        "unregistered enrichers must surface as partial, not ok"
    )

    summary = result.run_summary
    per = summary["per_enricher"]

    assert "topic_similarity" in per, "topic_similarity must have a row (was missing before fix)"
    assert "topic_consensus" in per, "topic_consensus must have a row (was missing before fix)"

    ts_row = per["topic_similarity"]
    assert ts_row["status"] == "not_run"
    assert ts_row["reason"] == "not_registered"
    assert ts_row["runs_total"] == 0

    tc_row = per["topic_consensus"]
    assert tc_row["status"] == "not_run"
    assert tc_row["reason"] == "not_registered"
    assert tc_row["runs_total"] == 0

    # The enricher that DID run must still have its normal row.
    assert per["grounding_rate"]["status"] == STATUS_OK


# ---------------------------------------------------------------------------
# (b) all enrichers registered and run → status ok (no regression)
# ---------------------------------------------------------------------------


def test_all_registered_enrichers_run_produces_ok(tmp_path: Path) -> None:
    """When every enabled enricher is registered and runs ok, status is ``ok``."""
    executor = _build_executor(
        tmp_path,
        registered=[
            _OkEnricher(_manifest("grounding_rate")),
            _OkEnricher(_manifest("temporal_velocity")),
        ],
        enabled_ids=["grounding_rate", "temporal_velocity"],
    )
    result = asyncio.run(executor.run())

    assert result.status == STATUS_OK, (
        f"expected 'ok' but got {result.status!r} — "
        "a clean run with no gaps must not regress to partial"
    )
    per = result.run_summary["per_enricher"]
    assert per["grounding_rate"]["status"] == STATUS_OK
    assert per["temporal_velocity"]["status"] == STATUS_OK


# ---------------------------------------------------------------------------
# (c) list_enabled returns skip tuples with correct reasons
# ---------------------------------------------------------------------------


def test_list_enabled_returns_skip_tuple_for_unregistered() -> None:
    """``list_enabled`` returns ``(enrichers, skips)``; unregistered id →
    ``(id, 'not_registered')`` in skips."""
    reg = EnricherRegistry()
    reg.register(_OkEnricher(_manifest("grounding_rate")))

    enrichers, skips = reg.list_enabled(
        EnricherSet(enabled_enrichers=["grounding_rate", "topic_similarity"])
    )

    assert [e.manifest.id for e in enrichers] == ["grounding_rate"]
    assert ("topic_similarity", "not_registered") in skips


def test_list_enabled_returns_skip_tuple_for_requires_opt_in() -> None:
    """An enricher with ``requires_opt_in=True`` missing the flag →
    ``(id, 'requires_opt_in')`` in skips."""
    reg = EnricherRegistry()
    reg.register(
        _OkEnricher(_manifest("query_synthesis", tier=EnricherTier.LLM, requires_opt_in=True))
    )

    enrichers, skips = reg.list_enabled(
        EnricherSet(enabled_enrichers=["query_synthesis"])  # opt_in_flags NOT set
    )

    assert enrichers == []
    assert ("query_synthesis", "requires_opt_in") in skips


def test_list_enabled_empty_skips_when_all_registered() -> None:
    """No skips when every enabled id is registered and passes opt-in."""
    reg = EnricherRegistry()
    reg.register(_OkEnricher(_manifest("grounding_rate")))
    reg.register(_OkEnricher(_manifest("temporal_velocity")))

    enrichers, skips = reg.list_enabled(
        EnricherSet(enabled_enrichers=["grounding_rate", "temporal_velocity"])
    )

    assert len(enrichers) == 2
    assert skips == []


# ---------------------------------------------------------------------------
# (c-extra) build_run_summary unavailable param
# ---------------------------------------------------------------------------


def test_build_run_summary_unavailable_adds_not_run_rows() -> None:
    """``build_run_summary`` with ``unavailable`` produces ``not_run`` rows."""
    m = new_metrics_for(
        enricher_id="grounding_rate",
        enricher_version="1.0.0",
        scope="corpus",
        tier="deterministic",
    )
    m.record_result(
        EnricherResult(status=STATUS_OK, data={"x": 1}, records_written=1),
        started_at="t0",
        finished_at="t1",
    )

    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=100,
        status=STATUS_PARTIAL,
        per_enricher={"grounding_rate": m},
        unavailable=[("topic_similarity", "not_registered"), ("topic_consensus", "not_registered")],
    )

    per = summary["per_enricher"]
    assert "topic_similarity" in per
    assert per["topic_similarity"]["status"] == "not_run"
    assert per["topic_similarity"]["reason"] == "not_registered"
    assert per["topic_similarity"]["runs_total"] == 0

    assert "topic_consensus" in per
    assert per["topic_consensus"]["status"] == "not_run"

    # Existing row unchanged.
    assert per["grounding_rate"]["status"] == STATUS_OK


def test_build_run_summary_no_unavailable_unchanged() -> None:
    """Calling ``build_run_summary`` without ``unavailable`` behaves exactly as before."""
    m = new_metrics_for(
        enricher_id="x",
        enricher_version="1.0.0",
        scope="corpus",
        tier="deterministic",
    )
    m.record_result(
        EnricherResult(status=STATUS_OK, data={"x": 1}),
        started_at="t0",
        finished_at="t1",
    )
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=0,
        status=STATUS_OK,
        per_enricher={"x": m},
        # no unavailable kwarg — tests backwards compat default
    )
    assert summary["status"] == STATUS_OK
    assert list(summary["per_enricher"].keys()) == ["x"]
