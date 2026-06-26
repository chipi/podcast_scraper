"""Unit tests for ``enrichment.protocol`` foundation types."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from podcast_scraper.enrichment.protocol import (
    ALL_STATUSES,
    Enricher,
    EnricherManifest,
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
    sync_enricher,
)

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------


def test_all_statuses_covers_the_full_terminal_set() -> None:
    assert ALL_STATUSES == frozenset(
        {
            STATUS_OK,
            STATUS_FAILED,
            STATUS_TIMEOUT,
            STATUS_QUARANTINED,
            STATUS_CANCELLED,
            STATUS_SKIPPED,
        }
    )


# ---------------------------------------------------------------------------
# EnricherResult invariants
# ---------------------------------------------------------------------------


def test_enricher_result_ok_requires_data() -> None:
    with pytest.raises(ValueError, match="non-None data"):
        EnricherResult(status=STATUS_OK)


def test_enricher_result_ok_with_data_passes() -> None:
    r = EnricherResult(status=STATUS_OK, data={"x": 1})
    assert r.status == STATUS_OK
    assert r.data == {"x": 1}
    assert r.retry_count == 0
    assert r.records_written == 0


def test_enricher_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        EnricherResult(status="garbage")


@pytest.mark.parametrize(
    "status",
    [STATUS_FAILED, STATUS_TIMEOUT, STATUS_QUARANTINED, STATUS_CANCELLED, STATUS_SKIPPED],
)
def test_enricher_result_non_ok_can_have_none_data(status: str) -> None:
    r = EnricherResult(status=status, error="boom", error_class="RuntimeError")
    assert r.data is None
    assert r.error == "boom"


def test_enricher_result_is_frozen() -> None:
    r = EnricherResult(status=STATUS_OK, data={"x": 1})
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        r.status = STATUS_FAILED  # type: ignore[misc]


def test_enricher_result_carries_resilience_metadata() -> None:
    r = EnricherResult(
        status=STATUS_OK,
        data={"x": 1},
        retry_count=2,
        circuit_state="closed",
        duration_ms=42,
        records_written=10,
    )
    assert r.retry_count == 2
    assert r.circuit_state == "closed"
    assert r.duration_ms == 42
    assert r.records_written == 10


# ---------------------------------------------------------------------------
# EnricherManifest cost-cap fields
# ---------------------------------------------------------------------------


def test_enricher_manifest_defaults_for_cost_cap_are_none() -> None:
    """Per O1: ``None`` means unbounded; deterministic enrichers leave it at None."""
    m = _make_manifest(EnricherTier.DETERMINISTIC)
    assert m.max_cost_usd_per_run is None
    assert m.expected_duration_s is None
    assert m.requires_opt_in is False


def test_enricher_manifest_carries_cost_cap_when_set() -> None:
    m = EnricherManifest(
        id="nli_contradiction",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json", ".bridge.json"],
        writes="nli_contradiction.json",
        description="...",
        max_cost_usd_per_run=0.50,
        expected_duration_s=300,
    )
    assert m.max_cost_usd_per_run == pytest.approx(0.50)
    assert m.expected_duration_s == 300


def test_enricher_manifest_llm_tier_requires_opt_in_flag() -> None:
    """LLM-tier enrichers conventionally set ``requires_opt_in=True``."""
    m = EnricherManifest(
        id="query_synthesis",
        version="0.1.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.LLM,
        reads=[".gi.json"],
        writes="query_synthesis.json",
        description="future LLM tier",
        requires_opt_in=True,
    )
    assert m.requires_opt_in is True


# ---------------------------------------------------------------------------
# Enricher Protocol — runtime_checkable
# ---------------------------------------------------------------------------


class _MinimalEnricher:
    """Conforms to the ``Enricher`` protocol structurally."""

    @property
    def manifest(self) -> EnricherManifest:
        return _make_manifest(EnricherTier.DETERMINISTIC)

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict,
        ctx: RunContext,
    ) -> EnricherResult:
        return EnricherResult(status=STATUS_OK, data={"x": 1})


def test_enricher_protocol_runtime_check_accepts_conforming_class() -> None:
    assert isinstance(_MinimalEnricher(), Enricher)


def test_enricher_protocol_runtime_check_rejects_non_conforming_class() -> None:
    class _NoManifest:
        async def enrich(self, **kw):
            return EnricherResult(status=STATUS_OK, data={})

    assert not isinstance(_NoManifest(), Enricher)


# ---------------------------------------------------------------------------
# sync_enricher decorator
# ---------------------------------------------------------------------------


def test_sync_enricher_wraps_dict_return_into_ok_result() -> None:
    @sync_enricher
    def body(**_kw) -> dict:
        return {"computed": 42}

    result = asyncio.run(body())
    assert isinstance(result, EnricherResult)
    assert result.status == STATUS_OK
    assert result.data == {"computed": 42}


def test_sync_enricher_passes_through_enricher_result() -> None:
    @sync_enricher
    def body(**_kw) -> EnricherResult:
        return EnricherResult(status=STATUS_SKIPPED, error="not configured")

    result = asyncio.run(body())
    assert result.status == STATUS_SKIPPED
    assert result.error == "not configured"


def test_sync_enricher_catches_exception_as_failed_result() -> None:
    @sync_enricher
    def body(**_kw) -> dict:
        raise RuntimeError("boom")

    result = asyncio.run(body())
    assert result.status == STATUS_FAILED
    assert result.error == "boom"
    assert result.error_class == "RuntimeError"


def test_sync_enricher_rejects_invalid_return_type() -> None:
    @sync_enricher
    def body(**_kw):
        return 42  # not dict, not EnricherResult

    with pytest.raises(TypeError, match="must return dict or EnricherResult"):
        asyncio.run(body())


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


def test_run_context_carries_correlation_fields() -> None:
    ev = asyncio.Event()
    ctx = RunContext(
        run_id="run-1",
        parent_run_id="parent-1",
        enricher_id="topic_cooccurrence",
        enricher_version="1.0.0",
        tier="deterministic",
        attempt=1,
        job_id="job-1",
        cancel_event=ev,
    )
    assert ctx.run_id == "run-1"
    assert ctx.parent_run_id == "parent-1"
    assert ctx.attempt == 1
    assert ctx.cancel_event is ev


def test_run_context_standalone_has_no_parent() -> None:
    ctx = RunContext(
        run_id="run-2",
        parent_run_id=None,
        enricher_id="x",
        enricher_version="0",
        tier="deterministic",
        attempt=1,
        job_id="run-2",  # == run_id for standalone
        cancel_event=asyncio.Event(),
    )
    assert ctx.parent_run_id is None
    assert ctx.job_id == ctx.run_id


# ---------------------------------------------------------------------------
# EpisodeArtifactBundle
# ---------------------------------------------------------------------------


def test_episode_artifact_bundle_carries_optional_artifact_paths() -> None:
    b = EpisodeArtifactBundle(
        metadata_path=Path("metadata/0001 - ep.metadata.json"),
        gi_path=Path("metadata/0001 - ep.gi.json"),
        kg_path=Path("metadata/0001 - ep.kg.json"),
        bridge_path=Path("metadata/0001 - ep.bridge.json"),
        episode_id="guid-1",
        stem="0001 - ep",
    )
    assert b.stem == "0001 - ep"
    assert b.kg_path is not None
    assert b.kg_path.name == "0001 - ep.kg.json"


def test_episode_artifact_bundle_optional_paths_can_be_none() -> None:
    b = EpisodeArtifactBundle(
        metadata_path=Path("x.metadata.json"),
        gi_path=None,
        kg_path=None,
        bridge_path=None,
        episode_id="guid-2",
        stem="x",
    )
    assert b.gi_path is None


# ---------------------------------------------------------------------------
# EnricherSet (minimal stub — chunk 7 extends)
# ---------------------------------------------------------------------------


def test_enricher_set_defaults_empty() -> None:
    s = EnricherSet()
    assert s.enabled_enrichers == []
    assert s.is_enabled("foo") is False
    assert s.get_config("foo") == {}
    assert s.has_opt_in("foo") is False


def test_enricher_set_is_enabled_tracks_explicit_list() -> None:
    s = EnricherSet(enabled_enrichers=["topic_cooccurrence", "temporal_velocity"])
    assert s.is_enabled("topic_cooccurrence")
    assert s.is_enabled("temporal_velocity")
    assert not s.is_enabled("nli_contradiction")


def test_enricher_set_per_enricher_config_lookup() -> None:
    s = EnricherSet(per_enricher_config={"nli_contradiction": {"threshold": 0.6}})
    assert s.get_config("nli_contradiction") == {"threshold": 0.6}
    assert s.get_config("other") == {}


def test_enricher_set_opt_in_flag_lookup() -> None:
    s = EnricherSet(opt_in_flags={"query_synthesis": True, "x": False})
    assert s.has_opt_in("query_synthesis") is True
    assert s.has_opt_in("x") is False
    assert s.has_opt_in("missing") is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(tier: EnricherTier) -> EnricherManifest:
    return EnricherManifest(
        id="x",
        version="1.0.0",
        scope=EnricherScope.EPISODE,
        tier=tier,
        reads=[".kg.json"],
        writes="x.json",
        description="test enricher",
    )
