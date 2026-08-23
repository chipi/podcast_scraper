"""E5 — the executor emits a terminal completion log line.

The run's final status used to live ONLY in ``enrichments/run_summary.json`` on disk, so a
stalled / non-completing enrichment had no VictoriaLogs signal to alert on. The completion
line ``"enrichment: run complete run_id=... status=..."`` is that signal; the E5 staleness
alert (config/grafana/alerts/common/enrichment.yaml) fires on its absence. (#1811 E5)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from podcast_scraper.enrichment.executor import EnrichmentExecutor
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherSet,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.registry import EnricherRegistry


def _manifest(eid: str) -> EnricherManifest:
    return EnricherManifest(
        id=eid,
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[],
        writes=f"{eid}.json",
        description="test",
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


def test_run_emits_completion_log_with_status(tmp_path: Path, caplog) -> None:
    registry = EnricherRegistry()
    registry.register(_OkEnricher(_manifest("grounding_rate")))
    executor = EnrichmentExecutor(
        corpus_root=tmp_path,
        registry=registry,
        enricher_set=EnricherSet(enabled_enrichers=["grounding_rate"]),
    )

    with caplog.at_level(logging.INFO):
        result = asyncio.run(executor.run())

    completion = [r for r in caplog.records if "enrichment: run complete" in r.getMessage()]
    assert completion, "executor must emit a terminal 'enrichment: run complete' log line (E5)"
    msg = completion[0].getMessage()
    assert f"status={result.status}" in msg, f"completion line must carry the status: {msg!r}"
    assert f"run_id={result.run_id}" in msg, f"completion line must carry the run_id: {msg!r}"
