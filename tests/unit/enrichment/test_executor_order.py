"""A corpus enricher that reads another one's output runs after it.

The executor runs a phase's enrichers together. ``news_connection`` measures the storyline file
``topic_theme_clusters`` writes in the same run, and ran 0.3 s before it on 2026-11-20 (the
database session's run of prod's nightly in the release image): it measured the previous build,
10 articles attached where the build attached 18. An enricher names another's output in its
manifest's ``reads``; it now starts only after that enricher has finished.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

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


def _manifest(eid: str, reads: List[str]) -> EnricherManifest:
    return EnricherManifest(
        id=eid,
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=reads,
        writes=f"{eid}.json",
        description="test",
    )


class _Timed:
    """Records when it started and whether another enricher's output existed then."""

    def __init__(
        self,
        eid: str,
        log: List[Any],
        *,
        reads: List[str],
        delay: float = 0.0,
        watch: str | None = None,
    ) -> None:
        self._manifest = _manifest(eid, reads)
        self._log = log
        self._delay = delay
        self._watch = watch

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: Any,
        config: dict,
        ctx: RunContext,
    ) -> EnricherResult:
        seen = (corpus_root / "enrichments" / self._watch).is_file() if self._watch else None
        self._log.append(("start", self._manifest.id, seen))
        await asyncio.sleep(self._delay)
        self._log.append(("end", self._manifest.id, None))
        return EnricherResult(status=STATUS_OK, data={"id": self._manifest.id}, records_written=1)


def _run(tmp_path: Path, enrichers: list) -> None:
    registry = EnricherRegistry()
    for e in enrichers:
        registry.register(e)
    enricher_set = EnricherSet(enabled_enrichers=[e.manifest.id for e in enrichers])
    executor = EnrichmentExecutor(
        corpus_root=tmp_path, registry=registry, enricher_set=enricher_set
    )
    asyncio.run(executor.run())


def test_a_reader_starts_after_the_writer_it_reads(tmp_path: Path) -> None:
    log: List[Any] = []
    writer = _Timed("storylines", log, reads=[".kg.json"], delay=0.2)
    reader = _Timed("measure", log, reads=[".kg.json", "storylines.json"], watch="storylines.json")
    _run(tmp_path, [reader, writer])
    starts = {eid: i for i, (kind, eid, _seen) in enumerate(log) if kind == "start"}
    ends = {eid: i for i, (kind, eid, _seen) in enumerate(log) if kind == "end"}
    assert ends["storylines"] < starts["measure"], log
    assert [seen for kind, eid, seen in log if kind == "start" and eid == "measure"] == [True]


def test_enrichers_without_a_dependency_still_run_together(tmp_path: Path) -> None:
    log: List[Any] = []
    slow = _Timed("slow", log, reads=[".kg.json"], delay=0.2)
    fast = _Timed("fast", log, reads=[".kg.json"])
    _run(tmp_path, [slow, fast])
    starts = {eid: i for i, (kind, eid, _seen) in enumerate(log) if kind == "start"}
    ends = {eid: i for i, (kind, eid, _seen) in enumerate(log) if kind == "end"}
    assert starts["fast"] < ends["slow"], log


def test_a_cycle_runs_everything_rather_than_waiting_forever(tmp_path: Path) -> None:
    log: List[Any] = []
    a = _Timed("a", log, reads=["b.json"])
    b = _Timed("b", log, reads=["a.json"])
    _run(tmp_path, [a, b])
    assert sorted(eid for kind, eid, _seen in log if kind == "end") == ["a", "b"]
