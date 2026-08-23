"""End-to-end incrementality through the executor (#1649).

``test_staleness.py`` covers the decision function. This file covers the thing that actually
matters: that the executor *acts* on it, and — more importantly — that an upstream repair is
not silently skipped.

The failure being guarded against is subtle and expensive. Key staleness on
``enricher_version``/``schema_version`` alone (the fields already persisted, so the obvious
choice) and the corpus repair in #1655 becomes a no-op: GI is rewritten with correct speakers,
enrichment re-runs, every episode reads "unchanged at the same version", 678 skipped, green
run, nothing repaired. Same shape as the bug being repaired — a signal reporting on the
machinery instead of the work.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.executor import (
    EnrichmentExecutor,
    ExecutorOptions,
)
from podcast_scraper.enrichment.paths import episode_enrichment_path
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherSet,
    EnricherTier,
    EpisodeArtifactBundle,
)
from podcast_scraper.enrichment.registry import EnricherRegistry

pytestmark = [pytest.mark.unit]

STATUS_OK = "ok"


class _CountingEnricher:
    """Episode-scope enricher that records how many episodes it was asked to process."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._manifest = EnricherManifest(
            id="counter",
            version="1.0.0",
            scope=EnricherScope.EPISODE,
            tier=EnricherTier.DETERMINISTIC,
            reads=[],
            writes="counter.json",
            description="counts how many episodes it was asked to process",
        )

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(self, *, bundle, corpus_root, all_bundles, config, ctx) -> EnricherResult:
        self.calls.append(bundle.stem if bundle else "corpus")
        return EnricherResult(status=STATUS_OK, data={"n": 1}, records_written=1)


def _make_episode(corpus_root: Path, stem: str, *, insights: int = 2) -> EpisodeArtifactBundle:
    meta_dir = corpus_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = meta_dir / f"{stem}.metadata.json"
    metadata_path.write_text(json.dumps({"episode": {"episode_id": stem}}), encoding="utf-8")
    gi_path = meta_dir / f"{stem}.gi.json"
    gi_path.write_text(
        json.dumps({"nodes": [{"type": "Insight", "id": f"i{n}"} for n in range(insights)]}),
        encoding="utf-8",
    )
    return EpisodeArtifactBundle(
        metadata_path=metadata_path,
        gi_path=gi_path,
        kg_path=None,
        bridge_path=None,
        episode_id=stem,
        stem=stem,
    )


def _executor(corpus_root: Path, enricher: _CountingEnricher) -> EnrichmentExecutor:
    registry = EnricherRegistry()
    registry.register(enricher)
    return EnrichmentExecutor(
        corpus_root=corpus_root,
        registry=registry,
        enricher_set=EnricherSet(enabled_enrichers=["counter"]),
    )


def _run(executor: EnrichmentExecutor, bundles, **opts):
    return asyncio.run(executor.run(episode_bundles=bundles, options=ExecutorOptions(**opts)))


class TestIncrementality:
    def test_first_run_enriches_every_episode(self, tmp_path: Path) -> None:
        bundles = [_make_episode(tmp_path, f"000{i} - ep") for i in range(3)]
        enricher = _CountingEnricher()
        _run(_executor(tmp_path, enricher), bundles)
        assert len(enricher.calls) == 3

    def test_second_run_with_no_changes_does_no_work(self, tmp_path: Path) -> None:
        """The whole point: a 16-episode ingest must not trigger a 678-episode pass."""
        bundles = [_make_episode(tmp_path, f"000{i} - ep") for i in range(3)]
        first = _CountingEnricher()
        _run(_executor(tmp_path, first), bundles)
        assert len(first.calls) == 3

        second = _CountingEnricher()
        _run(_executor(tmp_path, second), bundles)
        assert second.calls == [], "unchanged episodes were re-enriched"

    def test_an_upstream_gi_change_re_enriches_only_that_episode(self, tmp_path: Path) -> None:
        """THE test for #1649 — the corpus repair must not be silently skipped.

        The enricher version is deliberately unchanged between runs, exactly as it will be
        during the #1655 repair. Only the episode whose GI was rewritten may re-run.
        """
        bundles = [_make_episode(tmp_path, f"000{i} - ep") for i in range(3)]
        _run(_executor(tmp_path, _CountingEnricher()), bundles)

        # Upstream repair rewrites ONE episode's GI with named speakers.
        repaired = bundles[1]
        assert repaired.gi_path is not None
        repaired.gi_path.write_text(
            json.dumps({"nodes": [{"type": "Insight", "properties": {"speaker": "Simon Last"}}]}),
            encoding="utf-8",
        )

        second = _CountingEnricher()
        _run(_executor(tmp_path, second), bundles)
        assert second.calls == [repaired.stem]

    def test_force_re_enriches_everything(self, tmp_path: Path) -> None:
        bundles = [_make_episode(tmp_path, f"000{i} - ep") for i in range(3)]
        _run(_executor(tmp_path, _CountingEnricher()), bundles)

        forced = _CountingEnricher()
        _run(_executor(tmp_path, forced), bundles, force=True)
        assert len(forced.calls) == 3

    def test_rewriting_gi_with_identical_bytes_is_not_a_change(self, tmp_path: Path) -> None:
        """The pipeline rewrites identical files routinely; that must not defeat the skip."""
        bundles = [_make_episode(tmp_path, "0001 - ep")]
        _run(_executor(tmp_path, _CountingEnricher()), bundles)

        gi = bundles[0].gi_path
        assert gi is not None
        gi.write_text(gi.read_text(encoding="utf-8"), encoding="utf-8")

        second = _CountingEnricher()
        _run(_executor(tmp_path, second), bundles)
        assert second.calls == []

    def test_the_fingerprint_is_persisted_next_to_the_output(self, tmp_path: Path) -> None:
        """Without this the next run has nothing to compare and incrementality never engages."""
        bundles = [_make_episode(tmp_path, "0001 - ep")]
        enricher = _CountingEnricher()
        _run(_executor(tmp_path, enricher), bundles)

        written = json.loads(
            episode_enrichment_path(bundles[0], "counter.json").read_text(encoding="utf-8")
        )
        assert written.get("input_fingerprint")
