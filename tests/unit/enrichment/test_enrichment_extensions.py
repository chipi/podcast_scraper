"""Enrichers come from the platform and from installed extensions (ADR-162 decision 5).

With no extension the platform registers its own five deterministic enrichers and nothing else,
and a profile that lists a private enricher simply runs without it. With one, everything the
extension contributes reaches the registries the enrichment code builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers import (
    all_deterministic_enricher_ids,
    PUBLIC_DETERMINISTIC_ENRICHER_IDS,
    register_deterministic_enrichers,
)
from podcast_scraper.enrichment.eval.admission import known_enricher_manifests
from podcast_scraper.enrichment.eval.registry import ScorerRegistry
from podcast_scraper.enrichment.eval.scorers import register_builtin_scorers
from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile
from podcast_scraper.enrichment.query_enrichers import register_deterministic_query_enrichers
from podcast_scraper.enrichment.query_registry import QueryEnricherRegistry
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.web_wiring import register_web_enrichers
from podcast_scraper.extensions import (
    _from_module,
    EnrichmentContribution,
    Extension,
    use_extensions,
)

pytestmark = pytest.mark.unit

PRIVATE = {"topic_theme_clusters", "temporal_velocity", "topic_similarity", "topic_consensus"}


def _intelligence() -> Extension:
    ext = _from_module("podcast_scraper.enrichment.intelligence_extension")
    assert ext is not None
    return ext


def test_without_extensions_only_the_public_enrichers_register() -> None:
    with use_extensions([]):
        reg = EnricherRegistry()
        register_deterministic_enrichers(reg)
        assert set(reg.all_ids()) == set(PUBLIC_DETERMINISTIC_ENRICHER_IDS)
        assert not PRIVATE & set(known_enricher_manifests())
        assert register_web_enrichers(EnricherRegistry()) == []
        queries = QueryEnricherRegistry()
        register_deterministic_query_enrichers(queries, corpus_root_provider=Path)
        assert queries.all_ids() == []


def test_a_profile_naming_private_enrichers_runs_without_them() -> None:
    with use_extensions([]):
        enabled = set(enricher_set_for_profile("cloud_balanced").enabled_enrichers)
    assert not PRIVATE & enabled
    assert set(PUBLIC_DETERMINISTIC_ENRICHER_IDS) <= enabled


def test_the_intelligence_extension_restores_every_private_enricher() -> None:
    with use_extensions([_intelligence()]):
        reg = EnricherRegistry()
        register_deterministic_enrichers(reg)
        assert {"topic_theme_clusters", "temporal_velocity"} <= set(reg.all_ids())
        assert PRIVATE | {"person_web", "org_web"} <= set(known_enricher_manifests())
        assert set(register_web_enrichers(EnricherRegistry())) == {"person_web", "org_web"}
        queries = QueryEnricherRegistry()
        register_deterministic_query_enrichers(queries, corpus_root_provider=Path)
        assert queries.all_ids() == ["query_topic_relatedness"]
        scorers = ScorerRegistry()
        register_builtin_scorers(scorers)
        assert scorers.has("topic_similarity")
        assert {"topic_theme_clusters", "temporal_velocity"} <= set(
            all_deterministic_enricher_ids()
        )


def test_a_contributed_deterministic_enricher_registers_after_the_public_ones() -> None:
    from podcast_scraper.enrichment.enrichers.insight_density import InsightDensityEnricher

    class _Fake(InsightDensityEnricher):
        manifest = InsightDensityEnricher.manifest.__class__(
            **{**InsightDensityEnricher.manifest.__dict__, "id": "fake_density"}
        )

    ext = Extension(
        name="fake", enrichment=EnrichmentContribution(deterministic=lambda: (_Fake(),))
    )
    with use_extensions([ext]):
        reg = EnricherRegistry()
        register_deterministic_enrichers(reg)
        assert reg.all_ids()[-1] == "fake_density"


def test_with_ml_hands_the_registry_to_every_extensions_wiring(tmp_path: Path) -> None:
    import asyncio

    from podcast_scraper.enrichment.cli import build_arg_parser, run_cli

    calls: list[tuple[object, object]] = []
    ext = Extension(
        name="fake",
        enrichment=EnrichmentContribution(ml_wiring=lambda reg, es: calls.append((reg, es))),
    )
    (tmp_path / "viewer_operator.yaml").write_text(
        "enrichment:\n  enrichers:\n    topic_cooccurrence_corpus: {}\n", encoding="utf-8"
    )
    args = build_arg_parser().parse_args(["--output-dir", str(tmp_path), "--with-ml"])
    with use_extensions([ext]):
        assert asyncio.run(run_cli(args)) == 0
    assert len(calls) == 1
    registry, _enricher_set = calls[0]
    assert isinstance(registry, EnricherRegistry)


def test_the_pipeline_asks_for_ml_only_when_an_installed_enricher_needs_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cloud_balanced lists topic_similarity, which needs an injected provider. With the extension
    that provides it the queued job runs --with-ml; without it, there is nothing to wire."""
    from types import SimpleNamespace

    import podcast_scraper.server.jobs as jobs_mod
    from podcast_scraper.workflow.orchestration import _maybe_spawn_enrichment_after_pipeline

    seen: list[bool] = []

    def _enqueue(corpus_root: Path, **kw: Any) -> dict[str, str]:
        seen.append(kw["with_ml"])
        return {"job_id": "j", "status": "queued"}

    monkeypatch.setattr(jobs_mod, "enqueue_enrichment_job", _enqueue)
    cfg: Any = SimpleNamespace(enrichment={"enabled": True}, profile="cloud_balanced")
    with use_extensions([_intelligence()]):
        _maybe_spawn_enrichment_after_pipeline(cfg, str(tmp_path))
    with use_extensions([]):
        _maybe_spawn_enrichment_after_pipeline(cfg, str(tmp_path))
    assert seen == [True, False]
