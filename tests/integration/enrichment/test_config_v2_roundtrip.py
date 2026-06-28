"""Round-trip: YAML knobs land in the produced envelope.

RFC-088 v2 sweep — operator writes ``temporal_velocity.alpha`` via the
config surface; the CLI must pick it up; the produced envelope must
reflect it. Pre-fix the knob plumbing was wired only at the YAML
reader; this test pins the entire chain.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.cli import build_enricher_set_from_yaml
from podcast_scraper.enrichment.enrichers import register_deterministic_enrichers
from podcast_scraper.enrichment.executor import EnrichmentExecutor, ExecutorOptions
from podcast_scraper.enrichment.paths import discover_episode_bundles
from podcast_scraper.enrichment.registry import EnricherRegistry

pytestmark = pytest.mark.integration


def _seed_corpus(corpus: Path) -> None:
    meta_dir = corpus / "feeds" / "rss_example" / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True)
    # Two episodes with publish_date and one Topic each
    for i, date in enumerate(["2026-04-15T00:00:00Z", "2026-05-15T00:00:00Z"], start=1):
        stem = f"0001 - ep{i}"
        (meta_dir / f"{stem}.metadata.json").write_text(
            json.dumps({"episode": {"guid": f"guid-{i}"}}), encoding="utf-8"
        )
        (meta_dir / f"{stem}.kg.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"type": "Episode", "id": f"ep:{i}", "properties": {"publish_date": date}},
                        {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )


def test_alpha_knob_from_yaml_lands_in_envelope(tmp_path: Path) -> None:
    """Operator writes ``temporal_velocity.alpha: 0.9`` to viewer_operator.yaml
    → the produced ``enrichments/temporal_velocity.json`` envelope's
    ``data.alpha`` reflects the same value."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed_corpus(corpus)

    yaml_path = corpus / "viewer_operator.yaml"
    yaml_path.write_text(
        "enrichment:\n"
        "  enabled: true\n"
        "  enrichers:\n"
        "    temporal_velocity:\n"
        "      alpha: 0.9\n"
        "      window_months: 6\n",
        encoding="utf-8",
    )

    enricher_set = build_enricher_set_from_yaml(yaml_path)
    assert "temporal_velocity" in enricher_set.enabled_enrichers
    assert enricher_set.get_config("temporal_velocity")["alpha"] == 0.9

    registry = EnricherRegistry()
    register_deterministic_enrichers(registry)
    executor = EnrichmentExecutor(
        corpus_root=corpus,
        registry=registry,
        enricher_set=enricher_set,
    )
    bundles = discover_episode_bundles(corpus)
    result = asyncio.run(
        executor.run(
            episode_bundles=bundles,
            options=ExecutorOptions(only=["temporal_velocity"]),
        )
    )
    assert result.status == "ok", result

    envelope_path = corpus / "enrichments" / "temporal_velocity.json"
    assert envelope_path.is_file()
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["data"]["alpha"] == 0.9
    assert len(envelope["data"]["window_months"]) == 6


def test_explicit_enabled_false_disables_enricher_in_yaml(tmp_path: Path) -> None:
    """Shape B opt-out: ``temporal_velocity.enabled: false`` removes it from
    the active set even with its block present + knobs preserved."""
    yaml_path = tmp_path / "operator.yaml"
    yaml_path.write_text(
        "enrichment:\n"
        "  enabled: true\n"
        "  enrichers:\n"
        "    temporal_velocity:\n"
        "      enabled: false\n"
        "      alpha: 0.9\n"
        "    grounding_rate: {}\n",
        encoding="utf-8",
    )
    enricher_set = build_enricher_set_from_yaml(yaml_path)
    assert "grounding_rate" in enricher_set.enabled_enrichers
    assert "temporal_velocity" not in enricher_set.enabled_enrichers
    # Config preserved so the operator can re-enable later.
    assert enricher_set.get_config("temporal_velocity")["alpha"] == 0.9
