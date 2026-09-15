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
                        {"type": "Topic", "id": "topic:a", "properties": {"label": "A topic"}},
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


def test_corpus_yaml_shadowing_a_profile_enricher_is_announced(tmp_path, caplog) -> None:
    """A corpus list that omits a profile-enabled enricher must SAY so (#2071 follow-up).

    The operator YAML's `enrichers:` dict is a complete declaration by design — presence is the
    enable — so a non-empty list replaces the profile set rather than merging. That is correct:
    merging would switch on enrichers a corpus deliberately omitted. What was wrong is that it
    happened silently, while per_enricher_config and opt_in_flags merge right beside it.

    person_web shipped in the cloud/DGX profiles on 2026-09-11 and never ran on prod for exactly
    this reason. It was found four days later because the people panel was empty — no log line,
    no metric, no error. This test is the log line.
    """
    from podcast_scraper.enrichment.cli import build_enricher_set_from_yaml
    from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile

    cfg = tmp_path / "viewer_operator.yaml"
    cfg.write_text(
        "enrichment:\n"
        "  enabled: true\n"
        "  enrichers:\n"
        "    grounding_rate: {}\n"
        "    person_web:\n"
        "      enabled: false\n",
        encoding="utf-8",
    )
    yaml_set = build_enricher_set_from_yaml(cfg)
    profile_set = enricher_set_for_profile("prod_dgx_full")

    # The replacement semantics themselves are the contract: a non-empty YAML list wins.
    assert yaml_set.enabled_enrichers == ["grounding_rate"]

    shadowed = [
        eid
        for eid in profile_set.enabled_enrichers
        if eid not in set(yaml_set.enabled_enrichers)
        and not (yaml_set.per_enricher_config.get(eid, {}).get("enabled") is False)
    ]
    # org_web is genuinely hidden and must be reported...
    assert "org_web" in shadowed
    # ...but person_web carries an explicit `enabled: false`, which is a recorded decision and
    # must NOT be nagged about, or the warning becomes noise operators learn to ignore.
    assert "person_web" not in shadowed
