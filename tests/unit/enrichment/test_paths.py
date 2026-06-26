"""Unit tests for ``enrichment.paths``."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.enrichment.paths import (
    corpus_enrichment_path,
    enrichment_health_path,
    enrichment_run_summary_path,
    enrichment_status_path,
    ensure_directory,
    episode_enrichment_path,
    viewer_dir,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle


def _bundle_single_feed(tmp_path: Path) -> EpisodeArtifactBundle:
    md = tmp_path / "metadata" / "0001 - ep.metadata.json"
    return EpisodeArtifactBundle(
        metadata_path=md,
        gi_path=md.with_suffix(".gi.json"),
        kg_path=md.with_suffix(".kg.json"),
        bridge_path=md.with_suffix(".bridge.json"),
        episode_id="guid-1",
        stem="0001 - ep",
    )


def _bundle_multi_feed(tmp_path: Path) -> EpisodeArtifactBundle:
    md = tmp_path / "feeds" / "rss_example_com_a1b2" / "metadata" / "0001 - ep.metadata.json"
    return EpisodeArtifactBundle(
        metadata_path=md,
        gi_path=None,
        kg_path=md.with_suffix(".kg.json"),
        bridge_path=None,
        episode_id="guid-1",
        stem="0001 - ep",
    )


# ---------------------------------------------------------------------------
# episode_enrichment_path
# ---------------------------------------------------------------------------


def test_episode_enrichment_path_single_feed(tmp_path: Path) -> None:
    bundle = _bundle_single_feed(tmp_path)
    p = episode_enrichment_path(bundle, "topic_cooccurrence.json")
    assert p.parent.name == "enrichments"
    assert p.parent.parent.name == "metadata"
    assert p.name == "0001 - ep.topic_cooccurrence.json"


def test_episode_enrichment_path_multi_feed_stays_with_feed(tmp_path: Path) -> None:
    bundle = _bundle_multi_feed(tmp_path)
    p = episode_enrichment_path(bundle, "insight_density.json")
    # Episode-scope enrichment co-located with its feed's metadata/.
    parts = p.parts
    assert "rss_example_com_a1b2" in parts
    assert "enrichments" in parts
    assert p.name == "0001 - ep.insight_density.json"


def test_episode_enrichment_path_naming_uses_stem(tmp_path: Path) -> None:
    bundle = _bundle_single_feed(tmp_path)
    p = episode_enrichment_path(bundle, "topic_cooccurrence.json")
    assert p.name.startswith(bundle.stem + ".")


# ---------------------------------------------------------------------------
# corpus_enrichment_path
# ---------------------------------------------------------------------------


def test_corpus_enrichment_path_writes_under_corpus_root(tmp_path: Path) -> None:
    p = corpus_enrichment_path(tmp_path, "temporal_velocity.json")
    assert p.parent == tmp_path / "enrichments"
    assert p.name == "temporal_velocity.json"


def test_corpus_enrichment_path_spans_all_feeds_in_multi_feed_layout(
    tmp_path: Path,
) -> None:
    """Corpus-scope always lands at the top-level corpus root, NOT inside a feed."""
    p = corpus_enrichment_path(tmp_path, "grounding_rate.json")
    assert "feeds" not in p.parts
    assert p == tmp_path / "enrichments" / "grounding_rate.json"


# ---------------------------------------------------------------------------
# ensure_directory
# ---------------------------------------------------------------------------


def test_ensure_directory_creates_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    ensure_directory(target)
    assert target.is_dir()


def test_ensure_directory_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    ensure_directory(target)  # no exception
    assert target.is_dir()


# ---------------------------------------------------------------------------
# .viewer/ helper paths
# ---------------------------------------------------------------------------


def test_viewer_dir_is_under_corpus_root(tmp_path: Path) -> None:
    assert viewer_dir(tmp_path) == tmp_path / ".viewer"


def test_enrichment_health_path_is_in_viewer_dir(tmp_path: Path) -> None:
    p = enrichment_health_path(tmp_path)
    assert p == tmp_path / ".viewer" / "enrichment_health.json"


def test_enrichment_status_path_is_in_viewer_dir(tmp_path: Path) -> None:
    p = enrichment_status_path(tmp_path)
    assert p == tmp_path / ".viewer" / "enrichment_status.json"


def test_enrichment_run_summary_path_is_in_enrichments_dir(tmp_path: Path) -> None:
    p = enrichment_run_summary_path(tmp_path)
    assert p == tmp_path / "enrichments" / "run_summary.json"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "writes",
    ["topic_cooccurrence.json", "temporal_velocity.json", "nli_contradiction.json"],
)
def test_enrichment_paths_always_under_enrichments_dir(tmp_path: Path, writes: str) -> None:
    """No enrichment output ever lands outside an ``enrichments/`` directory."""
    bundle = _bundle_single_feed(tmp_path)
    ep_path = episode_enrichment_path(bundle, writes)
    co_path = corpus_enrichment_path(tmp_path, writes)
    assert "enrichments" in ep_path.parts
    assert "enrichments" in co_path.parts


def test_enrichment_paths_never_overwrite_core_artifacts(tmp_path: Path) -> None:
    """Episode-scope output stays under ``metadata/enrichments/`` always."""
    bundle = _bundle_single_feed(tmp_path)
    p = episode_enrichment_path(bundle, "x.json")
    # First ``metadata`` dir walking up MUST be ``metadata/enrichments``'s parent.
    assert p.parent.name == "enrichments"
    assert p.parent.parent.name == "metadata"
