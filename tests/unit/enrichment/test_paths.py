"""Unit tests for ``enrichment.paths``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.paths import (
    corpus_enrichment_path,
    discover_episode_bundles,
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
    ["topic_cooccurrence.json", "temporal_velocity.json", "topic_consensus.json"],
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


# ---------------------------------------------------------------------------
# discover_episode_bundles
# ---------------------------------------------------------------------------


def _write_episode(
    meta_dir: Path,
    stem: str,
    *,
    guid: str,
    write_gi: bool = True,
    write_kg: bool = True,
    write_bridge: bool = True,
) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.metadata.json").write_text(
        '{"episode": {"guid": "' + guid + '"}}', encoding="utf-8"
    )
    if write_gi:
        (meta_dir / f"{stem}.gi.json").write_text("{}", encoding="utf-8")
    if write_kg:
        (meta_dir / f"{stem}.kg.json").write_text("{}", encoding="utf-8")
    if write_bridge:
        (meta_dir / f"{stem}.bridge.json").write_text("{}", encoding="utf-8")


def test_discover_episode_bundles_multi_feed_layout(tmp_path: Path) -> None:
    feed_a = tmp_path / "feeds" / "rss_feed_a" / "run_20260101-000000" / "metadata"
    feed_b = tmp_path / "feeds" / "rss_feed_b" / "run_20260101-000000" / "metadata"
    _write_episode(feed_a, "0001 - ep one", guid="guid-A1")
    _write_episode(feed_a, "0002 - ep two", guid="guid-A2")
    _write_episode(feed_b, "0001 - ep three", guid="guid-B1")

    bundles = discover_episode_bundles(tmp_path)
    stems = sorted(b.stem for b in bundles)
    assert stems == ["0001 - ep one", "0001 - ep three", "0002 - ep two"]
    guids = sorted(b.episode_id for b in bundles)
    assert guids == ["guid-A1", "guid-A2", "guid-B1"]
    for b in bundles:
        assert b.gi_path is not None and b.gi_path.is_file()
        assert b.kg_path is not None and b.kg_path.is_file()
        assert b.bridge_path is not None and b.bridge_path.is_file()


def test_discover_episode_bundles_reprocess_supersedes_older_run(tmp_path: Path) -> None:
    """Multiple ``run_*`` dirs, SAME episode reprocessed → only the newest run's bundle.

    Identity dedup by ``(feed_id, episode_id)``: a reprocess (same guid) supersedes its older
    "trophy" run. Distinct episodes scattered across runs are instead unioned (see the
    ``corpus_scope`` discovery tests) — enrichment now covers every distinct episode.
    """
    feed = tmp_path / "feeds" / "rss_feed_x"
    older = feed / "run_20260101-000000" / "metadata"
    newer = feed / "run_20260601-000000" / "metadata"
    _write_episode(older, "0001 - stale", guid="guid-SAME")
    _write_episode(newer, "0001 - fresh", guid="guid-SAME")

    bundles = discover_episode_bundles(tmp_path)
    assert len(bundles) == 1
    assert bundles[0].episode_id == "guid-SAME"
    assert "run_20260601-000000" in bundles[0].metadata_path.parts


def test_discover_episode_bundles_missing_siblings_left_none(tmp_path: Path) -> None:
    """Sibling artifacts left None when absent — enrichers can short-circuit."""
    meta = tmp_path / "feeds" / "rss_feed" / "run_20260101-000000" / "metadata"
    _write_episode(
        meta,
        "0001 - no gi",
        guid="guid-only-meta",
        write_gi=False,
        write_kg=False,
        write_bridge=False,
    )

    bundles = discover_episode_bundles(tmp_path)
    assert len(bundles) == 1
    b = bundles[0]
    assert b.gi_path is None
    assert b.kg_path is None
    assert b.bridge_path is None


def test_discover_episode_bundles_empty_corpus_returns_empty(tmp_path: Path) -> None:
    bundles = discover_episode_bundles(tmp_path)
    assert bundles == []


def test_discover_episode_bundles_falls_back_on_unreadable_metadata(tmp_path: Path) -> None:
    """Corrupt metadata.json yields episode_id=stem (no crash)."""
    meta = tmp_path / "feeds" / "rss_feed" / "run_20260101-000000" / "metadata"
    meta.mkdir(parents=True)
    (meta / "0001 - corrupt.metadata.json").write_text("not json {", encoding="utf-8")
    bundles = discover_episode_bundles(tmp_path)
    assert len(bundles) == 1
    assert bundles[0].episode_id == "0001 - corrupt"


# --- the id the catalog joins on (#1927 follow-up) --------------------------------------------


def _write_meta(root, stem: str, episode: dict) -> None:
    d = root / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": "showx"}, "episode": episode}), encoding="utf-8"
    )


def test_bundle_id_uses_episode_id_when_there_is_no_guid(tmp_path) -> None:
    """THE regression: a guid-less episode must still join back to the catalog.

    ``corpus_catalog._feed_and_episode_ids`` keys on ``episode.episode_id``. This chain read
    ``episode.guid`` and then fell straight through to the FILENAME STEM, skipping
    ``episode.episode_id`` entirely — so for any episode without a guid the ids could never be
    equal, and every consumer joining enricher rows to the catalog silently found nothing.

    The live consumer is ``feed_signals._show_grounding``: a miss makes it return ``None`` and the
    Show rail's grounding block just disappears, which reads as "no data for this show" rather
    than "the join is broken". Measured over ``tests/fixtures/**`` at the time of the fix: 102 of
    135 metadata files carried ``episode_id`` and no ``guid``.
    """
    _write_meta(tmp_path, "0001 - some episode", {"episode_id": "ep-abc123"})
    (bundle,) = discover_episode_bundles(tmp_path)
    assert bundle.episode_id == "ep-abc123", (
        "fell back to the filename stem instead of the id the catalog joins on"
    )


def test_guid_still_wins_when_both_are_present(tmp_path) -> None:
    """Ordering is unchanged for the prod-written case, so this fix cannot re-key an existing
    corpus. Measured over the same 135 fixtures: where both keys exist they are equal (0
    mismatches), so the preference is only a tie-break in practice."""
    _write_meta(tmp_path, "0002 - another", {"guid": "guid-1", "episode_id": "eid-1"})
    (bundle,) = discover_episode_bundles(tmp_path)
    assert bundle.episode_id == "guid-1"


def test_stem_remains_the_last_resort(tmp_path) -> None:
    """An episode with neither id still gets a stable, non-empty key."""
    _write_meta(tmp_path, "0003 - no ids", {"title": "No ids here"})
    (bundle,) = discover_episode_bundles(tmp_path)
    assert bundle.episode_id == "0003 - no ids"
