"""Every layout the pipeline WRITES must be a layout the pipeline can READ.

``discover_metadata_files`` is the single source of truth for corpus membership — indexing,
digest, topic-clusters, enrichment, catalog and staleness all read it — so a layout it misses is
invisible to every one of them simultaneously, and silently: "no episodes" is a legitimate state,
so nothing errors.

The gap this pins: a single-feed run (``--rss ... --output-dir X``) writes
``X/run_<id>/metadata/``. The walk that discovers nested run dirs was gated on ``feeds/`` existing,
so that corpus fell to the flat branch, which looks only at ``X/metadata``. Ingesting one feed into
a fresh directory produced a corpus where enrichment reported ``no_bundles`` and the index built
nothing — verified against a real 6-episode DGX ingest, which returned 0 from all three discovery
entry points before the fix and 5/6/5 after.

Layouts are enumerated in one place here so adding a fourth means adding a case, rather than
discovering the omission from an empty corpus months later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.paths import discover_episode_bundles
from podcast_scraper.search.corpus_scope import (
    discover_all_metadata_files,
    discover_metadata_files,
)

pytestmark = pytest.mark.unit


def _write_episode(meta_dir: Path, stem: str, episode_id: str) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "showx"},
                "episode": {"episode_id": episode_id, "title": stem},
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / f"{stem}.kg.json").write_text(
        json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
    )


#: (name, builder) for every layout the pipeline produces.
def _flat(root: Path) -> None:
    """Fixture shape: <root>/metadata/"""
    _write_episode(root / "metadata", "0001-a", "ep-a")


def _multi_feed(root: Path) -> None:
    """Prod shape: <root>/feeds/<feed>/run_<id>/metadata/"""
    _write_episode(root / "feeds" / "showx" / "run_20260101_000000" / "metadata", "0001-a", "ep-a")


def _single_feed(root: Path) -> None:
    """What ``--rss --output-dir X`` writes: <root>/run_<id>/metadata/"""
    _write_episode(root / "run_20260101-000000" / "metadata", "0001-a", "ep-a")


@pytest.mark.parametrize(
    "build", [_flat, _multi_feed, _single_feed], ids=["flat", "multi_feed", "single_feed"]
)
def test_every_written_layout_is_discoverable(build, tmp_path: Path) -> None:
    build(tmp_path)
    assert len(discover_metadata_files(tmp_path)) == 1, (
        f"{build.__doc__} is not discoverable — indexing, enrichment, digest, catalog and "
        "staleness would all see an empty corpus, with no error"
    )


@pytest.mark.parametrize(
    "build", [_flat, _multi_feed, _single_feed], ids=["flat", "multi_feed", "single_feed"]
)
def test_the_cumulative_view_agrees_with_the_scoped_one(build, tmp_path: Path) -> None:
    """The two entry points must not disagree about what a corpus contains."""
    build(tmp_path)
    assert len(discover_all_metadata_files(tmp_path)) == 1


@pytest.mark.parametrize(
    "build", [_flat, _multi_feed, _single_feed], ids=["flat", "multi_feed", "single_feed"]
)
def test_enrichment_sees_every_layout(build, tmp_path: Path) -> None:
    """Enrichment reads through the same rule; ``no_bundles`` was the visible symptom."""
    build(tmp_path)
    assert len(discover_episode_bundles(tmp_path)) == 1


def test_a_single_feed_corpus_with_several_runs_keeps_the_newest(tmp_path: Path) -> None:
    """The newest-run-per-episode rule has to hold for this layout too, not just feeds/."""
    _write_episode(tmp_path / "run_20260101-000000" / "metadata", "0001-a", "ep-a")
    _write_episode(tmp_path / "run_20260202-000000" / "metadata", "0001-a", "ep-a")
    scoped = discover_metadata_files(tmp_path)
    assert len(scoped) == 1, f"reprocessing double-counted the episode: {scoped}"
    assert "20260202" in str(scoped[0]), "the older run won"
    # The cumulative view deliberately keeps both.
    assert len(discover_all_metadata_files(tmp_path)) == 2


def test_an_empty_corpus_is_still_empty(tmp_path: Path) -> None:
    """The mirror: a permissive rule must not start inventing episodes."""
    (tmp_path / "run_20260101-000000").mkdir(parents=True)
    assert discover_metadata_files(tmp_path) == []
    assert discover_episode_bundles(tmp_path) == []
