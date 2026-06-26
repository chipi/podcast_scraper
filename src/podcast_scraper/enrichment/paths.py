"""Filesystem paths for enrichment outputs (multi-feed aware).

Per RFC-088 §Directory Structure:

* Episode-scope outputs: ``<bundle.metadata_path.parent>/enrichments/``
* Corpus-scope outputs: ``<corpus_root>/enrichments/``

In multi-feed layouts, episode-scope enrichments stay co-located with
their feed's ``metadata/enrichments/`` directory; corpus-scope outputs
always go to the top-level corpus root (spanning all feeds).

Invariants:

* Core artifacts are never in the ``enrichments/`` directory.
* Enrichment artifacts are never outside the ``enrichments/`` directory.
* ``discover_metadata_files`` (existing) globs ``*.metadata.json`` under
  ``metadata/``, NOT under ``metadata/enrichments/`` — adding enrichment
  outputs cannot accidentally surface as core artifacts.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle


def episode_enrichment_path(bundle: EpisodeArtifactBundle, writes: str) -> Path:
    """Resolve the on-disk path for an episode-scope enricher output.

    Naming convention: ``{stem}.{enricher_writes}``.

    Example:
        bundle.metadata_path = ``metadata/0001 - ep.metadata.json``
        bundle.stem          = ``0001 - ep``
        writes               = ``topic_cooccurrence.json``
        result               = ``metadata/enrichments/0001 - ep.topic_cooccurrence.json``
    """
    parent = bundle.metadata_path.parent / "enrichments"
    return parent / f"{bundle.stem}.{writes}"


def corpus_enrichment_path(corpus_root: Path, writes: str) -> Path:
    """Resolve the on-disk path for a corpus-scope enricher output.

    Example:
        corpus_root = ``output/``
        writes      = ``temporal_velocity.json``
        result      = ``output/enrichments/temporal_velocity.json``
    """
    return corpus_root / "enrichments" / writes


def ensure_directory(path: Path) -> None:
    """Create directory at *path* (and parents) if it doesn't exist.

    Used by ``health.py`` and ``status.py`` to create the ``.viewer/``
    directory on first write — standalone runs against corpora that
    lack a ``.viewer/`` directory work without the operator pre-creating
    it (chunk-1 lock audit §B6).
    """
    path.mkdir(parents=True, exist_ok=True)


def viewer_dir(corpus_root: Path) -> Path:
    """``.viewer/`` under the corpus root — shared with the existing jobs registry."""
    return corpus_root / ".viewer"


def enrichment_health_path(corpus_root: Path) -> Path:
    """``.viewer/enrichment_health.json`` — cross-run health persistence."""
    return viewer_dir(corpus_root) / "enrichment_health.json"


def enrichment_status_path(corpus_root: Path) -> Path:
    """``.viewer/enrichment_status.json`` — live-run progress + heartbeat."""
    return viewer_dir(corpus_root) / "enrichment_status.json"


def enrichment_run_summary_path(corpus_root: Path) -> Path:
    """``enrichments/run_summary.json`` — per-enricher outcomes per run."""
    return corpus_root / "enrichments" / "run_summary.json"
