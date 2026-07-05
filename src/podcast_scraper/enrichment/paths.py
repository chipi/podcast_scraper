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

import json
from pathlib import Path

from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle


def episode_enrichment_path(bundle: EpisodeArtifactBundle, writes: str) -> Path:
    """Resolve the on-disk path for an episode-scope enricher output.

    Naming convention: ``{stem}.{enricher_writes}``.

    Example:
        bundle.metadata_path = ``metadata/0001 - ep.metadata.json``
        bundle.stem          = ``0001 - ep``
        writes               = ``insight_density.json``
        result               = ``metadata/enrichments/0001 - ep.insight_density.json``
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


def discover_episode_bundles(corpus_root: Path) -> list[EpisodeArtifactBundle]:
    """Discover every episode bundle under *corpus_root* (latest run per feed).

    Walks the corpus using ``search.corpus_scope.discover_metadata_files`` so
    the same safe-walk + latest-run-per-feed dedup that the indexer uses also
    applies to enrichment. For each ``*.metadata.json``, the sibling
    ``{stem}.gi.json`` / ``{stem}.kg.json`` / ``{stem}.bridge.json`` files
    are attached when present; missing siblings stay ``None`` so enrichers
    can short-circuit per their ``manifest.reads`` declaration.

    Episode id resolution prefers the metadata's ``episode.guid``, then
    ``episode_id``, then ``guid``, falling back to the filename stem.
    """
    from podcast_scraper.search.corpus_scope import discover_metadata_files

    bundles: list[EpisodeArtifactBundle] = []
    for meta_path in discover_metadata_files(corpus_root):
        name = meta_path.name
        if not name.endswith(".metadata.json"):
            continue
        stem = name[: -len(".metadata.json")]
        meta_dir = meta_path.parent
        gi = meta_dir / f"{stem}.gi.json"
        kg = meta_dir / f"{stem}.kg.json"
        bridge = meta_dir / f"{stem}.bridge.json"
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        episode = payload.get("episode") if isinstance(payload, dict) else None
        episode_id = ""
        if isinstance(episode, dict):
            episode_id = str(episode.get("guid") or "")
        if not episode_id and isinstance(payload, dict):
            episode_id = str(payload.get("episode_id") or payload.get("guid") or "")
        if not episode_id:
            episode_id = stem
        bundles.append(
            EpisodeArtifactBundle(
                metadata_path=meta_path,
                gi_path=gi if gi.is_file() else None,
                kg_path=kg if kg.is_file() else None,
                bridge_path=bridge if bridge.is_file() else None,
                episode_id=episode_id,
                stem=stem,
            )
        )
    return bundles
