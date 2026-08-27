"""Run index generation for episode tracking.

This module creates index.json files that list all processed episodes
with their status, paths, and error information.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EpisodeIndexEntry:
    """Entry in run index for a single episode."""

    episode_id: str
    status: str  # "ok", "failed", "skipped"
    transcript_path: Optional[str] = None
    metadata_path: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stage: Optional[str] = None


@dataclass
class RunIndex:
    """Run index listing all processed episodes."""

    schema_version: str = "1.0.0"
    run_id: str = ""
    feed_url: Optional[str] = None
    created_at: str = ""
    episodes_processed: int = 0
    episodes_failed: int = 0
    episodes_skipped: int = 0
    pipeline_append: bool = False
    episodes: List[EpisodeIndexEntry] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize episodes list if None."""
        if self.episodes is None:
            self.episodes = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert index to dictionary."""
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "feed_url": self.feed_url,
            "created_at": self.created_at,
            "episodes_processed": self.episodes_processed,
            "episodes_failed": self.episodes_failed,
            "episodes_skipped": self.episodes_skipped,
            "episodes": [asdict(ep) for ep in self.episodes],
        }
        if self.pipeline_append:
            out["pipeline_append"] = True
        return out

    def to_json(self) -> str:
        """Convert index to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save index to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        index_json = self.to_json()
        filepath.write_text(index_json, encoding="utf-8")
        logger.debug("Run index saved to: %s", filepath)


def _build_status_map(episode_statuses: Optional[List[Any]]) -> Dict[str, Dict[str, Any]]:
    """Build status map from episode_statuses if available.

    Args:
        episode_statuses: Optional list of episode status objects

    Returns:
        Dictionary mapping episode_id to status info
    """
    status_map: Dict[str, Dict[str, Any]] = {}
    if episode_statuses:
        for status in episode_statuses:
            episode_id = getattr(status, "episode_id", None)
            if episode_id:
                status_map[episode_id] = {
                    "status": getattr(status, "status", "ok"),
                    "error_type": getattr(status, "error_type", None),
                    "error_message": getattr(status, "error_message", None),
                    "stage": getattr(status, "stage", None),
                }
    return status_map


def _find_transcript_file(
    episode: Any,
    episode_title_safe: str,
    transcripts_dir: str,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> Optional[str]:
    """Find transcript file for an episode.

    Args:
        episode: Episode object
        episode_title_safe: Safe episode title for filename
        transcripts_dir: Transcripts directory path
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix

    Returns:
        Relative path to transcript file or None if not found
    """
    if not os.path.exists(transcripts_dir):
        return None

    # Build base pattern (with or without run_suffix)
    if run_suffix:
        base_pattern = f"{episode.idx:04d} - {episode_title_safe}_{run_suffix}"
    else:
        base_pattern = f"{episode.idx:04d} - {episode_title_safe}"

    # Try exact match first (with run_suffix if provided)
    for ext in [".txt", ".md", ".html", ".vtt", ".srt"]:
        potential_path = os.path.join(transcripts_dir, f"{base_pattern}{ext}")
        if os.path.exists(potential_path):
            return os.path.relpath(potential_path, effective_output_dir)

    # If not found, try glob search (handles run_suffix variations)
    from pathlib import Path

    pattern_without_suffix = f"{episode.idx:04d} - {episode_title_safe}"
    for candidate in Path(transcripts_dir).glob(f"{pattern_without_suffix}*"):
        if candidate.is_file() and candidate.suffix in [
            ".txt",
            ".md",
            ".html",
            ".vtt",
            ".srt",
        ]:
            return os.path.relpath(str(candidate), effective_output_dir)

    return None


def _find_metadata_file(
    episode: Any,
    episode_title_safe: str,
    metadata_dir: str,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> Optional[str]:
    """Find metadata file for an episode.

    Args:
        episode: Episode object
        episode_title_safe: Safe episode title for filename
        metadata_dir: Metadata directory path
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix

    Returns:
        Relative path to metadata file or None if not found
    """
    # Check standard metadata directory
    if os.path.exists(metadata_dir):
        from pathlib import Path

        # Build base pattern (with or without run_suffix)
        if run_suffix:
            base_pattern = f"{episode.idx:04d} - {episode_title_safe}_{run_suffix}"
        else:
            base_pattern = f"{episode.idx:04d} - {episode_title_safe}"

        # Try exact match first (with run_suffix if provided)
        for ext in [".json", ".yaml", ".yml"]:
            potential_path = os.path.join(metadata_dir, f"{base_pattern}.metadata{ext}")
            if os.path.exists(potential_path):
                return os.path.relpath(potential_path, effective_output_dir)

        # If not found, try glob search (handles run_suffix variations)
        pattern_without_suffix = f"{episode.idx:04d} - {episode_title_safe}"
        for candidate in Path(metadata_dir).glob(f"{pattern_without_suffix}*.metadata.*"):
            if candidate.is_file():
                return os.path.relpath(str(candidate), effective_output_dir)

    # Also check for custom metadata_subdirectory (if it exists)
    try:
        from ..utils import filesystem

        for subdir_name in os.listdir(effective_output_dir):
            subdir_path = os.path.join(effective_output_dir, subdir_name)
            if (
                os.path.isdir(subdir_path)
                and subdir_name != filesystem.TRANSCRIPTS_SUBDIR
                and subdir_name != filesystem.METADATA_SUBDIR
            ):
                # Could be a custom metadata subdirectory
                from pathlib import Path

                pattern_without_suffix = f"{episode.idx:04d} - {episode_title_safe}"
                for candidate in Path(subdir_path).glob(f"{pattern_without_suffix}*.metadata.*"):
                    if candidate.is_file():
                        return os.path.relpath(str(candidate), effective_output_dir)
    except Exception:
        # Ignore errors when scanning directories
        pass

    return None


@dataclass(frozen=True)
class CorpusMetadataEntry:
    """One on-disk episode located by its STABLE identity (guid / episode_id), not feed order."""

    metadata_rel: str  # path relative to the corpus output_dir
    idx: int  # the NNNN prefix its transcript/metadata are stored under
    guid: Optional[str]
    episode_id: Optional[str]


# realpath(output_dir) -> {"by_guid": {...}, "by_id": {...}}. Skip-existing/append run this per
# episode; the corpus is stable within a run, so cache the scan. Keyed by realpath, and cleared by
# reset_corpus_metadata_index_cache_for_tests() so a reused tmp path can't leak across tests (#22).
_CORPUS_METADATA_INDEX_CACHE: Dict[str, Dict[str, Dict[str, CorpusMetadataEntry]]] = {}


def invalidate_corpus_metadata_index_cache() -> None:
    """Drop the cached corpus-metadata index so the next read re-scans from disk.

    Production callers use this after MUTATING a corpus on disk (e.g. the rollback DELETE) to force
    a fresh post-delete view. It clears the whole cache — fine because it is realpath-keyed and a
    rebuild is cheap; targeted per-corpus invalidation is a future refinement (harden #3).
    """
    _CORPUS_METADATA_INDEX_CACHE.clear()


def reset_corpus_metadata_index_cache_for_tests() -> None:
    """Test-isolation alias for :func:`invalidate_corpus_metadata_index_cache`."""
    invalidate_corpus_metadata_index_cache()


def _leading_idx(name: str) -> Optional[int]:
    digits = ""
    for ch in name:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _scan_corpus_metadata_index(output_dir: str) -> Dict[str, Dict[str, CorpusMetadataEntry]]:
    """Scan the corpus once, mapping guid AND episode_id -> :class:`CorpusMetadataEntry`.

    This is the drift-immune replacement for idx-derived filename lookups: the feed's enumerate
    position shifts whenever a feed grows, so "already processed?" and the append metadata lookup
    must resolve by the stable guid/episode_id instead (generalizes ``_on_disk_guid_index``).
    """
    root = Path(output_dir)
    by_guid: Dict[str, CorpusMetadataEntry] = {}
    by_id: Dict[str, CorpusMetadataEntry] = {}
    # Works at BOTH the feed output dir (run_*/metadata) used by skip-existing during a run, AND
    # the corpus root (feeds/<slug>/run_*/metadata) used by the rollback DELETE API.
    for pattern in (
        "metadata/*.metadata.*",
        "run_*/metadata/*.metadata.*",
        "feeds/*/metadata/*.metadata.*",
        "feeds/*/run_*/metadata/*.metadata.*",
    ):
        # codeql[py/path-injection] -- request path anchor-guarded (Type 1; CODEQL_DISMISSALS.md).
        for meta_path in sorted(root.glob(pattern)):
            if meta_path.name.startswith("._") or not meta_path.is_file():
                continue  # macOS AppleDouble
            try:
                text = meta_path.read_text(encoding="utf-8")
                data = (
                    yaml.safe_load(text)
                    if meta_path.suffix in (".yaml", ".yml")
                    else json.loads(text)
                )
            except (OSError, json.JSONDecodeError, yaml.YAMLError):
                continue
            episode = data.get("episode", {}) if isinstance(data, dict) else {}
            if not isinstance(episode, dict):
                continue
            idx = _leading_idx(meta_path.name)
            if idx is None:
                continue
            entry = CorpusMetadataEntry(
                metadata_rel=os.path.relpath(str(meta_path), output_dir),
                idx=idx,
                guid=(episode.get("guid") or None),
                episode_id=(episode.get("episode_id") or None),
            )
            guid = episode.get("guid")
            if isinstance(guid, str) and guid.strip():
                g = guid.strip()
                if g in by_guid:
                    # First-writer-wins, but a duplicate guid on disk (re-published / re-added
                    # episode) means callers that act on ONE entry (rollback episode delete) may
                    # leave a copy behind — surface it rather than resolve silently.
                    logger.warning(
                        "corpus_metadata_index: duplicate guid %s (%s and %s); keeping first",
                        g,
                        by_guid[g].metadata_rel,
                        entry.metadata_rel,
                    )
                else:
                    by_guid[g] = entry
            eid = episode.get("episode_id")
            if isinstance(eid, str) and eid.strip():
                e = eid.strip()
                if e in by_id:
                    logger.warning(
                        "corpus_metadata_index: duplicate episode_id %s (%s and %s); keeping first",
                        e,
                        by_id[e].metadata_rel,
                        entry.metadata_rel,
                    )
                else:
                    by_id[e] = entry
    return {"by_guid": by_guid, "by_id": by_id}


def corpus_metadata_index(output_dir: str) -> Dict[str, Dict[str, CorpusMetadataEntry]]:
    """Cached :func:`_scan_corpus_metadata_index` (per realpath output_dir)."""
    key = os.path.realpath(output_dir)
    if key not in _CORPUS_METADATA_INDEX_CACHE:
        _CORPUS_METADATA_INDEX_CACHE[key] = _scan_corpus_metadata_index(output_dir)
    return _CORPUS_METADATA_INDEX_CACHE[key]


def _episode_guid(episode: Any) -> Optional[str]:
    item = getattr(episode, "item", None)
    if item is None:
        return None
    try:
        elem = item.find("guid")
    except Exception:
        return None
    if elem is not None and getattr(elem, "text", None):
        return str(elem.text).strip() or None
    return None


def resolve_ondisk_idx_for_episode(episode: Any, output_dir: str) -> int:
    """The idx an episode's transcript/metadata are stored under on disk.

    Resolves by STABLE guid so skip-existing survives feed growth (the run-local ``episode.idx``
    shifts when a feed publishes between runs → silent reprocess + duplicates). Falls back to the
    run-local idx when the guid is unknown (a genuinely new episode).
    """
    guid = _episode_guid(episode)
    if guid:
        entry = corpus_metadata_index(output_dir)["by_guid"].get(guid)
        if entry is not None:
            return entry.idx
    return int(episode.idx)


def corpus_root_from_cfg(cfg: Any) -> Optional[str]:
    """True corpus root for BOTH corpus-run layouts, else ``None``.

    A run participates in a corpus two ways, and every corpus-wide lookup must resolve the
    SAME root for both. In BOTH, by the time the pipeline runs, ``cfg.output_dir`` is usually
    already the FEED leaf ``<corpus>/feeds/<slug>`` — so the shape check (parent dir literally
    named ``feeds``, lift two levels up) is the primary branch and the flag branch is a
    fallback for a flag-set cfg whose output_dir was not wrapped:

    - ``--single-feed-uses-corpus-layout`` (per-feed jobs): the config post-validator wraps
      ``output_dir`` to the feed leaf.
    - batch mode (``--feeds-spec`` / ``rss_urls`` — the nightly): BOTH multi-feed loops rebase
      each child cfg's ``output_dir`` to the feed leaf with the flag left False — the cli's own
      loop (cli.py, ``corpus_feed_output_dir(corpus_parent, url)``; the prod nightly path) and
      ``service.run_multi_feed`` (service.py, ``"output_dir": child_dir``).

    2026-08-27: every flag-only gate was blind in batch mode — skip-existing re-ingested the
    nightly's whole window twice over. This helper is the single resolution point so no call
    site re-derives the root from its own assumptions again.
    """
    out = getattr(cfg, "output_dir", None)
    if not out:
        return None
    p = Path(str(out)).resolve()
    if p.parent.name == "feeds" and p.parent.parent.name:
        return str(p.parent.parent)
    if getattr(cfg, "single_feed_uses_corpus_layout", False):
        return str(p)
    return None


def episode_metadata_rel_in_corpus(episode: Any, corpus_root: str) -> Optional[str]:
    """Return the metadata path (relative to *corpus_root*) if this episode already exists ANYWHERE
    in the corpus (by STABLE guid), else ``None`` (D7).

    Corpus-wide: ``corpus_metadata_index`` globs ``feeds/*/run_*/metadata/*`` across ALL runs, so an
    episode processed under a PRIOR run dir is found even when the current run writes a fresh run
    dir (``--single-feed-uses-corpus-layout``). Without this, skip-existing scoped to the fresh
    (empty) run dir silently re-transcribes an already-present episode.
    """
    guid = _episode_guid(episode)
    if not guid:
        return None
    entry = corpus_metadata_index(corpus_root)["by_guid"].get(guid)
    if entry is None:
        return None
    if os.path.isfile(os.path.join(corpus_root, entry.metadata_rel)):
        return entry.metadata_rel
    return None


def existing_transcript_path_in_corpus(episode: Any, corpus_root: str) -> Optional[str]:
    """Absolute path to this episode's existing transcript (or its metadata, as a presence marker)
    ANYWHERE in the corpus, else ``None`` (D7). Used by the corpus-layout skip-existing path so the
    reuse/segment-backfill/skip sub-cases resolve against the real prior-run artifact.
    """
    meta_rel = episode_metadata_rel_in_corpus(episode, corpus_root)
    if meta_rel is None:
        return None
    meta_abs = Path(corpus_root) / meta_rel
    stem = meta_abs.name
    base = stem
    for suffix in (".metadata.json", ".metadata.yaml", ".metadata.yml"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            break
    transcripts_dir = meta_abs.parent.parent / "transcripts"
    if transcripts_dir.is_dir():
        preferred = transcripts_dir / f"{base}.txt"
        if preferred.is_file():
            return str(preferred)
        for candidate in sorted(transcripts_dir.glob(f"{base}.*")):
            if candidate.is_file():
                return str(candidate)
    # Metadata present but no transcript file located → the metadata itself marks the episode as
    # processed (skip-existing only needs presence; the path is used for logging).
    return str(meta_abs)


def find_episode_metadata_relative_path(
    episode: Any,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> Optional[str]:
    """Locate an episode metadata file under *effective_output_dir* if present.

    Resolves by the STABLE guid first (drift-immune — the feed's enumerate position shifts when a
    feed grows), then falls back to the idx-derived filename lookup for episodes with no guid.

    Args:
        episode: Episode model instance
        effective_output_dir: Run output root (contains ``metadata/``)
        run_suffix: Whisper/metadata filename suffix for this run, if any

    Returns:
        Path relative to *effective_output_dir*, or ``None`` if not found.
    """
    from ..utils import filesystem

    guid = _episode_guid(episode)
    if guid:
        entry = corpus_metadata_index(effective_output_dir)["by_guid"].get(guid)
        if entry is not None and os.path.isfile(
            os.path.join(effective_output_dir, entry.metadata_rel)
        ):
            return entry.metadata_rel

    metadata_dir = os.path.join(effective_output_dir, filesystem.METADATA_SUBDIR)
    title_for_paths = filesystem.truncate_whisper_title(
        getattr(episode, "title_safe", episode.title), for_log=False
    )
    return _find_metadata_file(
        episode, title_for_paths, metadata_dir, effective_output_dir, run_suffix
    )


def _determine_episode_status(
    metadata_path: Optional[str],
    transcript_path: Optional[str],
    status_from_map: Optional[str],
    episode: Any,
) -> str:
    """Determine episode status from filesystem and status_map.

    Args:
        metadata_path: Path to metadata file or None
        transcript_path: Path to transcript file or None
        status_from_map: Status from status_map or None
        episode: Episode object

    Returns:
        Status string: "ok", "failed", or "skipped"
    """
    # Rule: metadata exists → processed, transcript exists → partially processed,
    # neither → skipped/failed
    if metadata_path:
        # Metadata exists → episode was fully processed
        return "ok"
    elif transcript_path:
        # Transcript exists but no metadata → partially processed
        # (transcribed but not summarized/metadata generated)
        # This could be "ok" if metadata generation was disabled, or "failed" if enabled
        # For now, treat as "ok" since transcript was successfully created
        return "ok"
    else:
        # Neither exists → determine if skipped or failed
        # Use status_map if available, otherwise infer from episode properties
        if status_from_map:
            return status_from_map
        elif not hasattr(episode, "transcript_url") or not episode.transcript_url:
            # No transcript URL → episode was skipped
            return "skipped"
        else:
            # Has transcript URL but no file → failed
            return "failed"


def _extract_episode_metadata_for_id(
    episode: Any,
) -> Tuple[Optional[str], Optional[str], Optional[datetime], Optional[int]]:
    """Extract episode metadata needed for ID generation.

    Args:
        episode: Episode object

    Returns:
        Tuple of (episode_guid, episode_link, episode_published_date, episode_number)
    """
    from ..rss.parser import extract_episode_published_date

    episode_guid = None
    episode_link = None
    episode_published_date = None
    episode_number = getattr(episode, "number", None)

    if hasattr(episode, "item") and episode.item is not None:
        # Extract GUID from RSS item
        guid_elem = episode.item.find("guid")
        if guid_elem is not None and guid_elem.text:
            episode_guid = guid_elem.text.strip()
        # Extract link
        link_elem = episode.item.find("link")
        if link_elem is not None and link_elem.text:
            episode_link = link_elem.text.strip()
        # Extract published date
        episode_published_date = extract_episode_published_date(episode.item)

    return episode_guid, episode_link, episode_published_date, episode_number


def build_failure_summary(index: RunIndex) -> Dict[str, Any]:
    """Aggregate failed episodes by error type for end-of-run reporting.

    Args:
        index: Completed RunIndex with episode entries.

    Returns:
        Dictionary with failure counts grouped by error_type,
        plus a flat list of failed episode IDs.
    """
    by_error: Dict[str, int] = {}
    failed_ids: List[str] = []
    for ep in index.episodes:
        if ep.status != "failed":
            continue
        failed_ids.append(ep.episode_id)
        key = ep.error_type or "unknown"
        by_error[key] = by_error.get(key, 0) + 1

    return {
        "total_failed": len(failed_ids),
        "by_error_type": dict(sorted(by_error.items(), key=lambda kv: kv[1], reverse=True)),
        "failed_episode_ids": failed_ids,
    }


def create_run_index(
    run_id: str,
    feed_url: Optional[str],
    episodes: List[Any],  # models.Episode
    effective_output_dir: str,
    episode_statuses: Optional[List[Any]] = None,
    run_suffix: Optional[str] = None,
    pipeline_append: bool = False,
) -> RunIndex:
    """Create a run index from processed episodes.

    Args:
        run_id: Run identifier
        feed_url: RSS feed URL
        episodes: List of episodes that were processed
        effective_output_dir: Output directory path
        episode_statuses: Optional list of episode status objects (from metrics)
        run_suffix: Optional run suffix used in transcript/metadata filenames
        pipeline_append: When True, mark index as produced under append/resume (GitHub #444)

    Returns:
        RunIndex object
    """
    from .. import models  # noqa: F401

    created_at = datetime.utcnow().isoformat() + "Z"

    # Build status map from episode_statuses if available
    status_map = _build_status_map(episode_statuses)

    # Scan output directory for actual files
    from ..utils import filesystem

    transcripts_dir = os.path.join(effective_output_dir, filesystem.TRANSCRIPTS_SUBDIR)
    metadata_dir = os.path.join(effective_output_dir, filesystem.METADATA_SUBDIR)

    index_entries: List[EpisodeIndexEntry] = []
    episodes_processed = 0
    episodes_failed = 0
    episodes_skipped = 0

    for episode in episodes:
        # Generate episode_id (same logic as metadata generation)
        from .metadata_generation import generate_episode_id

        # Extract episode metadata for ID generation
        episode_guid, episode_link, episode_published_date, episode_number = (
            _extract_episode_metadata_for_id(episode)
        )

        # Generate stable episode ID
        episode_id = generate_episode_id(
            feed_url=feed_url or "",
            episode_title=episode.title,
            episode_guid=episode_guid,
            published_date=episode_published_date,
            episode_link=episode_link,
            episode_number=episode_number,
        )

        # Get status from status_map if available (supplemental info)
        status_info = status_map.get(episode_id, {})
        status_from_map = status_info.get("status")

        # Find transcript and metadata files (check multiple patterns including run_suffix).
        # Filenames use truncate_whisper_title (build_whisper_output_name); match that here.
        episode_title_safe = getattr(episode, "title_safe", episode.title)
        title_for_paths = filesystem.truncate_whisper_title(episode_title_safe, for_log=False)

        # Find transcript file
        transcript_path = _find_transcript_file(
            episode, title_for_paths, transcripts_dir, effective_output_dir, run_suffix
        )

        # Find metadata file
        metadata_path = _find_metadata_file(
            episode, title_for_paths, metadata_dir, effective_output_dir, run_suffix
        )

        # Determine status from filesystem (primary source of truth)
        status = _determine_episode_status(metadata_path, transcript_path, status_from_map, episode)

        # Count by status
        if status == "ok":
            episodes_processed += 1
        elif status == "failed":
            episodes_failed += 1
        elif status == "skipped":
            episodes_skipped += 1

        # Create index entry
        entry = EpisodeIndexEntry(
            episode_id=episode_id,
            status=status,
            transcript_path=transcript_path,
            metadata_path=metadata_path,
            error_type=status_info.get("error_type"),
            error_message=status_info.get("error_message"),
            error_stage=status_info.get("stage"),
        )

        index_entries.append(entry)

    # Create run index
    run_index = RunIndex(
        schema_version="1.1.0" if pipeline_append else "1.0.0",
        run_id=run_id,
        feed_url=feed_url,
        created_at=created_at,
        episodes_processed=episodes_processed,
        episodes_failed=episodes_failed,
        episodes_skipped=episodes_skipped,
        pipeline_append=pipeline_append,
        episodes=index_entries,
    )

    return run_index
