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
    episodes: List[EpisodeIndexEntry] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize episodes list if None."""
        if self.episodes is None:
            self.episodes = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert index to dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "feed_url": self.feed_url,
            "created_at": self.created_at,
            "episodes_processed": self.episodes_processed,
            "episodes_failed": self.episodes_failed,
            "episodes_skipped": self.episodes_skipped,
            "episodes": [asdict(ep) for ep in self.episodes],
        }

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
        logger.info(f"Run index saved to: {filepath}")


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


def create_run_index(
    run_id: str,
    feed_url: Optional[str],
    episodes: List[Any],  # models.Episode
    effective_output_dir: str,
    episode_statuses: Optional[List[Any]] = None,
    run_suffix: Optional[str] = None,
) -> RunIndex:
    """Create a run index from processed episodes.

    Args:
        run_id: Run identifier
        feed_url: RSS feed URL
        episodes: List of episodes that were processed
        effective_output_dir: Output directory path
        episode_statuses: Optional list of episode status objects (from metrics)

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

        # Find transcript and metadata files (check multiple patterns including run_suffix)
        episode_title_safe = getattr(episode, "title_safe", episode.title)

        # Find transcript file
        transcript_path = _find_transcript_file(
            episode, episode_title_safe, transcripts_dir, effective_output_dir, run_suffix
        )

        # Find metadata file
        metadata_path = _find_metadata_file(
            episode, episode_title_safe, metadata_dir, effective_output_dir, run_suffix
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
        schema_version="1.0.0",
        run_id=run_id,
        feed_url=feed_url,
        created_at=created_at,
        episodes_processed=episodes_processed,
        episodes_failed=episodes_failed,
        episodes_skipped=episodes_skipped,
        episodes=index_entries,
    )

    return run_index
