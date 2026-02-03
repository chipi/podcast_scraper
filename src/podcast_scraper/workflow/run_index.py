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
from typing import Any, Dict, List, Optional

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
        from ..rss.parser import extract_episode_published_date
        from .metadata_generation import generate_episode_id

        # Extract episode metadata for ID generation
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

        # Build base pattern (with or without run_suffix)
        if run_suffix:
            base_pattern = f"{episode.idx:04d} - {episode_title_safe}_{run_suffix}"
        else:
            base_pattern = f"{episode.idx:04d} - {episode_title_safe}"

        # Find transcript file
        transcript_path = None
        if os.path.exists(transcripts_dir):
            # Try exact match first (with run_suffix if provided)
            for ext in [".txt", ".md", ".html", ".vtt", ".srt"]:
                potential_path = os.path.join(transcripts_dir, f"{base_pattern}{ext}")
                if os.path.exists(potential_path):
                    transcript_path = os.path.relpath(potential_path, effective_output_dir)
                    break

            # If not found, try glob search (handles run_suffix variations)
            if transcript_path is None:
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
                        transcript_path = os.path.relpath(str(candidate), effective_output_dir)
                        break

        # Find metadata file (check standard location and custom metadata_subdirectory)
        metadata_path = None

        # Check standard metadata directory
        if os.path.exists(metadata_dir):
            from pathlib import Path

            # Try exact match first (with run_suffix if provided)
            for ext in [".json", ".yaml", ".yml"]:
                potential_path = os.path.join(metadata_dir, f"{base_pattern}.metadata{ext}")
                if os.path.exists(potential_path):
                    metadata_path = os.path.relpath(potential_path, effective_output_dir)
                    break

            # If not found, try glob search (handles run_suffix variations)
            if metadata_path is None:
                pattern_without_suffix = f"{episode.idx:04d} - {episode_title_safe}"
                for candidate in Path(metadata_dir).glob(f"{pattern_without_suffix}*.metadata.*"):
                    if candidate.is_file():
                        metadata_path = os.path.relpath(str(candidate), effective_output_dir)
                        break

        # Also check for custom metadata_subdirectory (if it exists)
        if metadata_path is None:
            # Check if there's a custom metadata subdirectory
            try:
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
                        for candidate in Path(subdir_path).glob(
                            f"{pattern_without_suffix}*.metadata.*"
                        ):
                            if candidate.is_file():
                                metadata_path = os.path.relpath(
                                    str(candidate), effective_output_dir
                                )
                                break
                        if metadata_path:
                            break
            except Exception:
                # Ignore errors when scanning directories
                pass

        # Determine status from filesystem (primary source of truth)
        # Rule: metadata exists → processed, transcript exists → partially processed,
        # neither → skipped/failed
        if metadata_path:
            # Metadata exists → episode was fully processed
            status = "ok"
        elif transcript_path:
            # Transcript exists but no metadata → partially processed
            # (transcribed but not summarized/metadata generated)
            # This could be "ok" if metadata generation was disabled, or "failed" if enabled
            # For now, treat as "ok" since transcript was successfully created
            status = "ok"
        else:
            # Neither exists → determine if skipped or failed
            # Use status_map if available, otherwise infer from episode properties
            if status_from_map:
                status = status_from_map
            elif not hasattr(episode, "transcript_url") or not episode.transcript_url:
                # No transcript URL → episode was skipped
                status = "skipped"
            else:
                # Has transcript URL but no file → failed
                status = "failed"

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
