#!/usr/bin/env python3
"""Create canonical dataset JSON from existing eval data (RFC-041 Phase 0).

This script:
- Scans existing eval data directories
- Creates canonical dataset JSON files
- Computes content hashes for reproducibility
- Outputs to benchmarks/datasets/

Usage:
    python scripts/create_dataset_json.py --dataset-id indicator_v1 --eval-dir data/eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_episodes(eval_dir: Path) -> List[Dict[str, Any]]:  # noqa: C901
    """Discover episodes from eval directory structure.

    Recursively finds all .txt files in subdirectories and treats each as a transcript.
    Episode ID is derived from the filename (without extension).

    Args:
        eval_dir: Path to eval directory (e.g., data/eval)

    Returns:
        List of episode dictionaries
    """
    episodes = []
    seen_ids = set()

    # Recursively find all .txt files
    for txt_file in sorted(eval_dir.rglob("*.txt")):
        # Skip hidden files/directories
        if any(part.startswith(".") for part in txt_file.parts):
            continue

        # Use filename (without extension) as episode_id
        episode_id = txt_file.stem

        # Handle duplicate episode IDs by including parent directory name
        if episode_id in seen_ids:
            # Include parent directory name to make it unique
            parent_name = txt_file.parent.name
            episode_id = f"{parent_name}_{episode_id}"
            # If still duplicate, include more path components
            if episode_id in seen_ids:
                # Use relative path from eval_dir as episode_id
                try:
                    rel_path = txt_file.relative_to(eval_dir)
                    episode_id = (
                        str(rel_path).replace("/", "_").replace("\\", "_").replace(".txt", "")
                    )
                except ValueError:
                    # Fallback: use full path hash
                    episode_id = f"ep_{hash_text(str(txt_file))[:8]}"

        seen_ids.add(episode_id)

        # Use relative paths from project root
        try:
            transcript_path_rel = str(txt_file.relative_to(Path.cwd()))
        except ValueError:
            # Fallback to absolute if relative fails
            transcript_path_rel = str(txt_file)

        # Compute content hash
        try:
            transcript_hash = hash_text(txt_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read {txt_file}: {e}")
            continue

        # Look for related files in the same directory
        transcript_dir = txt_file.parent
        transcript_raw = None
        summary_long = None
        summary_short = None
        metadata_file = None

        # Try to find metadata JSON file (new format: *.metadata.json)
        stem = txt_file.stem
        metadata_candidates = [
            transcript_dir / f"{stem}.metadata.json",  # New format: p01_e01.metadata.json
            transcript_dir / "metadata.json",  # Old format: metadata.json in directory
        ]
        for candidate in metadata_candidates:
            if candidate.exists():
                metadata_file = candidate
                break

        # Try to find related files (same stem, different suffix)
        for related_file in transcript_dir.glob(f"{stem}.*"):
            if related_file == txt_file:
                continue
            if related_file.suffix == ".txt":
                # Check if it's a raw transcript or summary
                name_lower = related_file.name.lower()
                if "raw" in name_lower:
                    transcript_raw = related_file
                elif "long" in name_lower or "gold.long" in name_lower:
                    summary_long = related_file
                elif "short" in name_lower or "gold.short" in name_lower:
                    summary_short = related_file

        # Load metadata if available
        episode_metadata = {}
        if metadata_file and metadata_file.exists():
            try:
                metadata_content = json.loads(metadata_file.read_text(encoding="utf-8"))
                # Validate metadata structure with assertions
                from podcast_scraper.evaluation.metadata_validator import validate_episode_metadata

                validate_episode_metadata(metadata_content, episode_id)
                # Handle both old format (nested) and new format (flat)
                if "episode" in metadata_content:
                    # Old format: nested structure
                    episode_metadata = metadata_content
                else:
                    # New format: flat structure, convert to old format for compatibility
                    episode_metadata = {
                        "episode": {
                            "title": metadata_content.get("episode_title"),
                            "episode_id": metadata_content.get("source_episode_id"),
                        },
                        "feed": {
                            "title": metadata_content.get("feed_name"),
                            "url": metadata_content.get("feed_url"),
                            "language": metadata_content.get("language"),
                        },
                        "content": {
                            "duration_seconds": metadata_content.get("duration_seconds"),
                        },
                    }
            except AssertionError as e:
                logger.error(f"Metadata validation failed for {episode_id}: {e}")
                # Continue without metadata rather than failing entire dataset creation
            except Exception as e:
                logger.warning(f"Failed to load metadata for {episode_id}: {e}")

        episode = {
            "episode_id": episode_id,
            "title": episode_metadata.get("episode", {}).get("title", f"Episode {episode_id}"),
            "transcript_path": transcript_path_rel,
            "transcript_hash": transcript_hash,
            "preprocessing_profile": "cleaning_v3",  # Default
        }

        # Add optional paths
        if transcript_raw and transcript_raw.exists():
            try:
                episode["transcript_raw_path"] = str(transcript_raw.relative_to(Path.cwd()))
            except ValueError:
                episode["transcript_raw_path"] = str(transcript_raw)
        if summary_long and summary_long.exists():
            try:
                episode["golden_summary_long_path"] = str(summary_long.relative_to(Path.cwd()))
            except ValueError:
                episode["golden_summary_long_path"] = str(summary_long)
        if summary_short and summary_short.exists():
            try:
                episode["golden_summary_short_path"] = str(summary_short.relative_to(Path.cwd()))
            except ValueError:
                episode["golden_summary_short_path"] = str(summary_short)

        # Add duration if available
        if "content" in episode_metadata:
            duration_seconds = episode_metadata["content"].get("duration_seconds")
            if duration_seconds:
                episode["duration_minutes"] = round(duration_seconds / 60, 1)

        episodes.append(episode)

    return episodes


def filter_episodes_by_feed(
    episodes: List[Dict[str, Any]], max_episodes_per_feed: int = 1
) -> List[Dict[str, Any]]:
    """Filter episodes to keep first N episodes per feed.

    Args:
        episodes: List of episode dictionaries
        max_episodes_per_feed: Maximum number of episodes to keep per feed (default: 1)

    Returns:
        Filtered list with first N episodes per feed
    """
    # Group episodes by feed (extract from episode_id or transcript_path)
    feed_episodes: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        episode_id = episode.get("episode_id", "")
        transcript_path = episode.get("transcript_path", "")

        # Extract feed name from episode_id (e.g., "p01_e01" -> "feed-p01")
        feed = None
        if "_" in episode_id:
            # Try to extract feed prefix (e.g., "p01_e01" -> "feed-p01")
            prefix = episode_id.split("_")[0]
            if prefix.startswith("p") and len(prefix) == 3:
                feed = f"feed-{prefix}"

        # Fallback: extract from transcript_path
        if not feed:
            path_parts = transcript_path.split("/")
            for part in path_parts:
                if part.startswith("feed-"):
                    feed = part
                    break

        if not feed:
            # If we can't determine feed, include it anyway
            feed = "unknown"

        if feed not in feed_episodes:
            feed_episodes[feed] = []
        feed_episodes[feed].append(episode)

    # Keep first N episodes from each feed (sorted by episode_id)
    filtered = []
    for feed, feed_eps in sorted(feed_episodes.items()):
        # Sort episodes by episode_id and take first N
        sorted_eps = sorted(feed_eps, key=lambda x: x.get("episode_id", ""))
        selected = sorted_eps[:max_episodes_per_feed]
        filtered.extend(selected)
        logger.debug(
            f"Feed {feed}: selected {len(selected)} episode(s) (from {len(sorted_eps)} total)"
        )

    return filtered


def create_dataset_json(
    dataset_id: str,
    eval_dir: Path,
    description: str,
    content_regime: str = "explainer",
    max_episodes_per_feed: Optional[int] = None,
) -> Dict[str, Any]:
    """Create canonical dataset JSON.

    Args:
        dataset_id: Dataset identifier (e.g., "indicator_v1")
        eval_dir: Path to eval directory
        description: Human-readable description
        content_regime: Content regime type (e.g., "explainer", "narrative", "science")
        max_episodes_per_feed: If set, limit to first N episodes per feed (None = all episodes)

    Returns:
        Dataset dictionary
    """
    episodes = discover_episodes(eval_dir)

    if not episodes:
        raise ValueError(f"No episodes found in {eval_dir}")

    # Filter episodes per feed if requested
    if max_episodes_per_feed is not None:
        original_count = len(episodes)
        episodes = filter_episodes_by_feed(episodes, max_episodes_per_feed)
        logger.info(
            f"Filtered to {len(episodes)} episode(s) (first {max_episodes_per_feed} per feed, "
            f"from {original_count} total)"
        )

    dataset = {
        "dataset_id": dataset_id,
        "version": "1.0",
        "description": description,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "content_regime": content_regime,
        "num_episodes": len(episodes),
        "episodes": episodes,
    }

    return dataset


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create canonical dataset JSON from existing eval data (RFC-041 Phase 0)."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Dataset identifier (e.g., 'indicator_v1')",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="data/eval",
        help="Path to eval directory (default: data/eval)",
    )
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Human-readable description of the dataset",
    )
    parser.add_argument(
        "--content-regime",
        type=str,
        default="explainer",
        choices=["explainer", "narrative", "science"],
        help="Content regime type (default: explainer)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/datasets",
        help="Output directory for dataset JSON (default: benchmarks/datasets)",
    )
    parser.add_argument(
        "--max-episodes-per-feed",
        type=int,
        default=None,
        help="Limit to first N episodes per feed (for smoke/benchmark datasets)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error(f"Eval directory not found: {eval_dir}")
        sys.exit(1)

    # Create dataset JSON
    try:
        dataset = create_dataset_json(
            dataset_id=args.dataset_id,
            eval_dir=eval_dir,
            description=args.description,
            content_regime=args.content_regime,
            max_episodes_per_feed=args.max_episodes_per_feed,
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        sys.exit(1)

    # Write dataset JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset_id}.json"

    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Dataset JSON created: {output_path}")
    logger.info(f"Episodes: {dataset['num_episodes']}")


if __name__ == "__main__":
    main()
