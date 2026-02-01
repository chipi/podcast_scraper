#!/usr/bin/env python3
"""Generate source index JSON files for inventory management.

This script:
- Scans a source directory for feed subdirectories
- Finds all transcript (.txt) and metadata (.metadata.json) files
- Computes SHA256 hashes for transcripts
- Generates an index.json file in the source directory

The index enables:
- Programmatic dataset generation
- Drift detection (hash changes)
- Dataset definition validation
- Avoiding ad-hoc directory scanning

Usage:
    python scripts/generate_source_index.py --source-dir data/eval/sources/curated_5feeds_raw_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file.

    Args:
        path: File path

    Returns:
        SHA256 hash as hex string
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_transcript_and_metadata(feed_dir: Path, source_dir: Path) -> List[Dict[str, Any]]:
    """Find all transcript and metadata file pairs in a feed directory.

    Args:
        feed_dir: Feed directory (e.g., feed-p01)
        source_dir: Source directory (for relative paths)

    Returns:
        List of episode dictionaries
    """
    episodes = []

    # Find all .txt files (transcripts)
    txt_files = sorted(feed_dir.glob("*.txt"))
    feed_name = feed_dir.name

    for txt_file in txt_files:
        # Skip if it's not a transcript (e.g., if it's a summary)
        if "summary" in txt_file.name.lower() or "gold" in txt_file.name.lower():
            continue

        # Get episode ID from filename (without extension)
        source_episode_id = txt_file.stem

        # Find corresponding metadata file
        metadata_file = feed_dir / f"{source_episode_id}.metadata.json"
        if not metadata_file.exists():
            # Try old format: just metadata.json in directory
            metadata_file = feed_dir / "metadata.json"
            if not metadata_file.exists():
                metadata_file = None

        # Compute transcript hash
        try:
            transcript_hash = hash_file(txt_file)
        except Exception as e:
            logger.warning(f"Failed to hash {txt_file}: {e}")
            continue

        # Get relative paths from source directory
        try:
            transcript_path = str(txt_file.relative_to(source_dir))
        except ValueError:
            transcript_path = str(txt_file)

        meta_path = None
        if metadata_file:
            try:
                meta_path = str(metadata_file.relative_to(source_dir))
            except ValueError:
                meta_path = str(metadata_file)

        episode = {
            "source_episode_id": source_episode_id,
            "feed": feed_name,
            "transcript_path": transcript_path,
            "transcript_sha256": transcript_hash,
        }

        if meta_path:
            episode["meta_path"] = meta_path

        episodes.append(episode)

    return episodes


def generate_source_index(source_dir: Path) -> Dict[str, Any]:
    """Generate index JSON for a source directory.

    Args:
        source_dir: Source directory (e.g., curated_5feeds_raw_v1)

    Returns:
        Index dictionary
    """
    source_id = source_dir.name
    logger.info(f"Generating index for source: {source_id}")

    # Find all feed subdirectories
    feed_dirs = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    feed_dirs = sorted(feed_dirs)

    if not feed_dirs:
        logger.warning(f"No feed subdirectories found in {source_dir}")
        return {
            "source_id": source_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "episodes": [],
        }

    # Collect episodes from all feeds
    all_episodes = []
    for feed_dir in feed_dirs:
        logger.debug(f"Scanning feed: {feed_dir.name}")
        episodes = find_transcript_and_metadata(feed_dir, source_dir)
        all_episodes.extend(episodes)
        logger.info(f"Found {len(episodes)} episode(s) in {feed_dir.name}")

    # Create index
    index = {
        "source_id": source_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "episodes": sorted(all_episodes, key=lambda x: (x["feed"], x["source_episode_id"])),
    }

    return index


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate source index JSON files for inventory management."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help=(
            "Source directory (e.g., data/eval/sources/curated_5feeds_raw_v1) "
            "or parent directory containing multiple sources"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all source directories in the given directory (if parent directory provided)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_dir = Path(args.source_dir)
    if not base_dir.exists():
        logger.error(f"Directory does not exist: {base_dir}")
        sys.exit(1)

    if not base_dir.is_dir():
        logger.error(f"Path is not a directory: {base_dir}")
        sys.exit(1)

    # Determine source directories to process
    if args.all:
        # Find all subdirectories that look like source directories
        source_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not source_dirs:
            logger.warning(f"No source directories found in {base_dir}")
            sys.exit(0)
        logger.info(f"Found {len(source_dirs)} source directory(ies) to process")
    else:
        # Process single directory
        source_dirs = [base_dir]

    # Process each source directory
    total_episodes = 0
    for source_dir in sorted(source_dirs):
        logger.info(f"Processing source: {source_dir.name}")
        try:
            index = generate_source_index(source_dir)
        except Exception as e:
            logger.error(f"Failed to generate index for {source_dir}: {e}", exc_info=True)
            continue

        # Write index.json
        index_file = source_dir / "index.json"
        try:
            index_file.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Created index: {index_file}")
            logger.info(f"Total episodes in {source_dir.name}: {len(index['episodes'])}")
            total_episodes += len(index["episodes"])
        except Exception as e:
            logger.error(f"Failed to write index file {index_file}: {e}")

    if args.all:
        logger.info(
            f"Processed {len(source_dirs)} source directory(ies), total episodes: {total_episodes}"
        )


if __name__ == "__main__":
    main()
