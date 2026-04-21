#!/usr/bin/env python3
"""Materialize a dataset from dataset JSON.

This script:
- Reads a dataset JSON file
- Copies transcripts from sources/ to materialized/{dataset_id}/
- Validates that paths resolve and hashes match
- Writes a small meta.json per episode
- Ensures materialization is reproducible

This proves:
- Dataset JSON is correct
- Paths resolve correctly
- Hashes match expected values
- Materialization is reproducible

Usage:
    python scripts/materialize_dataset.py --dataset-id curated_5feeds_smoke_v1
        --dataset-file data/eval/datasets/curated_5feeds_smoke_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

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


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text.

    Args:
        text: Text content

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_materialized_dataset_readme(
    materialized_dir: Path,
    dataset_id: str,
    dataset_file: Path,
) -> None:
    """Create README.md for a materialized dataset directory using template.

    Args:
        materialized_dir: Path to materialized dataset directory
        dataset_id: Dataset identifier
        dataset_file: Path to source dataset JSON file
    """
    # Load template
    template_path = Path("data/eval/materialized/MATERIALIZED_DATASET_README_TEMPLATE.md")
    if not template_path.exists():
        raise FileNotFoundError(f"Materialized dataset README template not found: {template_path}")
    template_content = template_path.read_text(encoding="utf-8")

    # Determine source path (try to find relative path)
    try:
        source_path_rel = str(dataset_file.relative_to(Path.cwd()))
    except ValueError:
        source_path_rel = str(dataset_file)

    # Render template
    content = template_content.format(
        dataset_id=dataset_id,
        source_path_rel=source_path_rel,
    )

    readme_path = materialized_dir / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    logger.info(f"Created README.md: {readme_path}")


def materialize_dataset(dataset_file: Path, output_base: Path) -> None:  # noqa: C901
    """Materialize a dataset from dataset JSON.

    Args:
        dataset_file: Path to dataset JSON file
        output_base: Base directory for materialized datasets (e.g., data/eval/materialized)

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If hash mismatch or other validation error
        RuntimeError: If materialization fails
    """
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    logger.info(f"Loading dataset: {dataset_file}")

    # Load dataset JSON
    try:
        dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file: {e}") from e

    dataset_id = dataset.get("dataset_id")
    if not dataset_id:
        raise ValueError("Dataset JSON missing 'dataset_id' field")

    episodes = dataset.get("episodes", [])
    if not episodes:
        raise ValueError(f"Dataset '{dataset_id}' has no episodes")

    logger.info(f"Dataset: {dataset_id}, Episodes: {len(episodes)}")

    # Create materialized directory
    materialized_dir = output_base / dataset_id
    if materialized_dir.exists():
        logger.warning(f"Materialized directory already exists: {materialized_dir}")
        logger.info("Removing existing materialized directory for clean materialization...")
        shutil.rmtree(materialized_dir)

    materialized_dir.mkdir(parents=True, exist_ok=True)

    # Process each episode
    errors = []
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not episode_id:
            errors.append("Episode missing 'episode_id' field")
            continue

        transcript_path_str = episode.get("transcript_path")
        if not transcript_path_str:
            errors.append(f"Episode {episode_id} missing 'transcript_path'")
            continue

        expected_hash = episode.get("transcript_hash")
        if not expected_hash:
            errors.append(f"Episode {episode_id} missing 'transcript_hash'")
            continue

        # Resolve transcript path
        transcript_path = Path(transcript_path_str)
        if not transcript_path.is_absolute():
            # Try relative to current working directory
            transcript_path = Path.cwd() / transcript_path

        if not transcript_path.exists():
            errors.append(f"Episode {episode_id}: transcript file not found: {transcript_path}")
            continue

        # Validate hash
        logger.debug(f"Validating hash for episode {episode_id}...")
        try:
            transcript_content = transcript_path.read_text(encoding="utf-8")
            actual_hash = hash_text(transcript_content)
        except Exception as e:
            errors.append(f"Episode {episode_id}: failed to read/hash transcript: {e}")
            continue

        if actual_hash != expected_hash:
            error_msg = (
                f"Episode {episode_id}: HASH MISMATCH - transcript file has been modified!\n"
                f"  Expected hash: {expected_hash}\n"
                f"  Actual hash:   {actual_hash}\n"
                f"  File:          {transcript_path}\n"
                f"  This indicates the transcript file has changed since the dataset was created."
            )
            logger.error(error_msg)
            errors.append(
                f"Episode {episode_id}: hash mismatch "
                f"(expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}...)"
            )
            continue

        # Copy transcript to materialized directory (flat structure)
        materialized_transcript = materialized_dir / f"{episode_id}.txt"
        try:
            shutil.copy2(transcript_path, materialized_transcript)
            logger.debug(f"Copied transcript: {materialized_transcript}")
        except Exception as e:
            errors.append(f"Episode {episode_id}: failed to copy transcript: {e}")
            continue

        # Try to load speakers, expectations, and metadata_version from source metadata file
        from podcast_scraper.evaluation.metadata_validator import validate_episode_metadata

        speakers = None
        expectations = None
        metadata_version = "1.0"  # Default version
        source_metadata_candidates = [
            transcript_path.parent / f"{transcript_path.stem}.metadata.json",
            transcript_path.parent / "metadata.json",
        ]
        for candidate in source_metadata_candidates:
            if candidate.exists():
                try:
                    source_metadata = json.loads(candidate.read_text(encoding="utf-8"))
                    # Validate metadata structure with assertions
                    validate_episode_metadata(source_metadata, episode_id)
                    if "speakers" in source_metadata:
                        speakers = source_metadata["speakers"]
                    if "expectations" in source_metadata:
                        expectations = source_metadata["expectations"]
                    if "metadata_version" in source_metadata:
                        metadata_version = source_metadata["metadata_version"]
                    if speakers or expectations:
                        logger.debug(f"Loaded metadata from {candidate} for {episode_id}")
                        break
                except AssertionError as e:
                    logger.error(
                        f"Metadata validation failed for {episode_id} from {candidate}: {e}"
                    )
                    # Continue to next candidate or use defaults
                    continue
                except Exception as e:
                    logger.debug(f"Failed to load metadata from {candidate}: {e}")
                    continue

        # Create episode metadata JSON (metadata_version first for schema versioning)
        episode_meta = {
            "metadata_version": metadata_version,
            "episode_id": episode_id,
            "transcript_path": f"{episode_id}.txt",
            "transcript_hash": actual_hash,
            "source_transcript_path": str(transcript_path),
            "preprocessing_profile": episode.get("preprocessing_profile", "cleaning_v3"),
        }

        # Add optional fields
        if "title" in episode:
            episode_meta["title"] = episode["title"]
        if "transcript_raw_path" in episode:
            episode_meta["transcript_raw_path"] = episode["transcript_raw_path"]
        if "golden_summary_long_path" in episode:
            episode_meta["golden_summary_long_path"] = episode["golden_summary_long_path"]
        if "golden_summary_short_path" in episode:
            episode_meta["golden_summary_short_path"] = episode["golden_summary_short_path"]
        if "duration_minutes" in episode:
            episode_meta["duration_minutes"] = episode["duration_minutes"]

        # Add speakers and expectations if found in source metadata
        if speakers:
            episode_meta["speakers"] = speakers
        if expectations:
            episode_meta["expectations"] = expectations

        # Write episode metadata
        episode_meta_file = materialized_dir / f"{episode_id}.meta.json"
        episode_meta_file.write_text(
            json.dumps(episode_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug(f"Created metadata: {episode_meta_file}")

    # Check for errors
    if errors:
        error_summary = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Materialization failed with {len(errors)} error(s):\n{error_summary}")

    # Create dataset-level metadata
    try:
        source_dataset_file = str(dataset_file.relative_to(Path.cwd()))
    except ValueError:
        # If relative_to fails, use absolute path or just the filename
        if dataset_file.is_absolute():
            source_dataset_file = str(dataset_file)
        else:
            source_dataset_file = str(dataset_file)

    dataset_meta = {
        "dataset_id": dataset_id,
        "source_dataset_file": source_dataset_file,
        "num_episodes": len(episodes),
        "materialized_at": dataset.get("created_at"),  # Use dataset creation time
        "episodes": [
            {
                "episode_id": ep.get("episode_id"),
                "transcript_path": f"{ep.get('episode_id')}.txt",
                "meta_path": f"{ep.get('episode_id')}.meta.json",
            }
            for ep in episodes
        ],
    }

    dataset_meta_file = materialized_dir / "meta.json"
    dataset_meta_file.write_text(
        json.dumps(dataset_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Materialized dataset: {materialized_dir}")
    logger.info(f"Dataset metadata: {dataset_meta_file}")
    # Create README.md
    create_materialized_dataset_readme(
        materialized_dir=materialized_dir,
        dataset_id=dataset_id,
        dataset_file=dataset_file,
    )

    logger.info(f"Successfully materialized {len(episodes)} episode(s)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Materialize a dataset from dataset JSON (validation and file copying)."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Dataset identifier (e.g., 'curated_5feeds_smoke_v1')",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to dataset JSON file (default: data/eval/datasets/{dataset_id}.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval/materialized",
        help="Base directory for materialized datasets (default: data/eval/materialized)",
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

    # Determine dataset file path
    if args.dataset_file:
        dataset_file = Path(args.dataset_file)
    else:
        dataset_file = Path("data/eval/datasets") / f"{args.dataset_id}.json"

    output_base = Path(args.output_dir)

    # Materialize dataset
    try:
        materialize_dataset(dataset_file, output_base)
    except Exception as e:
        logger.error(f"Materialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
