#!/usr/bin/env python3
"""Promote a run to baseline or reference.

This script implements the promotion workflow where runs can be promoted
to baselines or references after execution. Promotion = move + lock + declare intent.

Usage:
    python scripts/promote_run.py \
        --run-id run_2026-01-16_11-52-03 \
        --as baseline \
        --promoted-id baseline_prod_authority_v2 \
        --reason "New production baseline with improved preprocessing" \
        [--rename-to baseline_ml_dev_authority_smoke_v1]

    python scripts/promote_run.py \
        --run-id run_2026-01-16_11-52-03 \
        --as reference \
        --promoted-id silver_gpt5_2_v1 \
        --dataset-id curated_5feeds_benchmark_v1 \
        --reason "Silver reference using GPT-5 for benchmark dataset"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_baseline_metadata(baseline_path: Path) -> dict:
    """Load baseline.json metadata.

    Args:
        baseline_path: Path to baseline directory

    Returns:
        Baseline metadata dictionary
    """
    baseline_json = baseline_path / "baseline.json"
    if baseline_json.exists():
        return json.loads(baseline_json.read_text(encoding="utf-8"))
    return {}


def determine_task_type(run_path: Path) -> Optional[str]:
    """Determine task type from run's metrics.json or predictions.jsonl.

    Args:
        run_path: Path to run directory

    Returns:
        Task type ("summarization", "ner_entities", or None if cannot determine)
    """
    # Try metrics.json first
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            task = metrics.get("task")
            if task:
                return task
        except Exception:
            pass

    # Try predictions.jsonl
    predictions_path = run_path / "predictions.jsonl"
    if predictions_path.exists():
        try:
            with open(predictions_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.strip():
                    pred = json.loads(first_line)
                    output = pred.get("output", {})
                    if "entities" in output:
                        return "ner_entities"
                    elif "summary_final" in output or "summary_long" in output:
                        return "summarization"
        except Exception:
            pass

    return None


def rename_baseline(
    baseline_path: Path,
    old_id: str,
    new_id: str,
    role: Literal["baseline", "reference"],
) -> Path:
    """Rename a baseline/reference and update all metadata files.

    Args:
        baseline_path: Current path to baseline/reference directory
        old_id: Current baseline/reference ID
        new_id: New baseline/reference ID
        role: Role (baseline or reference)

    Returns:
        New path to renamed baseline/reference directory

    Raises:
        ValueError: If new_id already exists
        FileNotFoundError: If baseline_path doesn't exist
    """
    # Determine new path
    if role == "baseline":
        new_path = baseline_path.parent / new_id
    else:  # reference
        # References follow new structure:
        # - Silver: references/silver/{reference_id}/
        # - Gold: references/gold/{task_type}/{reference_id}/
        # Keep same parent structure (silver/ or gold/{task_type}/)
        new_path = baseline_path.parent / new_id

    # Check if new path already exists
    if new_path.exists():
        raise ValueError(
            f"{role.capitalize()} '{new_id}' already exists at {new_path}. " f"Use a different ID."
        )

    logger.info(f"Renaming {role} '{old_id}' to '{new_id}'")
    logger.info(f"  Old path: {baseline_path}")
    logger.info(f"  New path: {new_path}")

    # Rename directory
    baseline_path.rename(new_path)

    # Update baseline.json
    baseline_json = new_path / "baseline.json"
    if baseline_json.exists():
        metadata = json.loads(baseline_json.read_text(encoding="utf-8"))
        metadata["promoted_id"] = new_id
        baseline_json.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Updated baseline.json: promoted_id -> {new_id}")

    # Update README.md
    readme_path = new_path / "README.md"
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        # Replace baseline title
        content = content.replace(f"# Baseline: {old_id}", f"# Baseline: {new_id}")
        content = content.replace(f"# Reference: {old_id}", f"# Reference: {new_id}")
        readme_path.write_text(content, encoding="utf-8")
        logger.info(f"Updated README.md: baseline name -> {new_id}")

    # Update fingerprint.json
    fingerprint_path = new_path / "fingerprint.json"
    if fingerprint_path.exists():
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if "run_context" in fingerprint and "baseline_id" in fingerprint["run_context"]:
            fingerprint["run_context"]["baseline_id"] = new_id
            fingerprint_path.write_text(
                json.dumps(fingerprint, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Updated fingerprint.json: baseline_id -> {new_id}")

    # Update metrics_report.md if it exists
    metrics_report_path = new_path / "metrics_report.md"
    if metrics_report_path.exists():
        content = metrics_report_path.read_text(encoding="utf-8")
        # Replace run_id references with baseline_id
        content = content.replace(
            f"**Run ID:** `{old_id}`",
            f"**Baseline ID:** `{new_id}`\n**Original Run ID:** `{old_id}`",
        )
        # Also handle if it already has Baseline ID format
        if f"**Baseline ID:** `{old_id}`" in content:
            content = content.replace(
                f"**Baseline ID:** `{old_id}`",
                f"**Baseline ID:** `{new_id}`",
            )
        metrics_report_path.write_text(content, encoding="utf-8")
        logger.info(f"Updated metrics_report.md: baseline ID -> {new_id}")

    # Update config.yaml id field if it exists
    config_path = new_path / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            config_content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_content and "id" in config_content:
                config_content["id"] = new_id
                config_path.write_text(
                    yaml.dump(config_content, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )
                logger.info(f"Updated config.yaml: id -> {new_id}")
        except ImportError:
            logger.warning("PyYAML not available, cannot update config.yaml id field")
        except Exception as e:
            logger.warning(f"Failed to update config.yaml id field: {e}")

    logger.info(f"✓ {role.capitalize()} renamed successfully: {new_path}")
    return new_path


def create_promotion_readme(
    promoted_path: Path,
    run_id: str,
    promoted_id: str,
    role: Literal["baseline", "reference"],
    reason: str,
    dataset_id: Optional[str] = None,
    reference_quality: Optional[str] = None,
) -> None:
    """Create README.md explaining the promotion.

    Args:
        promoted_path: Path to promoted directory
        run_id: Original run ID
        promoted_id: New promoted ID
        role: Role (baseline or reference)
        reason: Reason for promotion
        dataset_id: Dataset ID (for references)
        reference_quality: Reference quality (silver/gold, for references)
    """
    readme_path = promoted_path / "README.md"
    promoted_at = datetime.utcnow().isoformat()

    if role == "baseline":
        # Load baseline promotion template
        template_path = Path("data/eval/baselines/BASELINE_PROMOTION_README_TEMPLATE.md")
        if not template_path.exists():
            raise FileNotFoundError(
                f"Baseline promotion README template not found: {template_path}"
            )
        template_content = template_path.read_text(encoding="utf-8")

        # Render template
        content = template_content.format(
            promoted_id=promoted_id,
            run_id=run_id,
            promoted_at=promoted_at,
            reason=reason,
        )
    else:  # reference
        # Load reference promotion template
        template_path = Path("data/eval/references/REFERENCE_PROMOTION_README_TEMPLATE.md")
        if not template_path.exists():
            raise FileNotFoundError(
                f"Reference promotion README template not found: {template_path}"
            )
        template_content = template_path.read_text(encoding="utf-8")

        # Format quality section
        quality_section = ""
        if reference_quality:
            quality_section = f"\n**Reference Quality:** {reference_quality}\n"

        # Render template
        content = template_content.format(
            promoted_id=promoted_id,
            run_id=run_id,
            promoted_at=promoted_at,
            dataset_id=dataset_id or "N/A",
            quality_section=quality_section,
            reason=reason,
        )

    readme_path.write_text(content, encoding="utf-8")
    logger.info(f"Created README.md: {readme_path}")


def find_config_file(run_path: Path, run_id: str) -> Optional[Path]:
    """Find the experiment config file used for a run.

    Tries multiple strategies:
    1. Check run's README.md for config path
    2. Try configs/{run_id}.yaml or .yml
    3. Try configs/{baseline_id}.yaml or .yml (if run_id looks like baseline_id)

    Args:
        run_path: Path to run directory
        run_id: Run ID

    Returns:
        Path to config file if found, None otherwise
    """
    # Strategy 1: Check README.md for config path
    readme_path = run_path / "README.md"
    if readme_path.exists():
        try:
            content = readme_path.read_text(encoding="utf-8")
            # Look for "Config:" line
            for line in content.splitlines():
                if line.strip().startswith("- **Config:**") or line.strip().startswith(
                    "**Config:**"
                ):
                    # Extract path (might be in backticks or plain text)
                    config_str = line.split(":", 1)[1].strip().strip("`").strip()
                    if config_str and config_str != "N/A":
                        config_path = Path(config_str)
                        # Try as-is first
                        if config_path.exists():
                            return config_path
                        # Try relative to project root
                        project_root = Path.cwd()
                        abs_path = project_root / config_path
                        if abs_path.exists():
                            return abs_path
        except Exception as e:
            logger.debug(f"Failed to parse README for config path: {e}")

    # Strategy 2: Try configs/{run_id}.yaml, .yml, or no extension
    configs_dir = Path("data/eval/configs")
    # Try with extensions first
    for ext in [".yaml", ".yml"]:
        config_path = configs_dir / f"{run_id}{ext}"
        if config_path.exists():
            return config_path
    # Try without extension (some configs don't have extensions)
    config_path = configs_dir / run_id
    if config_path.exists() and config_path.is_file():
        return config_path

    logger.debug(f"Could not find config file for run {run_id}")
    return None


def copy_config_file(
    run_path: Path, dest_dir: Path, run_id: str, new_id: Optional[str] = None
) -> None:
    """Copy experiment config file to baseline/reference directory.

    Args:
        run_path: Path to source run directory
        dest_dir: Path to destination baseline/reference directory
        run_id: Run ID (used to find config)
        new_id: Optional new ID to set in config file (for consistency with baseline name)
    """
    config_path = find_config_file(run_path, run_id)
    if not config_path:
        logger.warning(
            f"Could not find config file for run {run_id}. "
            f"Baseline will not include config.yaml. "
            f"To include it, ensure the config file exists at data/eval/configs/{run_id}.yaml"
        )
        return

    dest_config = dest_dir / "config.yaml"
    try:
        shutil.copy2(config_path, dest_config)
        logger.info(f"Copied config file: {config_path} -> {dest_config}")

        # Update id field in config if new_id is provided
        if new_id:
            try:
                import yaml

                config_content = yaml.safe_load(dest_config.read_text(encoding="utf-8"))
                if config_content and "id" in config_content:
                    config_content["id"] = new_id
                    dest_config.write_text(
                        yaml.dump(config_content, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
                    logger.info(f"Updated config.yaml id field: {run_id} -> {new_id}")
            except ImportError:
                logger.warning("PyYAML not available, cannot update config id field")
            except Exception as e:
                logger.warning(f"Failed to update config id field: {e}")
    except Exception as e:
        logger.warning(f"Failed to copy config file: {e}")


def promote_run(
    run_id: str,
    role: Literal["baseline", "reference"],
    promoted_id: str,
    reason: str,
    runs_dir: Path = Path("data/eval/runs"),
    baselines_dir: Path = Path("data/eval/baselines"),
    references_dir: Path = Path("data/eval/references"),
    dataset_id: Optional[str] = None,
    reference_quality: Optional[str] = None,
    rename_to: Optional[str] = None,
) -> None:
    """Promote a run to baseline or reference.

    Promotion = move + lock + declare intent

    Args:
        run_id: Original run ID (in runs/)
        role: Role to promote to (baseline or reference)
        promoted_id: New ID for promoted run
        reason: Reason for promotion (documented in README.md)
        runs_dir: Directory containing runs
        baselines_dir: Directory for baselines
        references_dir: Directory for references
        dataset_id: Dataset ID (optional, for backward compatibility)
        reference_quality: Reference quality (silver/gold, optional for references)

    Raises:
        FileNotFoundError: If run doesn't exist
        ValueError: If promoted_id already exists or invalid role
    """
    # Validate role
    if role not in ("baseline", "reference"):
        raise ValueError(f"Invalid role: {role}. Must be 'baseline' or 'reference'")

    # dataset_id is optional for references (task type auto-detected from run)
    # Only used for backward compatibility or explicit override

    # Find source run
    run_path = runs_dir / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run not found: {run_path}")

    # Determine destination
    if role == "baseline":
        dest_dir = baselines_dir / promoted_id
    else:  # reference
        # Determine task type to choose correct reference structure
        task_type = determine_task_type(run_path)
        reference_quality_lower = (reference_quality or "").lower()

        if reference_quality_lower == "gold":
            # Gold references: references/gold/{task_type}/{reference_id}/
            if task_type == "ner_entities":
                dest_dir = references_dir / "gold" / "ner_entities" / promoted_id
            elif task_type == "summarization":
                dest_dir = references_dir / "gold" / "summarization" / promoted_id
            else:
                # Default to summarization if task type cannot be determined
                logger.warning(
                    f"Could not determine task type for run {run_id}, "
                    "defaulting to summarization for gold reference"
                )
                dest_dir = references_dir / "gold" / "summarization" / promoted_id
        else:
            # Silver references (default): references/silver/{reference_id}/
            # No dataset_id folder, no task_type folder
            dest_dir = references_dir / "silver" / promoted_id

    # Check if destination already exists (immutability)
    if dest_dir.exists():
        raise ValueError(
            f"Promoted {role} '{promoted_id}' already exists at {dest_dir}. "
            f"Promoted runs are immutable. Use a different ID."
        )

    # Create destination directory
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Promoting run '{run_id}' to {role} '{promoted_id}'")
    logger.info(f"  Source: {run_path}")
    logger.info(f"  Destination: {dest_dir}")
    logger.info(f"  Reason: {reason}")

    # Copy all artifacts (move would be better, but copy is safer)
    # We'll remove source after successful copy
    shutil.copytree(run_path, dest_dir)

    # Update baseline.json with promotion metadata
    baseline_json = dest_dir / "baseline.json"
    if baseline_json.exists():
        metadata = json.loads(baseline_json.read_text(encoding="utf-8"))
        metadata["promoted_from"] = run_id
        metadata["promoted_at"] = datetime.utcnow().isoformat() + "Z"
        metadata["promoted_as"] = role
        metadata["promoted_id"] = promoted_id
        if role == "reference":
            metadata["reference_quality"] = reference_quality
        baseline_json.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Copy config file to make baseline self-contained
    # IMPORTANT: If rename_to is provided, use it for the config id field to ensure
    # consistency between the baseline directory name and the config file's id field.
    # This ensures the config.yaml id matches the final baseline name.
    config_id = rename_to if rename_to else promoted_id
    copy_config_file(run_path, dest_dir, run_id, new_id=config_id)

    # Create README.md
    create_promotion_readme(
        dest_dir,
        run_id,
        promoted_id,
        role,
        reason,
        dataset_id,
        reference_quality,
    )

    # Remove source run (promotion is one-way)
    logger.info(f"Removing source run: {run_path}")
    shutil.rmtree(run_path)

    # Rename if requested
    final_dir = dest_dir
    if rename_to:
        final_dir = rename_baseline(dest_dir, promoted_id, rename_to, role)
        promoted_id = rename_to  # Update for final log message

    logger.info(f"✓ Run promoted successfully to {role}: {final_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Promote a run to baseline or reference (RFC-041 promotion workflow)."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Original run ID (in data/eval/runs/)",
    )
    parser.add_argument(
        "--as",
        type=str,
        choices=["baseline", "reference"],
        required=True,
        dest="role",
        help="Role to promote to (baseline or reference)",
    )
    parser.add_argument(
        "--promoted-id",
        type=str,
        required=True,
        help="New ID for promoted run (e.g., 'baseline_prod_authority_v2')",
    )
    parser.add_argument(
        "--reason",
        type=str,
        required=True,
        help="Reason for promotion (documented in README.md)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset ID (required for references, optional for baselines)",
    )
    parser.add_argument(
        "--reference-quality",
        type=str,
        choices=["silver", "gold"],
        default=None,
        help="Reference quality (silver/gold, for references only)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="data/eval/runs",
        help="Directory containing runs (default: data/eval/runs)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default="data/eval/baselines",
        help="Directory for baselines (default: data/eval/baselines)",
    )
    parser.add_argument(
        "--references-dir",
        type=str,
        default="data/eval/references",
        help="Directory for references (default: data/eval/references)",
    )
    parser.add_argument(
        "--rename-to",
        type=str,
        default=None,
        help="Optional: Rename baseline/reference after promotion to this ID",
    )

    args = parser.parse_args()

    try:
        promote_run(
            run_id=args.run_id,
            role=args.role,  # type: ignore
            promoted_id=args.promoted_id,
            reason=args.reason,
            runs_dir=Path(args.runs_dir),
            baselines_dir=Path(args.baselines_dir),
            references_dir=Path(args.references_dir),
            dataset_id=args.dataset_id,
            reference_quality=args.reference_quality,
            rename_to=args.rename_to,
        )
    except Exception as e:
        logger.error(f"Promotion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
