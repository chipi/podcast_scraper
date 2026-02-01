#!/usr/bin/env python3
"""Promote a run to baseline or reference.

This script implements the promotion workflow where runs can be promoted
to baselines or references after execution. Promotion = move + lock + declare intent.

Usage:
    python scripts/promote_run.py \
        --run-id run_2026-01-16_11-52-03 \
        --as baseline \
        --promoted-id baseline_prod_authority_v2 \
        --reason "New production baseline with improved preprocessing"

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
        dataset_id: Dataset ID (required for references)
        reference_quality: Reference quality (silver/gold, optional for references)

    Raises:
        FileNotFoundError: If run doesn't exist
        ValueError: If promoted_id already exists or invalid role
    """
    # Validate role
    if role not in ("baseline", "reference"):
        raise ValueError(f"Invalid role: {role}. Must be 'baseline' or 'reference'")

    # Validate dataset_id for references
    if role == "reference" and not dataset_id:
        raise ValueError("dataset_id is required when promoting to reference")

    # Find source run
    run_path = runs_dir / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run not found: {run_path}")

    # Determine destination
    if role == "baseline":
        dest_dir = baselines_dir / promoted_id
    else:  # reference
        # References are organized by dataset: references/{dataset_id}/{reference_id}/
        dest_dir = references_dir / dataset_id / promoted_id

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

    logger.info(f"✓ Run promoted successfully to {role}: {dest_dir}")


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
        )
    except Exception as e:
        logger.error(f"Promotion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
