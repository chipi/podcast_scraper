#!/usr/bin/env python3
"""Promote a frozen performance profile to a reference.

Copies a working profile YAML (and optional ``stage_truth.json`` / ``.monitor.log``
companions) into ``data/profiles/references/<promoted_id>.yaml`` with a ``promoted``
metadata block stamped into the YAML.

Usage:
    python scripts/eval/promote_profile.py \
        --source data/profiles/v2.6-wip-openai.yaml \
        --promoted-id v2.6.0-openai \
        --reason "Release v2.6.0 OpenAI reference profile"

    # Dry-run (preview only):
    python scripts/eval/promote_profile.py \
        --source data/profiles/issue-477/issue477-staged-gpt4o.yaml \
        --promoted-id v2.6.0-openai-gpt4o \
        --reason "gpt-4o staged pipeline reference" \
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCES_DIR = _ROOT / "data" / "profiles" / "references"

_REQUIRED_YAML_KEYS: List[str] = [
    "release",
    "date",
    "dataset_id",
    "stages",
    "totals",
]


def _load_profile(path: Path) -> Dict[str, Any]:
    """Load and validate a frozen profile YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    missing = [k for k in _REQUIRED_YAML_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Profile {path} missing required keys: {missing}")
    return raw


def _stage_truth_path(profile_path: Path) -> Path:
    """Companion stage_truth.json path for a profile YAML."""
    return profile_path.parent / f"{profile_path.stem}.stage_truth.json"


def _monitor_log_path(profile_path: Path) -> Path:
    """Optional RFC-065 monitor archive next to the profile YAML."""
    return profile_path.parent / f"{profile_path.stem}.monitor.log"


def promote(
    *,
    source: Path,
    promoted_id: str,
    reason: str,
    no_stage_truth_required: bool = False,
    dry_run: bool = False,
) -> Path:
    """Promote a working profile to a reference.

    Args:
        source: Path to the working profile YAML.
        promoted_id: Reference name (becomes filename stem).
        reason: Human-readable promotion rationale.
        no_stage_truth_required: Allow missing stage_truth.json.
        dry_run: Preview actions without writing.

    Returns:
        Path to the promoted reference YAML.

    Raises:
        FileNotFoundError: Source does not exist.
        ValueError: Validation failure (missing keys, bad name, etc.).
        FileExistsError: Reference already exists.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Source profile not found: {source}")

    if "wip" in promoted_id.lower():
        raise ValueError(f"promoted_id must not contain 'wip': {promoted_id!r}")

    dest = _REFERENCES_DIR / f"{promoted_id}.yaml"
    if dest.exists():
        raise FileExistsError(
            f"Reference already exists: {dest}  " "(references are immutable; choose a new ID)"
        )

    st_src = _stage_truth_path(source)
    mon_src = _monitor_log_path(source)
    has_stage_truth = st_src.is_file()
    has_monitor_log = mon_src.is_file()
    if not has_stage_truth and not no_stage_truth_required:
        raise FileNotFoundError(
            f"stage_truth.json not found at {st_src}  "
            "(use --no-stage-truth-required to override)"
        )

    doc = _load_profile(source)

    doc["promoted"] = {
        "promoted_id": promoted_id,
        "promoted_from": str(source),
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
    }

    if dry_run:
        logger.info("[dry-run] Would write: %s", dest)
        if has_stage_truth:
            st_dest = _REFERENCES_DIR / f"{promoted_id}.stage_truth.json"
            logger.info("[dry-run] Would copy:  %s -> %s", st_src, st_dest)
        if has_monitor_log:
            mon_dest = _REFERENCES_DIR / f"{promoted_id}.monitor.log"
            logger.info("[dry-run] Would copy:  %s -> %s", mon_src, mon_dest)
        logger.info(
            "[dry-run] promoted block:\n%s",
            yaml.safe_dump(
                {"promoted": doc["promoted"]},
                default_flow_style=False,
            ).rstrip(),
        )
        return dest

    _REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    dest.write_text(
        yaml.safe_dump(
            doc,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote reference profile: %s", dest)

    if has_stage_truth:
        st_dest = _REFERENCES_DIR / f"{promoted_id}.stage_truth.json"
        shutil.copy2(str(st_src), str(st_dest))
        logger.info("Copied stage truth:     %s", st_dest)

    if has_monitor_log:
        mon_dest = _REFERENCES_DIR / f"{promoted_id}.monitor.log"
        shutil.copy2(str(mon_src), str(mon_dest))
        logger.info("Copied monitor log:     %s", mon_dest)

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a frozen profile to a reference.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the working profile YAML",
    )
    parser.add_argument(
        "--promoted-id",
        required=True,
        help=(
            "Reference name (becomes filename stem, e.g. v2.6.0-openai). " "Must not contain 'wip'."
        ),
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Human-readable reason for promotion",
    )
    parser.add_argument(
        "--no-stage-truth-required",
        action="store_true",
        help="Allow promotion without a companion stage_truth.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without writing files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        dest = promote(
            source=args.source,
            promoted_id=args.promoted_id,
            reason=args.reason,
            no_stage_truth_required=args.no_stage_truth_required,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if not args.dry_run:
        logger.info("Done. Reference: %s", dest.name)


if __name__ == "__main__":
    main()
