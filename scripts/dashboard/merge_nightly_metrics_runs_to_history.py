#!/usr/bin/env python3
"""Merge artifacts/nightly-metrics-runs/run-*/latest-nightly.json into history + latest.

Each downloaded nightly bundle includes a snapshot for that workflow run. For local preview,
merge into one JSONL (chronological by GitHub run id) and copy ``latest-nightly.json`` from the
newest run — same pattern as ``merge_ci_metrics_runs_to_history.py`` for CI.

See ``fetch_nightly_metrics_artifacts.sh`` and ``fetch_nightly_metrics.sh`` (``N`` >= 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DASH = Path(__file__).resolve().parent
_sdash = str(_DASH)
if _sdash not in sys.path:
    sys.path.insert(0, _sdash)

from metrics_jsonl import dump_compact_line  # noqa: E402

logger = logging.getLogger(__name__)

RUN_DIR_RE = re.compile(r"^run-(\d+)$")


def parse_run_id(dirname: str) -> Optional[int]:
    """Return numeric workflow database id from directory name ``run-<id>``, else None."""
    m = RUN_DIR_RE.match(dirname)
    return int(m.group(1)) if m else None


def list_run_bundles(runs_dir: Path) -> List[Tuple[int, Path]]:
    """Pairs ``(run_id, bundle_dir)`` sorted by ``run_id`` ascending."""
    if not runs_dir.is_dir():
        return []
    out: List[Tuple[int, Path]] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        rid = parse_run_id(p.name)
        if rid is None:
            continue
        latest = p / "latest-nightly.json"
        if latest.is_file():
            out.append((rid, p))
    out.sort(key=lambda x: x[0])
    return out


def load_snapshots_ordered(runs_dir: Path) -> List[Dict[str, Any]]:
    """Load ``latest-nightly.json`` from each bundle, ordered oldest run first."""
    records: List[Dict[str, Any]] = []
    for rid, p in list_run_bundles(runs_dir):
        path = p / "latest-nightly.json"
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skip run %s (%s): %s", rid, path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("Skip run %s: latest-nightly.json is not a JSON object", rid)
            continue
        records.append(data)
    return records


def write_history_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write one compact JSON object per line (UTF-8)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    lines = [dump_compact_line(r) for r in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_latest_from_newest_run(runs_dir: Path, output_path: Path) -> bool:
    """Copy ``latest-nightly.json`` from the highest ``run-<id>`` bundle. Return True if copied."""
    bundles = list_run_bundles(runs_dir)
    if not bundles:
        return False
    _rid, newest = bundles[-1]
    src = newest / "latest-nightly.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge run-*/latest-nightly.json into history-nightly.jsonl for local dashboard."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/nightly-metrics-runs"),
        help="Directory containing run-<id>/ subfolders (default: artifacts/nightly-metrics-runs)",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        required=True,
        help="Path to write merged history-nightly.jsonl",
    )
    parser.add_argument(
        "--latest-output",
        type=Path,
        required=True,
        help="Path to write latest-nightly.json (from newest run-<id>)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log info about merged runs",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    bundles = list_run_bundles(args.runs_dir)
    records = load_snapshots_ordered(args.runs_dir)
    write_history_jsonl(records, args.history_output)
    copied = copy_latest_from_newest_run(args.runs_dir, args.latest_output)

    if args.verbose:
        logger.info(
            "Wrote %d history line(s) from %d bundle(s) under %s -> %s; latest copied=%s",
            len(records),
            len(bundles),
            args.runs_dir,
            args.history_output,
            copied,
        )
    elif not records and bundles:
        logger.warning("No valid latest-nightly.json records written (check JSON errors above)")
    elif not bundles:
        logger.warning("No run-* bundles with latest-nightly.json under %s", args.runs_dir)


if __name__ == "__main__":
    main()
