#!/usr/bin/env python3
"""Merge artifacts/ci-metrics-runs/run-*/latest-ci.json into one history-ci.jsonl.

Each downloaded CI bundle carries its own short history file. For local preview,
we need one JSONL with one compact object per workflow run, ordered by GitHub
run id (ascending = chronological), so dashboard trend charts show multiple points.

See build_local_metrics_preview.sh (calls this when run-* bundles exist).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensures metrics_jsonl resolves when loaded via importlib (tests) or unusual cwd.
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
        latest = p / "latest-ci.json"
        if latest.is_file():
            out.append((rid, p))
    out.sort(key=lambda x: x[0])
    return out


def load_snapshots_ordered(runs_dir: Path) -> List[Dict[str, Any]]:
    """Load ``latest-ci.json`` from each bundle, ordered oldest run first."""
    records: List[Dict[str, Any]] = []
    for rid, p in list_run_bundles(runs_dir):
        path = p / "latest-ci.json"
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skip run %s (%s): %s", rid, path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("Skip run %s: latest-ci.json is not a JSON object", rid)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge run-*/latest-ci.json files into history-ci.jsonl for local dashboard."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/ci-metrics-runs"),
        help="Directory containing run-<id>/ subfolders (default: artifacts/ci-metrics-runs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to write merged history-ci.jsonl",
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
    write_history_jsonl(records, args.output)

    if args.verbose:
        logger.info(
            "Wrote %d history line(s) from %d bundle(s) under %s -> %s",
            len(records),
            len(bundles),
            args.runs_dir,
            args.output,
        )
    elif not records and bundles:
        logger.warning("No valid latest-ci.json records written (check JSON errors above)")


if __name__ == "__main__":
    main()
