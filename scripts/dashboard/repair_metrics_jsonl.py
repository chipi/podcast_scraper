#!/usr/bin/env python3
"""
Rewrite metrics history JSONL as one compact JSON object per line.

Legacy CI appended pretty-printed JSON with `echo "$(cat latest.json)"`, which broke
JSONL. This script recovers objects and rewrites the file.

Usage:
  python scripts/dashboard/repair_metrics_jsonl.py metrics/history-ci.jsonl --in-place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DASHBOARD_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_SCRIPTS_DIR))

from metrics_jsonl import dump_compact_line, load_metrics_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair metrics JSONL history files")
    parser.add_argument("path", type=Path, help="Path to history-ci.jsonl or history-nightly.jsonl")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the file (default: print to stdout)",
    )
    args = parser.parse_args()
    path = args.path
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(1)

    records = load_metrics_history(path)
    lines = [dump_compact_line(r) for r in records]
    body = "\n".join(lines)
    if body:
        body += "\n"

    if args.in_place:
        path.write_text(body, encoding="utf-8")
        print(f"Wrote {len(records)} record(s) to {path}")
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
