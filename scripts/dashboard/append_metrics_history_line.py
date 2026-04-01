#!/usr/bin/env python3
"""
Emit one compact JSON line for appending to metrics history (JSONL).

Usage (from repo root):
  python scripts/dashboard/append_metrics_history_line.py metrics/latest-ci.json >> metrics/history-ci.jsonl

Do not use `echo "$(cat latest.json)"` — pretty-printed JSON spans multiple lines and breaks JSONL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: append_metrics_history_line.py <path-to-latest-metrics.json>", file=sys.stderr
        )
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("Metrics file must contain a JSON object", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(data, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
