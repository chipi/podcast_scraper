#!/usr/bin/env python3
"""Migrate legacy GIL JSON: Speaker nodes and speaker: ids -> Person / person: (RFC-072).

Idempotent. Back up your corpus before running. Example:

  python scripts/migrate_gi_speaker_to_person.py input.gi.json -o output.gi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to .gi.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here (default: overwrite input)",
    )
    args = parser.parse_args()
    inp: Path = args.input
    out_path: Path = args.output or inp
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2
    raw = json.loads(inp.read_text(encoding="utf-8"))
    from podcast_scraper.migrations.rfc072 import migrate_gil_document

    migrated = migrate_gil_document(raw)
    out_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
