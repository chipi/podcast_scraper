#!/usr/bin/env python3
"""RFC-097 chunk 5/6: backfill or recompute ``Insight.position_hint`` on existing GI
artifacts using the 4-step waterfall (RSS duration → segments end → max Quote ts → skip).

Pure arithmetic — no LLM calls. Works on every gi.json with Quote.timestamp_start_ms
populated. Optionally reads sibling ``*.segments.json`` for the step-2 fallback.

Idempotent: existing position_hint values are overwritten with the recomputed value.

Examples:

  # Recompute one file in-place
  python scripts/compute_gi_position_hints.py path/to/episode.gi.json

  # Recompute one file, reading sibling segments
  python scripts/compute_gi_position_hints.py path/to/episode.gi.json \\
      --segments path/to/episode.segments.json

  # Recompute a whole corpus (recursive); per-file sibling segments auto-detected
  python scripts/compute_gi_position_hints.py --corpus /path/to/corpus

Wraps ``podcast_scraper.migrations.gil_kg_identity_migrations.compute_position_hints_for_document``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional


def _load_segments(path: Optional[Path]) -> Optional[List[Any]]:
    if path is None or not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("segments"), list):
        return raw["segments"]
    return None


def _sibling_segments_path(gi_path: Path) -> Path:
    """Conventional sibling: ``episode.gi.json`` → ``episode.segments.json``."""
    name = gi_path.name
    if name.endswith(".gi.json"):
        return gi_path.with_name(name[: -len(".gi.json")] + ".segments.json")
    return gi_path.with_suffix(".segments.json")


def _process_file(in_path: Path, out_path: Path, segments_path: Optional[Path]) -> None:
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    segments = _load_segments(segments_path)
    from podcast_scraper.migrations.gil_kg_identity_migrations import (
        compute_position_hints_for_document,
    )

    out = compute_position_hints_for_document(raw, transcript_segments=segments)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Path to a single .gi.json file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here (default: overwrite input)",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="Path to sibling .segments.json (step 2 fallback). Auto-detected for --corpus.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Recursively recompute every .gi.json under this directory (in-place).",
    )
    args = parser.parse_args()

    if args.corpus is not None:
        root: Path = args.corpus
        if not root.is_dir():
            print(f"--corpus directory not found: {root}", file=sys.stderr)
            return 2
        count = 0
        for p in root.rglob("*.gi.json"):
            seg = _sibling_segments_path(p)
            _process_file(p, p, seg if seg.is_file() else None)
            count += 1
        print(f"Recomputed position_hint on {count} .gi.json files under {root}")
        return 0

    if args.input is None:
        parser.print_help(sys.stderr)
        return 2
    inp: Path = args.input
    out_path: Path = args.output or inp
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2
    _process_file(inp, out_path, args.segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
