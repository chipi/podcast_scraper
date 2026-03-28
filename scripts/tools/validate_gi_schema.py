#!/usr/bin/env python3
"""Validate gi.json files against the GIL artifact schema (strict mode).

Use from project root:
  python scripts/tools/validate_gi_schema.py path/to/dir
  python scripts/tools/validate_gi_schema.py file1.gi.json file2.gi.json

Exits 0 if all files pass; 1 if any fail (errors printed to stderr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root; add src to path if needed
try:
    from podcast_scraper.gi.io import read_artifact
    from podcast_scraper.gi.schema import validate_artifact
except ImportError:
    # Fallback when not installed
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.gi.io import read_artifact
    from podcast_scraper.gi.schema import validate_artifact


def collect_gi_json_paths(paths: list[Path]) -> list[Path]:
    """Return all .gi.json file paths from given files and directories."""
    result: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"Error: path does not exist: {p}", file=sys.stderr)
            sys.exit(1)
        if p.is_file():
            if not p.name.endswith(".gi.json"):
                print(f"Error: not a .gi.json file: {p}", file=sys.stderr)
                sys.exit(1)
            result.append(p)
            continue
        for child in p.rglob("*.gi.json"):
            result.append(child)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate gi.json files against the GIL artifact schema (strict)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories; directories are scanned for **/*.gi.json",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures",
    )
    args = parser.parse_args()
    files_to_validate = sorted(set(collect_gi_json_paths(args.paths)))
    if not files_to_validate:
        if not args.quiet:
            print("No .gi.json files found under given paths.", file=sys.stderr)
        return 0
    failed: list[tuple[Path, str]] = []
    for path in files_to_validate:
        try:
            data = read_artifact(path)
            validate_artifact(data, strict=True)
            if not args.quiet:
                print(f"OK {path}")
        except Exception as e:
            failed.append((path, str(e)))
            print(f"FAIL {path}: {e}", file=sys.stderr)
    if failed:
        print(
            f"\n{len(failed)} of {len(files_to_validate)} file(s) failed validation.",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        print(f"All {len(files_to_validate)} file(s) passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
