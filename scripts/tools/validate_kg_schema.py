#!/usr/bin/env python3
"""Validate kg.json files against the KG artifact schema (strict mode).

Use from project root:
  python scripts/tools/validate_kg_schema.py path/to/dir
  python scripts/tools/validate_kg_schema.py file1.kg.json file2.kg.json

Exits 0 if all files pass; 1 if any fail (errors printed to stderr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from podcast_scraper.kg.io import read_artifact
    from podcast_scraper.kg.schema import validate_artifact
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.kg.io import read_artifact
    from podcast_scraper.kg.schema import validate_artifact


def collect_kg_json_paths(paths: list[Path]) -> list[Path]:
    """Return all .kg.json file paths from given files and directories."""
    result: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"Error: path does not exist: {p}", file=sys.stderr)
            sys.exit(1)
        if p.is_file():
            if not p.name.endswith(".kg.json"):
                print(f"Error: not a .kg.json file: {p}", file=sys.stderr)
                sys.exit(1)
            result.append(p)
            continue
        for child in p.rglob("*.kg.json"):
            result.append(child)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate kg.json files against the KG artifact schema (strict)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories; directories are scanned for **/*.kg.json",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures",
    )
    args = parser.parse_args()
    files_to_validate = sorted(set(collect_kg_json_paths(args.paths)))
    if not files_to_validate:
        if not args.quiet:
            print("No .kg.json files found under given paths.", file=sys.stderr)
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
