#!/usr/bin/env python3
"""RFC-097 v2.0 KG migration: legacy ``Entity(kind=...)`` → typed ``Person`` / ``Organization``.

Idempotent. Back up your corpus before running. Examples:

  # Migrate one file in-place
  python scripts/migrate_kg_entity_to_person_org.py path/to/episode.kg.json

  # Migrate a whole corpus (recursive .kg.json walk)
  python scripts/migrate_kg_entity_to_person_org.py --corpus /path/to/corpus

Wraps ``podcast_scraper.migrations.gil_kg_identity_migrations.migrate_kg_document_v2``.

Status (see #1176): historical one-shot for the RFC-097 v2.0 KG shape
(typed ``Person`` / ``Organization`` nodes replacing the legacy
``Entity(kind=...)`` form). The read-time shim ``migrate_kg_document_v2`` in
``src/podcast_scraper/migrations/gil_kg_identity_migrations.py`` handles the
same rewrite transparently at read time, so serving a legacy corpus does not
require running this script. Prefer ``make upgrade-corpus`` (framework path)
when you own the corpus and can rewrite files — this script stays as an
escape hatch for one-off surgery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _migrate_file(in_path: Path, out_path: Path) -> None:
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    from podcast_scraper.migrations.gil_kg_identity_migrations import migrate_kg_document_v2

    migrated = migrate_kg_document_v2(raw)
    out_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Path to a single .kg.json file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here (default: overwrite input)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Recursively migrate every .kg.json under this directory (in-place).",
    )
    args = parser.parse_args()

    if args.corpus is not None:
        root: Path = args.corpus
        if not root.is_dir():
            print(f"--corpus directory not found: {root}", file=sys.stderr)
            return 2
        count = 0
        for p in root.rglob("*.kg.json"):
            _migrate_file(p, p)
            count += 1
        print(f"Migrated {count} .kg.json files under {root}")
        return 0

    if args.input is None:
        parser.print_help(sys.stderr)
        return 2
    inp: Path = args.input
    out_path: Path = args.output or inp
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2
    _migrate_file(inp, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
