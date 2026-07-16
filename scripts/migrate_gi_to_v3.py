#!/usr/bin/env python3
"""RFC-097 v3.0 GI migration: rewrite legacy ``MENTIONS`` → typed ``MENTIONS_PERSON`` /
``MENTIONS_ORG`` based on each edge target's canonical id prefix (``person:`` / ``org:``).
Also normalises ``Insight.insight_type`` from legacy vocab (``fact``/``opinion``) to the
v3 schema vocab (``claim``/``observation``); out-of-vocab types fall to ``unknown``.

Bumps ``schema_version`` to ``3.0``. Idempotent. Back up your corpus before running.

Examples:

  # Migrate one file in-place
  python scripts/migrate_gi_to_v3.py path/to/episode.gi.json

  # Migrate a whole corpus (recursive .gi.json walk)
  python scripts/migrate_gi_to_v3.py --corpus /path/to/corpus

Wraps ``podcast_scraper.migrations.gil_kg_identity_migrations.migrate_gi_document_v3``.

Status (see #1176): **superseded by** the framework migration
``upgrade/migrations/m0003_gi_v3_typed_mentions.py`` — running
``make upgrade-corpus`` walks the corpus, applies this transform per-file, and
records the step in the ledger. This standalone script stays as an escape
hatch for one-off surgery on a single ``.gi.json`` outside the framework
(e.g. rehearsing a migration on a copy before running framework-wide).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _migrate_file(in_path: Path, out_path: Path) -> None:
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    from podcast_scraper.migrations.gil_kg_identity_migrations import migrate_gi_document_v3

    migrated = migrate_gi_document_v3(raw)
    out_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        "--corpus",
        type=Path,
        default=None,
        help="Recursively migrate every .gi.json under this directory (in-place).",
    )
    args = parser.parse_args()

    if args.corpus is not None:
        root: Path = args.corpus
        if not root.is_dir():
            print(f"--corpus directory not found: {root}", file=sys.stderr)
            return 2
        count = 0
        for p in root.rglob("*.gi.json"):
            _migrate_file(p, p)
            count += 1
        print(f"Migrated {count} .gi.json files under {root}")
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
