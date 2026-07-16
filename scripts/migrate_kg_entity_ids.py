#!/usr/bin/env python3
"""Migrate legacy KG JSON: entity:person:/entity:organization: -> person:/org:; entity_kind -> kind.

Idempotent. Back up your corpus before running. Example:

  python scripts/migrate_kg_entity_ids.py input.kg.json -o output.kg.json

Status (see #1176): historical one-shot for the pre-RFC-097 KG id / kind
normalisation (v1.x → v1.2). The read-time shim ``migrate_kg_document`` in
``src/podcast_scraper/migrations/gil_kg_identity_migrations.py`` handles the
same rewrite transparently at read time; ``migrate_kg_document_v2`` chains
this into the RFC-097 v2 KG shape. No framework migration wraps this
script yet — add one (``upgrade/migrations/mNNNN_kg_v2_entity_typing.py``)
if you find corpora in the wild that need it in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to .kg.json")
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
    from podcast_scraper.migrations.gil_kg_identity_migrations import migrate_kg_document

    migrated = migrate_kg_document(raw)
    out_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
