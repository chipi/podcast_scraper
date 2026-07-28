#!/usr/bin/env python3
"""ADR-133/#1220 — upgrade ``tests/fixtures/viewer-validation-corpus`` GI artifacts to v3.1.

Every ``.gi.json`` in the validation corpus carries its unresolved diarization
speakers as ``Person`` nodes with ``person:speaker-NN`` ids (see
``build_synthetic_validation_corpus.build_gi``). v2.4 introduces the ``Voice``
node type for exactly those unresolved speakers (kept out of the Person/Org
identity graph), so the golden corpus needs the same retype the write-side now
emits and the migration brings old corpora forward with.

This script walks every ``*.gi.json`` under the validation corpus and applies
``migrate_gi_document_v3_1`` in place:

1. Retypes ``person:speaker-NN`` ``Person`` nodes to ``Voice`` (ids + SPOKEN_BY
   edges are untouched).
2. Bumps ``schema_version`` ``"3.0"`` -> ``"3.1"``.

KG artifacts are NOT touched: the synthetic KG models no per-speaker nodes
(only Entity/Topic), so there is nothing to retype there.

Run from the repo root:

    .venv/bin/python scripts/dev/upgrade_viewer_validation_corpus_to_v3_1.py

Idempotent: ``migrate_gi_document_v3_1`` is a no-op on an already-3.1 artifact
whose voices are already ``Voice``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from podcast_scraper.migrations.gil_kg_identity_migrations import (  # noqa: E402
    migrate_gi_document_v3_1,
)

_FIXTURES_VERSION = (
    (_REPO_ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()
)
_CORPUS_DIR = _REPO_ROOT / "tests" / "fixtures" / "viewer-validation-corpus" / _FIXTURES_VERSION


def _upgrade_file(path: Path) -> bool:
    """Apply the 3.1 migration in place. Returns True if the file changed."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    migrated = migrate_gi_document_v3_1(data)
    out = json.dumps(migrated, ensure_ascii=False, indent=2) + "\n"
    if out == raw:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_CORPUS_DIR,
        help="root of the viewer-validation-corpus version to upgrade",
    )
    args = parser.parse_args()

    corpus_dir: Path = args.corpus_dir
    files = sorted(corpus_dir.rglob("*.gi.json"))
    if not files:
        print(f"No .gi.json files under {corpus_dir}", file=sys.stderr)
        return 1

    changed = 0
    for path in files:
        if _upgrade_file(path):
            changed += 1
            print(f"  upgraded {path.relative_to(_REPO_ROOT)}")
    print(f"Done. {changed}/{len(files)} .gi.json files changed (rest already 3.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
