#!/usr/bin/env python3
"""One-shot reverse migration: GI v3.0 → v2.0 for a contaminated corpus.

Reverses the forward migration done by
``scripts/migrate_gi_to_v3.py`` for cases where a v2 corpus was
migrated and then operator-rolled-back. Lossy by design — see below.

What this reverts:
  * ``schema_version: "3.0"`` -> ``"2.0"``
  * ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` -> ``MENTIONS`` (untyped)
  * Strips top-level ``_retro_audit`` marker list

What this CANNOT reverse (and therefore stays normalized):
  * ``insight_type`` vocab — forward migration mapped legacy "fact"/
    "opinion" to "claim"/"observation". Original vocab is lost.
  * ``position_hint`` numeric — forward migration backfilled missing
    values with 0.5 and then `compute_position_hints` overwrote with
    real waterfall values. We cannot tell which were original.
  * Sweep-added MENTIONS_PERSON edges — once typed back to MENTIONS
    they look like organically-present v2 edges. The corpus will have
    N extra MENTIONS edges where N is the sweep's NER-added count.

For a truly pristine v2 corpus, the only path is `git`-grade
restoration from a snapshot or pipeline regeneration. This script is
for "good enough" rollback when v2-shape validation is the bar.

Usage:

    .venv/bin/python scripts/dev/revert_gi_v3_to_v2.py \\
        --corpus /path/to/contaminated/corpus

Idempotent: a v2 artifact passes through unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _revert_artifact(raw: dict) -> tuple[dict, bool]:
    """Return (reverted_artifact, was_modified)."""
    modified = False
    if raw.get("schema_version") == "3.0":
        raw["schema_version"] = "2.0"
        modified = True
    edges = raw.get("edges") or []
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("type") in ("MENTIONS_PERSON", "MENTIONS_ORG"):
            e["type"] = "MENTIONS"
            modified = True
    if "_retro_audit" in raw:
        del raw["_retro_audit"]
        modified = True
    return raw, modified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing",
    )
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"--corpus directory not found: {args.corpus}", file=sys.stderr)
        return 2

    total = 0
    modified_count = 0
    for gi_path in args.corpus.rglob("*.gi.json"):
        total += 1
        raw = json.loads(gi_path.read_text(encoding="utf-8"))
        reverted, was_modified = _revert_artifact(raw)
        if was_modified:
            modified_count += 1
            if not args.dry_run:
                gi_path.write_text(
                    json.dumps(reverted, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    # Also strip any sweep summary JSONs at the corpus root.
    summary_glob = list(args.corpus.glob("_retro_audit_*.json"))
    for s in summary_glob:
        if not args.dry_run:
            s.unlink()

    verb = "would revert" if args.dry_run else "reverted"
    print(f"{verb} {modified_count}/{total} .gi.json files under {args.corpus}")
    if summary_glob:
        verb = "would remove" if args.dry_run else "removed"
        print(f"{verb} {len(summary_glob)} retro-audit summary JSON(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
