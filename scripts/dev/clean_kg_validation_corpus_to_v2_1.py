#!/usr/bin/env python3
"""ADR-133 corpus hygiene — bring the validation-corpus KG artifacts up to v2.1.

The synthetic ``*.kg.json`` fixtures carried three pre-existing drifts, all
unrelated to any single feature and none produced by the real KG write-side:

1. ``schema_version`` was the integer-string ``"2"`` (the real write-side emits
   decimal ``"2.0"``/``"2.1"``; the strict gate rejects ``"2"``).
2. Legacy ``Entity`` nodes synthesised from topic phrases (``category:"Concept"``,
   e.g. ``entity:welcome-back-to``) — a node type the RFC-097 v2 KG pipeline no
   longer emits, redundant with the Topic nodes already present.
3. ``extraction`` was ``{"provider","model"}`` — missing all three strict-required
   fields (``model_version`` / ``extracted_at`` / ``transcript_ref``).

This script rewrites every ``*.kg.json`` in place to the clean v2.1 shape the
fixed ``build_synthetic_validation_corpus.build_kg`` now emits: Episode + Topic
nodes only, proper ``extraction``, ``schema_version`` ``"2.1"``. The
``extracted_at`` + ``transcript_ref`` are sourced from the sibling ``*.gi.json``
so the two artifacts stay consistent per episode.

Run from the repo root:

    .venv/bin/python scripts/dev/clean_kg_validation_corpus_to_v2_1.py

Idempotent: an already-clean 2.1 artifact is written back unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

_FIXTURES_VERSION = (
    (_REPO_ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()
)
_CORPUS_DIR = _REPO_ROOT / "tests" / "fixtures" / "viewer-validation-corpus" / _FIXTURES_VERSION

_MODEL_VERSION = "synthetic-validation-corpus-v1"


def _sibling_gi_fields(kg_path: Path) -> tuple[str | None, str | None]:
    """(transcript_ref, extracted_at) from the sibling ``*.gi.json``, if present."""
    gi_path = kg_path.with_name(kg_path.name.replace(".kg.json", ".gi.json"))
    if not gi_path.exists():
        return None, None
    gi = json.loads(gi_path.read_text(encoding="utf-8"))
    tref: str | None = None
    extracted_at: str | None = None
    for n in gi.get("nodes", []):
        if not isinstance(n, dict):
            continue
        props = n.get("properties") or {}
        if n.get("type") == "Quote" and tref is None:
            tref = props.get("transcript_ref")
        if n.get("type") == "Episode" and extracted_at is None:
            extracted_at = props.get("publish_date")
    return tref, extracted_at


def _clean(data: dict[str, Any], *, transcript_ref: str, extracted_at: str) -> dict[str, Any]:
    entity_ids = {
        n.get("id")
        for n in data.get("nodes", [])
        if isinstance(n, dict) and n.get("type") == "Entity"
    }
    data["nodes"] = [
        n for n in data.get("nodes", []) if not (isinstance(n, dict) and n.get("type") == "Entity")
    ]
    data["edges"] = [
        e
        for e in data.get("edges", [])
        if not (isinstance(e, dict) and (e.get("from") in entity_ids or e.get("to") in entity_ids))
    ]
    data["extraction"] = {
        "model_version": _MODEL_VERSION,
        "extracted_at": extracted_at,
        "transcript_ref": transcript_ref,
    }
    data["schema_version"] = "2.1"
    return data


def _clean_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    tref, extracted_at = _sibling_gi_fields(path)
    ep_label = path.name.replace(".kg.json", "")
    tref = tref or f"transcripts/{ep_label}.txt"
    extracted_at = extracted_at or "1970-01-01T00:00:00"
    cleaned = _clean(data, transcript_ref=tref, extracted_at=extracted_at)
    out = json.dumps(cleaned, indent=2, sort_keys=True) + "\n"
    if out == raw:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=_CORPUS_DIR)
    args = parser.parse_args()

    files = sorted(args.corpus_dir.rglob("*.kg.json"))
    if not files:
        print(f"No .kg.json files under {args.corpus_dir}", file=sys.stderr)
        return 1

    changed = 0
    for path in files:
        if _clean_file(path):
            changed += 1
            print(f"  cleaned {path.relative_to(_REPO_ROOT)}")
    print(f"Done. {changed}/{len(files)} .kg.json files changed (rest already 2.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
