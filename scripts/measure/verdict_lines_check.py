#!/usr/bin/env python3
"""Check every reviewer-verdict row's deciding line against the corpus. Read-only.

A verdicts file (``src/podcast_scraper/upgrade/verdicts/<migration>.tsv``) row with a basis names
its deciding segment (``segment_idx``) and the sha256 of that segment's text (``line_sha256``), so a
public file proves its evidence without carrying transcript text. Nothing checked the hash against
the served segment. This does: for each such row it resolves the episode's ``segments.json`` the
way the migrations do (the metadata's ``content.transcript_file_path``, relative to the run) and
compares ``line_sha256(segments[segment_idx].text)``.

    python scripts/measure/verdict_lines_check.py --corpus <root> [--migration <id>] [--json <out>]

Prints, per verdicts file: rows checked, matching, mismatching (the line's text is not the one
reviewed), missing (no metadata, no segments, or no such index), and up to 10 example rows. Exits
1 when any row mismatches or is missing. A corpus that has moved on since the review (a later
migration that rewrote segment text) mismatches by design: the output names the rows to re-read.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import (  # noqa: E402
    _load,
    _run_relative,
)
from podcast_scraper.upgrade.reviewer_verdicts import line_sha256, VERDICTS_DIR  # noqa: E402


def _rows(path: Path) -> List[Dict[str, str]]:
    body = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    )
    return list(csv.DictReader(io.StringIO(body), delimiter="\t"))


def _segment_texts(corpus: Path, meta_rel: str) -> List[str] | None:
    meta_path = corpus / meta_rel
    meta = _load(meta_path)
    if not isinstance(meta, dict):
        return None
    rel = _run_relative(
        (meta.get("content") or {}).get("transcript_file_path"), meta_path.parent.parent
    )
    if not rel.endswith(".txt"):
        return None
    payload = _load(meta_path.parent.parent / rel.replace(".txt", ".segments.json"))
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    if not isinstance(rows, list):
        return None
    return [str(r.get("text") or "") if isinstance(r, dict) else "" for r in rows]


def check(corpus: Path, verdicts: Path) -> Dict[str, Any]:
    """Counts and examples for one verdicts file against ``corpus``."""
    out: Dict[str, Any] = {
        "file": verdicts.name,
        "checked": 0,
        "match": 0,
        "mismatch": [],
        "missing": [],
    }
    cache: Dict[str, List[str] | None] = {}
    for row in _rows(verdicts):
        idx, want = (row.get("segment_idx") or "").strip(), (row.get("line_sha256") or "").strip()
        if not idx.isdigit() or len(want) != 64:
            continue
        out["checked"] += 1
        key = {"meta": row.get("meta", ""), "voice": row.get("voice", ""), "segment_idx": int(idx)}
        meta_rel = row.get("meta") or ""
        if meta_rel not in cache:
            cache[meta_rel] = _segment_texts(corpus, meta_rel)
        texts = cache[meta_rel]
        if texts is None or int(idx) >= len(texts):
            out["missing"].append(key)
        elif line_sha256(texts[int(idx)]) == want:
            out["match"] += 1
        else:
            out["mismatch"].append(key)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--verdicts-dir", type=Path, default=VERDICTS_DIR)
    ap.add_argument("--migration", action="append", default=[])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    files = sorted(args.verdicts_dir.glob("*.tsv"))
    if args.migration:
        files = [f for f in files if f.stem in set(args.migration)]
    reports = [check(args.corpus, f) for f in files]
    bad = 0
    for r in reports:
        bad += len(r["mismatch"]) + len(r["missing"])
        print(
            f"{r['file']}: checked {r['checked']}  match {r['match']}  "
            f"mismatch {len(r['mismatch'])}  missing {len(r['missing'])}"
        )
        for kind in ("mismatch", "missing"):
            for key in r[kind][:10]:
                print(f"  {kind}: {key['meta']}  {key['voice']}  segment {key['segment_idx']}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"VERDICT LINES: {'FAIL' if bad else 'PASS'} ({bad} rows mismatch or missing)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
