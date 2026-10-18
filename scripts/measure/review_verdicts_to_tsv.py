#!/usr/bin/env python3
"""Turn step-8 reviewers' read files into a migration's committed verdicts file.

    python scripts/measure/review_verdicts_to_tsv.py --migration 0031_named_voices_shown \\
        --corpus <clone> --reviewed-commit 97c551101 --reviewer "step-8 reviewer (fresh advisor)" \\
        --read <reviewer read file.tsv> [--read <another>] \\
        --out src/podcast_scraper/upgrade/verdicts/0031_named_voices_shown.tsv

A reviewer's read file has one row per insight read (columns ``episode, voice, name, verdict``; the
episode as its title). This maps every row to the list entry it judged, by the episode's own
metadata (episode title -> episode_id, checked against the list entry and the served per-file
path), and the verdict text through ``VERDICT_TEXT``, an explicit table: text not in it stops the
run. Rows for one voice must agree. Every mapping is printed, so the conversion is read, not
trusted. Nothing here names an entry: the decisions are the reviewers'.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.upgrade import reviewer_verdicts as rv  # noqa: E402
from podcast_scraper.upgrade.corpus_selection import select_served_artifacts  # noqa: E402
from podcast_scraper.upgrade.registry import get_migrations  # noqa: E402

#: Every verdict text a reviewer has written, and what it means. Unknown text stops the run.
VERDICT_TEXT = {
    "CORRECT": "CORRECT",
    "CORRECT (newly named)": "CORRECT",
    "CORRECT person (narrator Justin Norman, 'there's already 50 people in line ahead of me'); "
    "NOTE insight has no speaker field (quotes split with an unknown voice), shown now": "CORRECT",
    "CORRECT PERSON, MISSPELLED NAME": "CORRECT_MISSPELLED",
    "WRONG PERSON": "WRONG_PERSON",
}


def _entries(migration_id: str) -> Dict[Tuple[str, str], str]:
    """``{(episode_id, voice): listed name}`` the migration says a review judges (its own list)."""
    migration = next((m for m in get_migrations() if m.id == migration_id), None)
    if migration is None:
        raise SystemExit(f"not registered: {migration_id}")
    names = getattr(migration, "review_names", None)
    if names is None:
        raise SystemExit(f"{migration_id} reads no review verdicts (no review_names)")
    return dict(names())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--migration", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reviewed-commit", required=True)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--read", required=True, action="append", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    names = _entries(args.migration)
    by_title: Dict[str, List[Tuple[str, str]]] = {}
    for meta_path in select_served_artifacts(args.corpus, ".metadata.json")[0]:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        ep = m.get("episode") or {}
        by_title.setdefault(str(ep.get("title") or ""), []).append(
            (str(ep.get("episode_id") or ""), str(meta_path.relative_to(args.corpus)))
        )
    decided: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {}
    for read in args.read:
        for n, row in enumerate(csv.DictReader(read.open(encoding="utf-8"), delimiter="\t"), 2):
            text = row["verdict"].strip()
            if text not in VERDICT_TEXT:
                raise SystemExit(f"{read.name} row {n}: verdict text not in VERDICT_TEXT: {text!r}")
            verdict = VERDICT_TEXT[text]
            hits = [
                (eid, meta)
                for eid, meta in by_title.get(row["episode"].strip(), [])
                if (eid, row["voice"].strip()) in names
            ]
            if len(hits) != 1:
                raise SystemExit(
                    f"{read.name} row {n}: title {row['episode']!r} + {row['voice']} matches "
                    f"{len(hits)} list entries"
                )
            eid, meta = hits[0]
            key = (eid, row["voice"].strip())
            if row["name"].strip() != names[key]:
                raise SystemExit(
                    f"{read.name} row {n}: {key} read as {row['name']!r}, list {names[key]!r}"
                )
            evidence = (row.get("reason") or "").strip().replace("\t", " ")
            prev = decided.get(key)
            if prev and prev[2] != verdict:
                raise SystemExit(f"{read.name} row {n}: {key} judged {prev[2]} and {verdict}")
            if not prev or (verdict != "CORRECT" and not prev[3]):
                decided[key] = (meta, names[key], verdict, evidence)
            print(
                f"{read.name}:{n}  {text[:40]!r:44} -> {verdict:18} {key[0]} {key[1]} {names[key]}"
            )
    lines = [
        f"# migration: {args.migration}",
        f"# reviewed_commit: {args.reviewed_commit}",
        f"# list_sha256: {rv.list_sha256(names)}",
        f"# reviewer: {args.reviewer}",
        f"# date: {args.date}",
        "\t".join(rv.COLUMNS),
    ]
    for (eid, voice), (meta, name, verdict, evidence) in sorted(decided.items()):
        lines.append("\t".join((eid, meta, voice, name, verdict, evidence)))
    text = "\n".join(lines) + "\n"
    rv.parse(text, args.migration, names)  # the reader's own validation, before anything is written
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    unreviewed = sorted(set(names) - set(decided))
    print(
        f"wrote {args.out}: {len(decided)} entries; list entries with no verdict: {len(unreviewed)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
