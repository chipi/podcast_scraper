#!/usr/bin/env python3
"""m0025 on a COPY of the episodes it touches: does it publish what the fixed roster publishes?

Reads the live corpus, writes only under ``--copy``. For every episode m0025 would write, refuse
or leave, plus every episode the roster replay changed (``--voice-changes``, the JSONL of
``roster_replay.py``):

1. copies the episode's files, applies m0025 to the copy;
2. PARITY, per voice: where the stored label equals what the OLD roster replays (the record is a
   faithful product of today's code), m0025's label must equal what the NEW roster replays.
   Where the stored record was written by older code, m0025's changes are printed for reading;
3. INTEGRITY on the copy: every GI quote slices to its text, every ad-free segment and turn row
   to its line, ``verify`` passes, a second apply writes nothing, ``undo`` restores every file;
4. the live files of those episodes are byte-identical before and after.

    python scripts/measure/m0025_parity.py --corpus /app/output --copy /tmp/m25/copy \\
        --old roster=/tmp/old_roster.py --new roster=src/.../roster.py \\
        --voice-changes voice.jsonl

Exit 1 on any parity disagreement, integrity failure or live change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roster_replay as rr  # noqa: E402


def _rows(path: Path) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("segments", [])
    return [r for r in rows if isinstance(r, dict)]


def _labels(seg_path: Path) -> Dict[str, str]:
    out: Dict[str, set] = {}
    for r in _rows(seg_path):
        if r.get("speaker"):
            out.setdefault(str(r["speaker"]), set()).add(
                str(r.get("speaker_label") or r["speaker"])
            )
    return {v: "|".join(sorted(s)) for v, s in out.items()}


def _transcript(meta: Path) -> Path:
    rel = json.loads(meta.read_text(encoding="utf-8"))["content"]["transcript_file_path"]
    return Path(rel) if os.path.isabs(rel) else meta.parent.parent / rel


def _episode_files(meta: Path) -> List[Path]:
    t = _transcript(meta)
    stem = meta.name[: -len(".metadata.json")]
    return [
        p
        for p in list(meta.parent.glob(stem + ".*")) + list(t.parent.glob(t.name[:-4] + ".*"))
        if p.is_file()
    ]


def _sha(paths: List[Path]) -> Dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _integrity(copy: Path) -> List[str]:
    bad: List[str] = []
    for gi in copy.rglob("*.gi.json"):
        tdir = gi.parent.parent / "transcripts"
        for node in json.loads(gi.read_text(encoding="utf-8")).get("nodes") or []:
            p = node.get("properties") or {}
            if node.get("type") != "Quote" or not isinstance(p.get("char_start"), int):
                continue
            f = tdir / Path(str(p.get("transcript_ref") or "")).name
            if f.is_file() and f.read_text(encoding="utf-8")[p["char_start"] : p["char_end"]] != (
                p.get("text")
            ):
                bad.append(f"quote {node.get('id')} in {gi.name}")
    for seg in copy.rglob("*.adfree.segments.json"):
        t = Path(str(seg).replace(".adfree.segments.json", ".adfree.txt"))
        if not t.is_file():
            continue
        text = t.read_text(encoding="utf-8")
        for r in _rows(seg):
            if isinstance(r.get("char_start"), int) and text[r["char_start"] : r["char_end"]] != (
                r.get("text")
            ):
                bad.append(f"segment in {seg.name}")
    for turns in copy.rglob("*.turns.json"):
        doc = json.loads(turns.read_text(encoding="utf-8"))
        text = (turns.parent / Path(doc["source"]["transcript_ref"]).name).read_text(
            encoding="utf-8"
        )
        seg = turns.parent / Path(doc["source"]["segments_ref"]).name
        if doc["source"].get("segments_sha256") != hashlib.sha256(seg.read_bytes()).hexdigest():
            bad.append(f"stale segments hash in {turns.name}")
        for row in doc.get("turns") or []:
            line = text[: row["char_start"]].rsplit("\n", 1)[-1]
            if line != f"{row.get('speaker_label') or row.get('speaker')}: ":
                bad.append(f"turn {row.get('turn_id')} in {turns.name}")
    return bad


def main(argv: Optional[List[str]] = None) -> int:
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts
    from podcast_scraper.upgrade.migration import MigrationContext
    from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import (
        dry_run_report,
        OnePersonOneEntryMigration,
        undo,
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--copy", type=Path, required=True)
    ap.add_argument("--old", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--new", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--voice-changes", type=Path, help="roster_replay.py JSONL output")
    ap.add_argument("--known-bad-quotes", type=int, default=0, help="quotes misaligned BEFORE")
    args = ap.parse_args(argv)
    src, dst = args.corpus, args.copy

    report = list(dry_run_report(src))
    targets = {r["meta"] for r in report}
    titles = set()
    if args.voice_changes:
        for line in args.voice_changes.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if "episode" in d:
                titles.add(d["episode"])
    eps: Dict[str, Path] = {}
    for meta in select_served_artifacts(src, ".metadata.json")[0]:
        rel = str(meta.relative_to(src))
        title = (json.loads(meta.read_text(encoding="utf-8")).get("episode") or {}).get("title")
        if rel in targets or title in titles:
            eps[rel] = meta
    live = [p for meta in eps.values() for p in _episode_files(meta)]
    live_before = _sha(live)
    if dst.exists():
        shutil.rmtree(dst)
    for p in live:
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    copy_before = _sha([p for p in dst.rglob("*") if p.is_file()])

    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=dst))
    print(f"apply: {result.message[:300]}")
    print(f"refused: {result.details['refused']}  left: {result.details['left']}")

    rr.index_siblings(src)
    old = rr.load_variant(rr._variant_arg(args.old))
    new = rr.load_variant(rr._variant_arg(args.new))
    agree = disagree = older = older_differ = 0
    for meta in sorted(eps.values()):
        ep = rr.load_episode(meta)
        seg_live = Path(str(_transcript(meta)).replace(".txt", ".segments.json"))
        stored, after = _labels(seg_live), _labels(dst / seg_live.relative_to(src))
        o = {
            v: (r.name if r.named else v) for v, r in rr.replay(old["roster"], ep).by_voice.items()
        }
        n = {
            v: (r.name if r.named else v) for v, r in rr.replay(new["roster"], ep).by_voice.items()
        }
        for v in sorted(set(stored) | set(n)):
            if stored.get(v) == o.get(v):
                if after.get(v) == n.get(v):
                    agree += 1
                else:
                    disagree += 1
                    print(
                        f"  DISAGREE {meta.name[:45]} {v}: stored/old={stored.get(v)!r} "
                        f"m0025={after.get(v)!r} new={n.get(v)!r}"
                    )
            else:
                older += 1
                if after.get(v) != stored.get(v) or after.get(v) != n.get(v):
                    mark = "" if after.get(v) == n.get(v) else "  != new"
                    older_differ += after.get(v) != n.get(v)
                    print(
                        f"  older record {meta.name[:40]} {v}: stored={stored.get(v)!r} "
                        f"old={o.get(v)!r} m0025={after.get(v)!r} new={n.get(v)!r}{mark}"
                    )
    bad = _integrity(dst)
    ok_verify, verify_msg = m.verify(MigrationContext(corpus_root=dst))
    second = m.apply(MigrationContext(corpus_root=dst)).details["files_written"]
    restored, refused = undo(dst)
    copy_after = _sha(
        [p for p in dst.rglob("*") if p.is_file() and p.name in {Path(k).name for k in copy_before}]
    )
    undo_diff = [k for k in copy_before if copy_before[k] != copy_after.get(k)]
    live_diff = [k for k, h in _sha(live).items() if live_before.get(k) != h]
    print(
        json.dumps(
            {
                "episodes": len(eps),
                "voices_faithful": agree + disagree,
                "agree": agree,
                "disagree": disagree,
                "voices_older_record": older,
                "voices_older_record_m0025_ne_new": older_differ,
                "integrity_failures": len(bad),
                "integrity_sample": bad[:5],
                "verify": [ok_verify, verify_msg],
                "second_apply_files": second,
                "undo_restored": restored,
                "undo_refused": len(refused),
                "undo_diff": len(undo_diff),
                "live_changed": len(live_diff),
            }
        )
    )
    failed = (
        disagree
        or len(bad) > args.known_bad_quotes
        or not ok_verify
        or second
        or refused
        or undo_diff
        or live_diff
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
