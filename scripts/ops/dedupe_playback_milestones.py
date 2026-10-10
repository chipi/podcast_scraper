#!/usr/bin/env python3
"""Remove repeated playback milestones from every user's playback log. Dry-run unless ``--apply``.

Why: before ``6c6dcd104`` a listener who reopened an episode and resumed past 25% reported 25
again with a new timestamp, and the server kept it; the beta dashboard counts rows per milestone,
so each re-report read as another listener reaching it (seen on a phone 2026-10-10). The server
now keeps one row per user, episode and milestone; this pass applies the same rule to the rows
already written, once.

A record is a repeat when an EARLIER line of the same user's ``playback_events.jsonl`` has the same
``slug`` and ``milestone``: the first is kept, every other line is kept byte for byte. Each
rewritten file is backed up beside itself (``<file>.bak-dedupe-milestones``) and receipted in
``<appdata>/dedupe_playback_milestones.jsonl``. ``--undo`` restores each file from its backup and
keeps any lines the app appended after the apply. A file that changes while it is being rewritten
is skipped and reported, never overwritten.

    python scripts/ops/dedupe_playback_milestones.py --appdata /app/appdata            # dry-run
    python scripts/ops/dedupe_playback_milestones.py --appdata /app/appdata --apply
    python scripts/ops/dedupe_playback_milestones.py --appdata /app/appdata --undo

Exit 1 when an apply leaves a repeat behind or skips a file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG_NAME = "playback_events.jsonl"
BACKUP_SUFFIX = ".bak-dedupe-milestones"
RECEIPTS_FILE = "dedupe_playback_milestones.jsonl"


def dedupe_lines(text: str) -> Tuple[str, int]:
    """``(text without repeated milestones, lines dropped)``; the first of each is kept."""
    seen: set = set()
    kept: List[str] = []
    dropped = 0
    for line in text.splitlines(keepends=True):
        try:
            rec = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        if isinstance(rec, dict) and "milestone" in rec and "slug" in rec:
            key = (rec.get("slug"), rec.get("milestone"))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        kept.append(line)
    return "".join(kept), dropped


def _logs(appdata: Path) -> List[Path]:
    return sorted((appdata / "users").glob(f"*/{LOG_NAME}"))


def _replace_if_unchanged(path: Path, original: str, new: str) -> bool:
    """Write ``new`` over ``path`` only if it still reads ``original`` (the app may append)."""
    tmp = path.with_name(path.name + ".tmp-dedupe")
    tmp.write_text(new, encoding="utf-8")
    shutil.copymode(path, tmp)
    if path.read_text(encoding="utf-8") != original:
        tmp.unlink()
        return False
    os.replace(tmp, path)
    return True


def run(appdata: Path, apply: bool) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"users": 0, "files": 0, "dropped": 0, "skipped_changed": []}
    receipts: List[Dict[str, Any]] = []
    for path in _logs(appdata):
        stats["users"] += 1
        original = path.read_text(encoding="utf-8")
        new, dropped = dedupe_lines(original)
        if not dropped:
            continue
        stats["files"] += 1
        stats["dropped"] += dropped
        if not apply:
            continue
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(path, backup)
        if not _replace_if_unchanged(path, original, new):
            stats["skipped_changed"].append(str(path.relative_to(appdata)))
            continue
        receipts.append(
            {"path": str(path.relative_to(appdata)), "dropped": dropped, "written_len": len(new)}
        )
    if receipts:
        with (appdata / RECEIPTS_FILE).open("a", encoding="utf-8") as fh:
            for r in receipts:
                fh.write(json.dumps(r) + "\n")
    return stats


def undo(appdata: Path) -> Dict[str, Any]:
    """Restore every rewritten log from its backup, keeping lines appended since the apply."""
    restored, refused = 0, []
    try:
        lines = (appdata / RECEIPTS_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"restored": 0, "refused": ["no receipts: nothing to undo"]}
    for line in lines:
        r = json.loads(line)
        path = appdata / r["path"]
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            refused.append(f"{r['path']}: backup missing")
            continue
        current = path.read_text(encoding="utf-8")
        appended = current[int(r["written_len"]) :]
        new = backup.read_text(encoding="utf-8") + appended
        if _replace_if_unchanged(path, current, new):
            backup.unlink()
            restored += 1
        else:
            refused.append(f"{r['path']}: changed during undo, retry")
    (appdata / RECEIPTS_FILE).rename(appdata / (RECEIPTS_FILE + ".undone"))
    return {"restored": restored, "refused": refused}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--appdata", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    mode.add_argument("--undo", action="store_true", help="restore every log this pass wrote")
    args = ap.parse_args(argv)
    appdata = args.appdata.resolve()
    if args.undo:
        out = undo(appdata)
        print(json.dumps(out))
        return 1 if out["refused"] else 0
    stats = run(appdata, args.apply)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **stats}))
    if args.apply:
        left = run(appdata, False)["dropped"]
        print(json.dumps({"left_after_apply": left}))
        return 1 if left or stats["skipped_changed"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
