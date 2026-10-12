#!/usr/bin/env python3
"""Compare two whole-corpus naming replays: absolute coverage, then every changed voice. Read-only.

Inputs are two ``roster_replay.py --trace-out`` files over the SAME snapshot (run the baseline and
the change side by side, so a reprocess landing in between cannot pass for a naming change):

    for k in 0 1 2 3; do
      python scripts/measure/roster_replay.py --corpus /app/output --repool --signatures \\
        --shard $k/4 --trace-out base.$k.jsonl --new roster=/tmp/base_roster.py &
      python scripts/measure/roster_replay.py --corpus /app/output --repool --signatures \\
        --shard $k/4 --trace-out new.$k.jsonl --new roster=src/.../roster.py &
    done; wait; cat base.?.jsonl > base.jsonl; cat new.?.jsonl > new.jsonl
    python scripts/measure/naming_replay_compare.py base.jsonl new.jsonl

Prints, for each side: episodes, episodes publishing NO named voice, named voices, episodes with a
named host; then how many episodes gained their first name and how many lost every name; then
every voice whose name or role changed, with its share of the talk, its own first words, and the
episode's host line that welcomes or introduces someone.

Rule of use (#2304): read every changed voice against its own words before trusting a total. Every
rejected attempt on that arc had good totals and wrong voices.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

Key = Tuple[Optional[str], Optional[str]]
_HOST_LINE = re.compile(r"\b(?:welcome|joined by|with me|my guest|our guest|today's guest)\b", re.I)


def load(path: Path) -> Dict[Key, Dict[str, Any]]:
    """Trace records keyed by (feed, episode); the trailing summary line is skipped."""
    out: Dict[Key, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "meta_path" in rec:
            out[(rec.get("feed"), rec.get("episode"))] = rec
    return out


def _named(role: Dict[str, Any]) -> Optional[str]:
    return role.get("name") if role.get("named") else None


def coverage(records: Dict[Key, Dict[str, Any]]) -> Dict[str, int]:
    rosters = [rec.get("roster") or {} for rec in records.values()]
    return {
        "episodes": len(rosters),
        "no_named_voice": sum(1 for r in rosters if not any(_named(v) for v in r.values())),
        "named_voices": sum(1 for r in rosters for v in r.values() if _named(v)),
        "named_host_episodes": sum(
            1 for r in rosters if any(_named(v) and v.get("role") == "host" for v in r.values())
        ),
    }


def episode_moves(a: Dict[Key, Dict[str, Any]], b: Dict[Key, Dict[str, Any]]) -> Dict[str, int]:
    gained = lost = 0
    for k in set(a) & set(b):
        was = any(_named(v) for v in (a[k].get("roster") or {}).values())
        now = any(_named(v) for v in (b[k].get("roster") or {}).values())
        gained += not was and now
        lost += was and not now
    return {"gained_first_name": gained, "lost_every_name": lost}


def changed_voices(
    a: Dict[Key, Dict[str, Any]], b: Dict[Key, Dict[str, Any]]
) -> Iterable[Tuple[Key, str, Dict[str, Any], Dict[str, Any]]]:
    """(episode, voice, old role, new role) for every voice whose name or role differs."""
    for k in sorted(set(a) & set(b), key=lambda x: (str(x[0]), str(x[1]))):
        ra, rb = a[k].get("roster") or {}, b[k].get("roster") or {}
        for v in sorted(set(ra) | set(rb)):
            x, y = ra.get(v) or {}, rb.get(v) or {}
            if (_named(x), x.get("role")) != (_named(y), y.get("role")):
                yield k, v, x, y


def voice_texts(meta_path: str) -> Dict[str, List[str]]:
    """Each voice's segment texts, from the episode's ``.segments.json`` next to its transcript."""
    meta = Path(meta_path)
    doc = json.loads(meta.read_text(encoding="utf-8"))
    tp = meta.parent.parent / doc["content"]["transcript_file_path"]
    segs = json.loads(Path(str(tp).replace(".txt", ".segments.json")).read_text(encoding="utf-8"))
    segs = segs if isinstance(segs, list) else segs.get("segments", [])
    out: Dict[str, List[str]] = {}
    for s in segs:
        out.setdefault(str(s.get("speaker")), []).append(str(s.get("text") or "").strip())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--no-evidence", action="store_true", help="skip reading segments")
    args = ap.parse_args(argv)
    a, b = load(args.base), load(args.new)
    for label, recs in (("base", a), ("new", b)):
        print(label, json.dumps(coverage(recs)))
    print(json.dumps(episode_moves(a, b)))
    n = 0
    for (feed, episode), v, x, y in changed_voices(a, b):
        n += 1
        print(f"===== {str(feed)[:30]} | {str(episode)[:80]}")
        line = f"   {v}: {_named(x)}/{x.get('role')} -> {_named(y)}/{y.get('role')}"
        if not args.no_evidence:
            try:
                texts = voice_texts(b[(feed, episode)]["meta_path"])
            except (OSError, ValueError, KeyError):
                texts = {}
            total = sum(len(" ".join(t)) for t in texts.values()) or 1
            share = len(" ".join(texts.get(v, []))) / total
            print(f"{line} ({share:.0%} of the words)")
            print("      says:", " / ".join(t[:140] for t in texts.get(v, [])[:2]))
            roster = b[(feed, episode)].get("roster") or {}
            for h in [u for u, r in roster.items() if r.get("role") == "host"][:2]:
                hit = next((t for t in texts.get(h, []) if _HOST_LINE.search(t)), "")
                print(f"      host {h} ({_named(roster[h])}): {hit[:160]}")
        else:
            print(line)
    print(f"changed voices: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
