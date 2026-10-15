#!/usr/bin/env python3
"""Replay the ad detector's cut ranges, old code vs new, over a directory of segment files (#2360).

``run`` writes, per episode, the ranges ``excise_ad_regions`` cuts from the segments' joined text
(the text ``ad_time_intervals`` reads). Run it under the code before the fix and after, then
``--compare`` with the hand-read categories of the 289 prod episodes that had a pre-roll cut
(``short`` and ``promo``: the lead before the first ad pattern is an ad or show promo; ``CONTENT``:
it is the episode's own talk; see docs/wip/PREROLL-CUT-PROD-AUDIT-2026-10-15.md).

    PYTHONPATH=<old checkout>/src python scripts/eval/replay_preroll_cut.py run DIR old.json
    python scripts/eval/replay_preroll_cut.py run DIR new.json
    python scripts/eval/replay_preroll_cut.py compare old.json new.json categories.json

Pass criteria (exit 1 on any failure):
- every CONTENT episode's first cut no longer opens at 0, and keeps its talk up to at most
  PREROLL_PITCH chars before the first ad pattern;
- a CONTENT read that starts with an opener ("sponsored by") is cut from that sentence, no earlier;
- every short / promo episode's cut is unchanged;
- every episode keeps every range after its first unchanged, and an episode whose first range did
  not open at 0 is unchanged entirely (nothing outside the pre-roll moves).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def run(root: Path, out: Path) -> int:
    from podcast_scraper.gi.ad_regions import excise_ad_regions

    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*.segments.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
        text = " ".join(str(s.get("text") or "") for s in segs if isinstance(s, dict))
        _, _, meta = excise_ad_regions(text)
        result[str(path.relative_to(root))] = [list(r) for r in meta.excised_ranges]
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"{len(result)} episodes -> {out}")
    return 0 if result else 1


def compare(old_p: Path, new_p: Path, cats_p: Path, pitch: int = 600, lookahead: int = 300) -> int:
    old = json.loads(old_p.read_text(encoding="utf-8"))
    new = json.loads(new_p.read_text(encoding="utf-8"))
    # categories.json: [{path, category, first_hit, opener}, ...] -- the hand-read audit, with the
    # char offset of each episode's first ad pattern and whether that pattern opens a read.
    rows = json.loads(cats_p.read_text(encoding="utf-8"))
    cats = {row["path"]: (row["category"], row["first_hit"]) for row in rows}
    opener = {row["path"] for row in rows if row.get("opener")}
    failures: list[str] = []
    changed = 0
    for path, o in old.items():
        n = new.get(path)
        if n is None:
            failures.append(f"missing in new: {path}")
            continue
        if o != n:
            changed += 1
        cat = cats.get(path)
        if not (o and o[0][0] == 0):
            if o != n:
                failures.append(f"no pre-roll, yet changed: {path}")
            continue
        if o[1:] != n[1:]:
            failures.append(f"a later range moved: {path}")
        if cat and cat[0] == "CONTENT":
            first_hit = cat[1]
            if not n or n[0][0] == 0:
                failures.append(f"CONTENT still cut from 0: {path}")
            elif n[0][0] < first_hit - pitch:
                failures.append(f"CONTENT cut starts {first_hit - n[0][0]} chars early: {path}")
            elif path in opener and n[0][0] < first_hit - lookahead:
                # An opener ("sponsored by") starts the read: the cut starts in its sentence.
                failures.append(f"opener read cut {first_hit - n[0][0]} chars early: {path}")
        elif cat and o != n:
            failures.append(f"{cat[0]} episode changed: {path}")
    content = sum(1 for c, _ in cats.values() if c == "CONTENT")
    print(f"{len(old)} episodes · {changed} changed · {content} CONTENT expected to change")
    for f in failures[:20]:
        print("FAIL", f)
    print("PASS" if not failures else f"FAIL: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "run" and len(sys.argv) == 4:
        raise SystemExit(run(Path(sys.argv[2]), Path(sys.argv[3])))
    if mode == "compare" and len(sys.argv) == 5:
        raise SystemExit(compare(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
    print(__doc__)
    raise SystemExit(2)
