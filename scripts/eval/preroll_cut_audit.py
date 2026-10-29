#!/usr/bin/env python3
"""How much talk before the first ad pattern does the forced start=0 pre-roll cut take? (read-only)

Usage::

    .venv/bin/python scripts/eval/preroll_cut_audit.py <dir of *.segments.json> out.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from podcast_scraper.gi.ad_regions import _distinct_hits, excise_ad_regions, SCAN_CHARS


def _talk(segs: list[dict[str, Any]]) -> float:
    return sum(max(0.0, float(s.get("end") or 0) - float(s.get("start") or 0)) for s in segs)


def audit(path: Path, root: Path) -> dict[str, Any] | None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    segs = [s for s in segs if isinstance(s, dict)]
    spans, parts, cursor = [], [], 0
    for s in segs:
        t = str(s.get("text") or "")
        parts.append(t)
        spans.append((cursor, cursor + len(t), s))
        cursor += len(t) + 1
    text = " ".join(parts)
    _, _, meta = excise_ad_regions(text)
    if not meta.excised_ranges or meta.excised_ranges[0][0] != 0:
        return None
    hi = meta.excised_ranges[0][1]
    hits = _distinct_hits(text[:SCAN_CHARS])
    first_hit = hits[0][0] if hits else hi
    cut = [s for a, _, s in spans if a < hi]
    lead = [s for _, b, s in spans if b <= first_hit]
    hit_seg = next((s for _, b, s in spans if b > first_hit), spans[-1][2])
    return {
        "path": str(path.relative_to(root)),
        "cut_end_s": round(float(cut[-1].get("end") or 0), 1),
        "first_hit_s": round(float(hit_seg.get("start") or 0), 1),
        "lead_s": round(_talk(lead), 1),
        "lead_text": " ".join(str(s.get("text") or "") for s in lead)[:400],
        "hit_text": text[first_hit : first_hit + 120],
    }


def main() -> int:
    root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    out = [r for p in sorted(root.rglob("*.segments.json")) if (r := audit(p, root))]
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    n = len(out)
    print("episodes with a pre-roll cut:", n)
    if not n:
        return 1
    for key in ("cut_end_s", "first_hit_s", "lead_s"):
        v = sorted(o[key] for o in out)
        print(f"{key}: median {statistics.median(v)} p90 {v[int(0.9 * n)]} max {v[-1]}")
    for th in (15, 30, 60, 120):
        print(f"lead > {th}s: {sum(o['lead_s'] > th for o in out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
