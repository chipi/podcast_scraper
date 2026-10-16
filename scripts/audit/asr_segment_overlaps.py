"""Served episodes whose ASR segments overlap in time — the #2370 word-timestamp drift. Read-only.

Before ``e7234aaec``, a word split across two segments by faster-whisper-server shifted every later
segment's words, so the stored segments drift by up to minutes and two passages end up sharing the
same timestamps. The visible symptom is a segment that starts before the previous one has ended.

Measured on prod 2026-10-16 (2,504 served episodes with segments): 2,023 have no overlap and
646 have 1-9 small ones (worst under 5 s in all but 4), which is ordinary ASR jitter. The drift
shows as hundreds: 8 episodes had 175-2,330 overlaps with a worst of 2.5-26.5 s. Two of the ten
episodes found by listening had only 3 and 5 overlaps, so a low count does NOT clear an episode.

Judged on the copy the API serves (``dedupe_metadata_paths_newest_run_per_episode``). One JSON
line per served episode: guid, metadata path, segment count, overlaps, worst overlap in seconds.

Usage (inside the api container)::

    python scripts/audit/asr_segment_overlaps.py --corpus-dir /app/output > overlaps.jsonl
    python scripts/audit/asr_segment_overlaps.py --corpus-dir /app/output --min-overlaps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Overlaps smaller than this are rounding between adjacent segments, not drift.
TOLERANCE_S = 0.05


def overlaps(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How many segments start before the latest end seen so far, and by how much at worst."""
    count = 0
    worst = 0.0
    prev_end: Optional[float] = None
    for seg in segments:
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or 0)
        if prev_end is not None and start < prev_end - TOLERANCE_S:
            count += 1
            worst = max(worst, prev_end - start)
        prev_end = end if prev_end is None else max(prev_end, end)
    return {"segs": len(segments), "overlaps": count, "worst_s": round(worst, 1)}


def scan(root: Path) -> List[Dict[str, Any]]:
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
    )

    mds = [p for p in root.rglob("*.metadata.json") if ".trash" not in p.parts]
    rows = []
    for md in dedupe_metadata_paths_newest_run_per_episode(root, mds):
        stem = md.name[: -len(".metadata.json")]
        seg_path = md.parent.parent / "transcripts" / f"{stem}.segments.json"
        episode = (json.loads(md.read_text(encoding="utf-8")).get("episode")) or {}
        row: Dict[str, Any] = {
            "guid": episode.get("guid") or episode.get("episode_id"),
            "md": str(md.relative_to(root)),
        }
        if not seg_path.is_file():
            rows.append({**row, "segs": 0, "overlaps": None})
            continue
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        segments = data if isinstance(data, list) else data.get("segments") or []
        rows.append({**row, **overlaps(segments)})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--min-overlaps", type=int, default=0, help="print only episodes with at least this many"
    )
    args = parser.parse_args(argv)
    for row in scan(args.corpus_dir):
        if (row.get("overlaps") or 0) >= args.min_overlaps:
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
