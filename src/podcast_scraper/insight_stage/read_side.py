"""Ad reads inside a transcript's detected ad ranges, for clips that must never play an ad (moved
verbatim from ``server.app_episode_search``, E04 S3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _detected_ad_reads(raw: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """The ad READS inside the detector's ranges, not the ranges themselves.

    A detected range is the stretch the pipeline would cut, and it can sweep in content: in a
    fixture episode with a pre-roll at 12 s and a break at 42 s it ran 0-54 s, over two real
    points. A clip only needs to never play an ad, so here a segment is an ad when it lies in a
    range AND its own text reads as one, in any language the ad patterns cover
    (``gi.filters.AD_PATTERNS_BY_LANGUAGE``). Reads one line apart in the
    same range join up, so a sponsor read split across three segments is one span."""
    from podcast_scraper.gi.ad_regions import ad_time_intervals
    from podcast_scraper.gi.filters import AD_PATTERNS_BY_LANGUAGE

    ranges = ad_time_intervals(raw)
    if not ranges:
        return []
    patterns = [p for group in AD_PATTERNS_BY_LANGUAGE.values() for p in group]

    def range_of(seg: Dict[str, Any]) -> Optional[int]:
        mid = (float(seg.get("start") or 0.0) + float(seg.get("end") or 0.0)) / 2.0
        return next((i for i, (a, b) in enumerate(ranges) if a <= mid <= b), None)

    reads: List[Tuple[int, int]] = []  # (index in raw, which range)
    for i, seg in enumerate(raw):
        r = range_of(seg)
        if r is not None and any(p.search(str(seg.get("text") or "")) for p in patterns):
            reads.append((i, r))
    spans: List[Tuple[int, int]] = []
    prev: Optional[Tuple[int, int]] = None
    for i, r in reads:
        lo = int(float(raw[i].get("start") or 0.0) * 1000)
        hi = int(float(raw[i].get("end") or 0.0) * 1000)
        if prev is not None and prev[1] == r and i - prev[0] <= 2:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
        prev = (i, r)
    return spans
