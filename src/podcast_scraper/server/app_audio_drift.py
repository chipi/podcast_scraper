"""Time maps for episodes whose served audio no longer matches what we transcribed (#2359,
RFC-129).

Some podcast hosts insert ads into an episode when it is fetched, and change them over time, so
the stream a listener plays can be shifted from our transcript. ``build-maps`` lines the archived
original up with the stream as served now and writes ``<corpus>/audio_time_maps.json``. The api
sends an episode's map with it, and each reel clip's ``play_start_ms`` / ``play_end_ms``; the
player converts with the map. No episode is hidden for drifting.

``DRIFT_FILE`` (``drift-check``) lists the episodes whose served length differs; ``build-maps``
reads it to choose what to map. Both: ``python -m podcast_scraper.audio_sync``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

DRIFT_FILE = "audio_timing_drift.json"


MAPS_FILE = "audio_time_maps.json"
_MAPS_NS = "audio_time_maps"
Anchors = Tuple[Tuple[float, float], ...]


def _load_maps(path: Path) -> Dict[str, Anchors]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: Dict[str, Anchors] = {}
    for rel, entry in (doc.items() if isinstance(doc, dict) else []):
        anchors = entry.get("anchors") if isinstance(entry, dict) else None
        try:
            pairs = tuple(sorted((float(a), float(o)) for a, o in anchors or []))
        except (TypeError, ValueError):
            continue
        if pairs:
            out[str(rel)] = pairs
    return out


def time_maps(root: Path) -> Dict[str, Anchors]:
    """Per episode (metadata path), anchors ``(transcript_s, offset_s)`` written by ``build-maps``
    (``python -m podcast_scraper.audio_sync``): from ``transcript_s`` on, the served stream runs
    ``offset_s`` seconds later than the transcript. Absent or unreadable file means no maps."""
    from podcast_scraper import perf_cache

    path = Path(root) / MAPS_FILE
    try:
        token = os.stat(path).st_mtime
    except OSError:
        return {}
    result: Dict[str, Anchors] = perf_cache.get_or_compute(
        _MAPS_NS, str(path), token, lambda: _load_maps(path)
    )
    return result


def to_stream(anchors: Anchors, t: float) -> float:
    """Transcript time ``t`` as a time in the served stream."""
    off = 0.0
    for at, o in anchors:
        if t >= at:
            off = o
        else:
            break
    return t + off


def episode_anchors(root: Path, metadata_relpath: str) -> Optional[Anchors]:
    """The episode's map, or ``None`` when its stream needs none."""
    return time_maps(root).get(metadata_relpath)


def play_window(
    root: Path, metadata_relpath: str, start_ms: int, end_ms: int
) -> Optional[Tuple[int, int]]:
    """A clip's window in the stream served now. ``None`` when an ad changed inside the clip: the
    window would play the ad, so a reel leaves the clip out."""
    anchors = episode_anchors(root, metadata_relpath)
    if not anchors:
        return start_ms, end_ms
    s, e = to_stream(anchors, start_ms / 1000.0), to_stream(anchors, (end_ms - 1) / 1000.0)
    if abs((e - s) - (end_ms - 1 - start_ms) / 1000.0) > 0.05:
        return None
    return round(s * 1000), round(s * 1000) + (end_ms - start_ms)
