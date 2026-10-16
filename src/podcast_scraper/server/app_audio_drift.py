"""Episodes whose served audio no longer matches the audio we transcribed (#2359, option C).

Some podcast hosts insert ads into an episode when it is fetched, and change them over time. Our
clip times are right for the audio we transcribed; a listener who gets a different ad load hears
the wrong moment (measured 2026-10-15: every clip of 3 podtrac / BBC episodes off by 9-52 s).

``scripts/ops/audio_drift_check.py`` compares each episode's served length with the length we
transcribed and writes the episodes that differ to ``<corpus>/audio_timing_drift.json``:

    {"checked_at": "...", "episodes": {"<metadata_relative_path>": {"served_s": ..., "ref_s": ...}}}

This module reads that file (cached until it changes). A short listen or a Learn day stitches clips
from many episodes with nothing around them to recover from a wrong second, so those leave drifted
episodes out (``app_goal_reel._episode_rows``). The file is optional: absent or unreadable means
no episode is known to have drifted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

DRIFT_FILE = "audio_timing_drift.json"
_NS = "audio_timing_drift"


def _load(path: Path) -> FrozenSet[str]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    episodes = doc.get("episodes") if isinstance(doc, dict) else None
    return frozenset(str(k) for k in episodes) if isinstance(episodes, dict) else frozenset()


def drifted_episodes(root: Path) -> FrozenSet[str]:
    """Metadata paths (relative to the corpus) of episodes whose served audio has drifted."""
    from podcast_scraper import perf_cache

    path = Path(root) / DRIFT_FILE
    try:
        token = os.stat(path).st_mtime
    except OSError:
        return frozenset()
    result: FrozenSet[str] = perf_cache.get_or_compute(_NS, str(path), token, lambda: _load(path))
    return result


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
    """Per episode (metadata path), anchors ``(transcript_s, offset_s)`` written by
    ``scripts/ops/audio_retime.py``: from ``transcript_s`` on, the served stream runs ``offset_s``
    seconds later than the transcript. Absent or unreadable file means no maps."""
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
