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
from typing import FrozenSet

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
