"""Name what the api is doing when its memory climbs (player-api OOM kills, 2026-10-17/19).

player-api's main process went from ~0.8 GB to 3.8-8 GB within a minute, three times, each during
an app home-screen burst shortly after a start. Replaying every request path and warmer step alone
peaked at about 1 GB, so the allocation only shows up live. This daemon thread samples the
process's resident memory; once it crosses ``PODCAST_MEM_WATCHDOG_MB`` it logs one WARNING with
the in-flight requests and the innermost project frames of every thread, and again at each
further 512 MB. The next spike then names its own code path.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from typing import Callable, Optional

from podcast_scraper.utils.access_log import inflight_requests

logger = logging.getLogger(__name__)

STEP_MB = 512
_PROJECT_MARK = os.sep + "podcast_scraper" + os.sep


def rss_mb() -> int:
    """Resident set size of this process in MB (Linux ``/proc``); 0 where unavailable."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _thread_frames(limit: int = 4) -> list[str]:
    """Each thread's innermost project frames (file:line function), outermost first."""
    names = {t.ident: t.name for t in threading.enumerate()}
    out: list[str] = []
    for ident, frame in sys._current_frames().items():
        ours = [
            f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
            for f in traceback.extract_stack(frame)
            if _PROJECT_MARK in f.filename and "mem_watchdog" not in f.filename
        ]
        if ours:
            out.append(f"{names.get(ident, ident)}: " + " > ".join(ours[-limit:]))
    return out


def report(rss: int) -> str:
    """The WARNING text: memory, in-flight requests (oldest first), and thread frames."""
    now = time.monotonic()
    reqs = sorted(inflight_requests(), key=lambda r: r[2])
    req_txt = "; ".join(f"{m} {p} {now - t0:.1f}s" for m, p, t0 in reqs) or "none"
    frames = " | ".join(_thread_frames()) or "none"
    return f"mem watchdog: rss={rss}MB in_flight=[{req_txt}] threads=[{frames}]"


def watch(
    threshold_mb: int,
    stop: threading.Event,
    *,
    interval_s: float = 1.0,
    sample: Callable[[], int] = rss_mb,
) -> None:
    """Report once above ``threshold_mb`` and again each ``STEP_MB`` higher; re-arm below it."""
    next_mb = threshold_mb
    while not stop.wait(interval_s):
        rss = sample()
        if rss >= next_mb:
            logger.warning("%s", report(rss))
            next_mb = rss + STEP_MB
        elif rss < threshold_mb:
            next_mb = threshold_mb


def start_mem_watchdog() -> Optional[threading.Event]:
    """Start the watchdog thread when ``PODCAST_MEM_WATCHDOG_MB`` is a positive integer."""
    try:
        threshold = int(os.environ.get("PODCAST_MEM_WATCHDOG_MB", "0") or 0)
    except ValueError:
        logger.warning("PODCAST_MEM_WATCHDOG_MB is not an integer; memory watchdog off")
        return None
    if threshold <= 0:
        return None
    stop = threading.Event()
    threading.Thread(target=watch, args=(threshold, stop), name="mem-watchdog", daemon=True).start()
    logger.info("memory watchdog on: reports above %d MB", threshold)
    return stop
