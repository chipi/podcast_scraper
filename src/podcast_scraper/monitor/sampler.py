"""Background RSS / CPU sampling for a remote PID (RFC-065)."""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import psutil

SampleTuple = Tuple[float, float, float]  # monotonic_s, rss_mb, cpu_percent


class CrossProcessSampler:
    """Poll ``psutil.Process(pid)`` on a background thread."""

    def __init__(self, pid: int, interval_s: float = 0.5) -> None:
        self._pid = pid
        self._interval = max(0.05, float(interval_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[SampleTuple] = []

    def start(self) -> None:
        """Begin background sampling until :meth:`stop` is called."""

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    proc = psutil.Process(self._pid)
                    with proc.oneshot():
                        rss_mb = proc.memory_info().rss / (1024**2)
                        cpu = proc.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                self.samples.append((time.monotonic(), rss_mb, cpu))
                if len(self.samples) > 30_000:
                    self.samples = self.samples[-20_000:]
                if self._stop.wait(self._interval):
                    break

        self._thread = threading.Thread(target=_loop, name="CrossProcessSampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampler thread to exit and wait briefly for join."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2.0)
