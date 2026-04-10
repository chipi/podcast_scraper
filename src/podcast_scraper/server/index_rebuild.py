"""Background vector index rebuild coordination (GitHub #507 follow-up)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI

logger = logging.getLogger(__name__)


class CorpusRebuildGate:
    """One mutex + status per resolved corpus path (in-memory; single-worker oriented)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._last_error: str | None = None

    def try_begin(self) -> bool:
        """Return True if this call started a rebuild slot; False if already running."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._last_error = None
            return True

    def end(self, error_message: str | None) -> None:
        """Mark rebuild finished and store the last error message (if any)."""
        with self._lock:
            self._running = False
            self._last_error = error_message

    def snapshot(self) -> Tuple[bool, str | None]:
        """Return ``(rebuild_in_progress, last_error)`` for stats polling."""
        with self._lock:
            return self._running, self._last_error


def _gates_map(app: FastAPI) -> Dict[str, CorpusRebuildGate]:
    raw = getattr(app.state, "index_rebuild_gates", None)
    if raw is None:
        raw = {}
        app.state.index_rebuild_gates = raw
    return raw


def gate_for_corpus(app: FastAPI, corpus_resolved: Path) -> CorpusRebuildGate:
    """Return the mutex gate for a corpus path, creating it on first use."""
    key = str(corpus_resolved.resolve())
    gates = _gates_map(app)
    if key not in gates:
        gates[key] = CorpusRebuildGate()
    return gates[key]


def rebuild_status_snapshot(app: FastAPI, corpus_resolved: Path) -> Tuple[bool, str | None]:
    """Return ``(in_progress, last_error)`` for index stats / UI polling."""
    return gate_for_corpus(app, corpus_resolved).snapshot()
