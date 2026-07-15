"""Operator-facing token/cost usage snapshot — the read behind ``GET /api/usage`` and the MCP tool.

Self-contained: rolls up the ``llm_cost`` telemetry a run already wrote to disk (``run.log`` /
JSONL under the corpus), sliced by any dimension (model / operation / episode / run / provider …),
with the full token breakdown and de-dup by request_id. Works with NO external o11y backend; when
Loki/Langfuse are present the same events flow there too, but this never depends on them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from podcast_scraper.workflow.token_usage_rollup import (
    rollup_events,
    slice_dimensions,
)

logger = logging.getLogger(__name__)

# Files under a corpus that may carry JSON-line ``llm_cost`` events.
_EVENT_GLOBS = ("**/run.log", "**/*.llm_cost.jsonl", "**/metrics_events.jsonl")


def _iter_events_from_file(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    import json

    for line in text.splitlines():
        brace = line.find("{")
        if brace == -1 or "llm_cost" not in line:
            continue
        try:
            obj = json.loads(line[brace:])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("event_type") == "llm_cost":
            yield obj


def _discover_event_files(root: Path, run_id: Optional[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in _EVENT_GLOBS:
        for p in root.glob(pattern):
            if p.is_file() and (run_id is None or run_id in p.as_posix()):
                files.append(p)
    return sorted(set(files))


def usage_rollup_snapshot(
    source: str | Path,
    *,
    group_by: Tuple[str, ...] = ("provider", "model"),
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Roll up ``llm_cost`` telemetry under ``source`` (a corpus dir, run dir, or single log file).

    ``group_by`` is validated against :func:`slice_dimensions`; unknown dims fall back to
    ``("provider", "model")``. Returns the rollup (total + per-group), the dimensions available to
    slice by, and the files it read — plus an ``uninstrumented`` flag when files were found but no
    ``llm_cost`` events were in them (the old silent-drop symptom, now diagnosable).
    """
    valid = set(slice_dimensions())
    gb = tuple(d for d in group_by if d in valid) or ("provider", "model")

    root = Path(source)
    files: List[Path]
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = _discover_event_files(root, run_id)
    else:
        files = []

    events: List[Dict[str, Any]] = []
    for f in files:
        events.extend(_iter_events_from_file(f))

    result = rollup_events(events, group_by=gb)
    payload = result.as_dict()
    payload["dimensions"] = slice_dimensions()
    payload["source_files"] = [f.as_posix() for f in files]
    payload["run_id"] = run_id
    payload["uninstrumented"] = bool(files) and result.total.calls == 0
    return payload
