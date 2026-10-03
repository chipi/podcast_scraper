"""Show sidecar events: what happened to a SHOW during a run, appended as it happens.

The show sidecar (``feeds/<feed>/show.json``, see :mod:`show_metadata`) is folded at the end of a
feed run from two inputs. Most of it is read back from artifacts already on disk; this module is
the other input — the things that happen during a run and are written NOWHERE else:

* ``hosts_detected``     — the show's host set and the branch that produced it;
* ``transcript_refused`` — a publisher transcript refused because it does not say who speaks;
* ``kg_extraction_failed`` — the model's reply could not be parsed into a graph;
* ``stage_failed``       — a stage reported failure without raising;
* ``error``              — any ERROR log line raised while the run was bound to this show.

One file per run (``feeds/<feed>/show_events/<run_dir>.jsonl``), append-only, so the pipeline's
parallel workers never rewrite each other's lines and a run that dies before finalize still leaves
its events for the next fold. Bound once per feed run: a batch processes feeds strictly one after
another in a process (``utils/correlation.py``), so one module-level binding is enough and thread-
pool workers see it.

Every path is best-effort. A sidecar must never fail the run it describes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .show_metadata import feed_dir_for_run

logger = logging.getLogger(__name__)

EVENTS_SUBDIR = "show_events"
#: Longest message kept from an ERROR log line — enough to identify it, bounded so one huge
#: traceback-in-a-message cannot balloon the sidecar.
_MAX_MESSAGE = 600

_EVENTS_PATH: Optional[Path] = None
_HANDLER: Optional[logging.Handler] = None


def events_path() -> Optional[Path]:
    """The file the current run appends to, or None when no show is bound."""
    return _EVENTS_PATH


def record_show_event(kind: str, **fields: Any) -> None:
    """Append one event to the bound show's sidecar. A no-op when no show is bound."""
    path = _EVENTS_PATH
    if path is None:
        return
    try:
        from ..obs.events import emit_event

        emit_event(kind, sink="file", path=path, **fields)
    except Exception:  # noqa: BLE001 — the sidecar must never break the run
        logger.debug("show event %s not recorded", kind, exc_info=True)


class _ErrorsToShow(logging.Handler):
    """Copies every ERROR (and worse) record into the bound show's sidecar."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(__name__) or record.name == "podcast_scraper.events":
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — a malformed record must not raise from logging
            message = str(record.msg)
        record_show_event(
            "error",
            logger_name=record.name,
            level=record.levelname,
            message=message[:_MAX_MESSAGE],
            exception=(
                record.exc_info[0].__name__
                if record.exc_info and record.exc_info[0] is not None
                else None
            ),
        )


def bind_show(run_dir: Path | str) -> Optional[Path]:
    """Start recording this run's show events. Returns the events file, or None (no feed layout).

    Binding again (the next feed of a batch) replaces the previous binding.
    """
    global _EVENTS_PATH, _HANDLER
    unbind_show()
    run = Path(run_dir)
    feed_dir = feed_dir_for_run(run)
    if feed_dir is None:
        return None
    _EVENTS_PATH = feed_dir / EVENTS_SUBDIR / f"{run.name}.jsonl"
    _HANDLER = _ErrorsToShow()
    logging.getLogger("podcast_scraper").addHandler(_HANDLER)
    return _EVENTS_PATH


def unbind_show() -> None:
    """Stop recording; remove the error handler."""
    global _EVENTS_PATH, _HANDLER
    if _HANDLER is not None:
        logging.getLogger("podcast_scraper").removeHandler(_HANDLER)
    _HANDLER = None
    _EVENTS_PATH = None
