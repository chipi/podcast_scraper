"""A pre-existing root handler must never silence the process (#1807).

2026-08-22. Two production incidents presented as SILENCE rather than as slowness, and every
log-capture mitigation built during them captured nothing. The cause was not the capture; it was
that nothing wrote to stdout at all.

``apply_log_level`` used to read:

    if not root_logger.handlers:
        <add a console handler>
    else:
        <only update the handlers already there>

In the prod image OpenTelemetry attaches its own handler to the ROOT logger at import, so
``handlers`` was never empty and the else-branch ran. Measured inside the deployed image:

    ROOT HANDLERS after import:          [<LoggingHandler (NOTSET)>]
    ROOT HANDLERS after apply_log_level: [<LoggingHandler (INFO)>]
    HAS CONSOLE HANDLER: False

and the probe's own warning line never appeared. A 31-minute run emitted three application log
lines, all three from the reindex SUBPROCESS, which calls ``logging.basicConfig`` itself.

It does not reproduce on a dev box: without the OTEL extra the root logger is empty at import and
the old code took the happy branch. So these tests install a stand-in handler to force the
condition that only production had.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.workflow.orchestration import apply_log_level

pytestmark = [pytest.mark.unit]


class _ForeignHandler(logging.Handler):
    """Stands in for opentelemetry.sdk._logs.LoggingHandler: on root, not a console."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def _console_handlers(handlers) -> list:
    """Console = a StreamHandler that is NOT a FileHandler (FileHandler subclasses it, so a
    --log-file run must not be mistaken for having a console)."""
    return [
        h
        for h in handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]


def _added_by(call, root) -> list:
    """Handlers apply_log_level ADDED, measured as a delta.

    pytest's LogCaptureHandler is itself a StreamHandler and the logging plugin attaches it to
    ROOT at the start of the call phase — after fixtures run — so it cannot be cleared in a
    fixture. Left in place it makes the product correctly conclude "a console already exists",
    and the test would then assert against the harness rather than the code. Park it for the
    duration of the measurement and put it back.
    """
    parked = [h for h in root.handlers if type(h).__module__.startswith("_pytest")]
    for h in parked:
        root.removeHandler(h)
    try:
        before = list(root.handlers)
        call()
        return [h for h in root.handlers if h not in before]
    finally:
        for h in parked:
            root.addHandler(h)


def test_a_foreign_root_handler_does_not_suppress_the_console(clean_root) -> None:
    """THE regression. Red before this fix: exactly the prod condition."""
    clean_root.addHandler(_ForeignHandler())

    added = _added_by(lambda: apply_log_level("INFO", None, False), clean_root)

    assert _console_handlers(added), (
        "no console handler was attached because the root logger already had one from another "
        "library — this is the #1807 silence: every app log line goes to that handler and NOTHING "
        "reaches stdout, so docker logs, the box-local tee, the Actions log and VictoriaLogs are "
        "all empty"
    )


def test_the_console_handler_writes_to_stderr(clean_root) -> None:
    """Not just present — pointed somewhere docker actually captures."""
    import sys

    clean_root.addHandler(_ForeignHandler())
    added = _added_by(lambda: apply_log_level("INFO", None, False), clean_root)
    assert [h for h in _console_handlers(added) if h.stream is sys.stderr]


def test_an_empty_root_still_gets_exactly_one_console(clean_root) -> None:
    """The path that always worked must keep working, and must not double up."""
    added = _added_by(lambda: apply_log_level("INFO", None, False), clean_root)
    assert len(_console_handlers(added)) == 1


def test_repeated_calls_do_not_stack_console_handlers(clean_root) -> None:
    """cli calls this per batch and run_pipeline calls it per feed — duplicates would N-plicate
    every line, which is its own kind of unreadable."""

    def _call_four_times() -> None:
        for _ in range(4):
            apply_log_level("INFO", None, False)

    added = _added_by(_call_four_times, clean_root)
    assert len(_console_handlers(added)) == 1, (
        "four calls added more than one console handler — every log line would appear that many "
        "times"
    )


def test_the_foreign_handler_is_kept(clean_root) -> None:
    """The fix ADDS a console; it must not evict OTEL's handler and break log shipping."""
    foreign = _ForeignHandler()
    clean_root.addHandler(foreign)
    _added_by(lambda: apply_log_level("INFO", None, False), clean_root)
    assert foreign in clean_root.handlers


def test_a_line_logged_after_setup_actually_reaches_the_console(clean_root) -> None:
    """The end-to-end property the incidents needed: emit, then SEE it.

    The handler's stream is swapped for a StringIO so the assertion reads what the app wrote,
    rather than trusting pytest's capture plumbing.
    """
    import io

    clean_root.addHandler(_ForeignHandler())
    added = _added_by(lambda: apply_log_level("INFO", None, False), clean_root)
    console = _console_handlers(added)[0]
    console.stream = io.StringIO()

    logging.getLogger("podcast_scraper.test_probe").warning("PROBE LINE")

    assert "PROBE LINE" in console.stream.getvalue()
