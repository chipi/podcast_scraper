"""Optional stdin hook to trigger py-spy CPU flamegraphs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_PY_SPY_HINT_EMITTED = False
_RECORD_LOCK = threading.Lock()


def _emit_py_spy_missing_hint_once() -> None:
    global _PY_SPY_HINT_EMITTED
    if _PY_SPY_HINT_EMITTED:
        return
    _PY_SPY_HINT_EMITTED = True
    print(
        "Monitor: press 'f' to capture a CPU flamegraph (requires py-spy; "
        "pip install -e '.[monitor]').",
        file=sys.stderr,
    )


def _record_flamegraph(py_spy_bin: str, output_dir: str) -> None:
    dbg = Path(output_dir) / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_svg = dbg / f"flamegraph_{ts}.svg"
    pid = os.getpid()
    cmd = [
        py_spy_bin,
        "record",
        "-p",
        str(pid),
        "-d",
        "5",
        "-o",
        str(out_svg),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            print(f"py-spy: wrote {out_svg}", file=sys.stderr)
        else:
            err = (proc.stderr or proc.stdout or "").strip()
            print(
                f"py-spy record failed (exit {proc.returncode})" + (f": {err}" if err else ""),
                file=sys.stderr,
            )
    except subprocess.TimeoutExpired:
        print("py-spy record timed out.", file=sys.stderr)
    except OSError as exc:
        print(f"py-spy record failed: {exc}", file=sys.stderr)


def _stdin_listener_loop(stop: threading.Event, output_dir: str, py_spy_bin: str) -> None:
    while not stop.is_set():
        try:
            import select

            readable, _, _ = select.select([sys.stdin], [], [], 0.5)
        except (OSError, ValueError):
            break
        if not readable:
            continue
        try:
            chunk = sys.stdin.read(1)
        except OSError:
            break
        if not chunk:
            continue
        if chunk in ("f", "F"):
            with _RECORD_LOCK:
                _record_flamegraph(py_spy_bin, output_dir)


def start_py_spy_stdin_listener(
    *,
    output_dir: str,
    enabled: bool,
) -> Optional[Callable[[], None]]:
    """Start a daemon thread that watches stdin for ``f`` to run ``py-spy record``.

    Only starts when ``enabled``, stdin is a TTY, and ``py-spy`` is on ``PATH``.
    On unsupported platforms (e.g. Windows console), returns ``None`` without starting.

    Args:
        output_dir: Run directory; flamegraphs go under ``debug/flamegraph_*.svg``.
        enabled: Typically ``cfg.monitor``.

    Returns:
        A callable to request thread shutdown, or ``None`` if no listener was started.
    """
    if not enabled:
        return None
    if not sys.stdin.isatty():
        return None
    if sys.platform == "win32":
        return None
    py_spy_bin = shutil.which("py-spy")
    if not py_spy_bin:
        _emit_py_spy_missing_hint_once()
        return None

    stop = threading.Event()

    thread = threading.Thread(
        target=_stdin_listener_loop,
        args=(stop, output_dir, py_spy_bin),
        name="podcast_scraper_py_spy_stdin",
        daemon=True,
    )
    thread.start()

    def _shutdown() -> None:
        stop.set()

    return _shutdown
