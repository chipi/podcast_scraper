"""Memray process wrapper for heap profiling (RFC-065 Phase 3)."""

from __future__ import annotations

import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional, Sequence

MEMRAY_ACTIVE_ENV = "PODCAST_SCRAPER_MEMRAY_ACTIVE"

_CLI_MODULE = "podcast_scraper.cli"
_SERVICE_MODULE = "podcast_scraper.service"


def _default_memray_path_for_cli(args: Namespace, feed_urls: Sequence[str]) -> Path:
    from ..utils import filesystem

    corpus = (getattr(args, "output_dir", None) or "").strip()
    if len(feed_urls) >= 2 and corpus:
        base = Path(corpus)
    elif feed_urls:
        base = Path(filesystem.derive_output_dir(feed_urls[0], getattr(args, "output_dir", None)))
    else:
        base = Path(".")
    dbg = base / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return dbg / f"memray_{ts}.bin"


def _default_memray_path_for_cfg(output_dir: Optional[str]) -> Path:
    od = (output_dir or "").strip()
    base = Path(od) if od else Path.cwd()
    dbg = base / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return dbg / f"memray_{ts}.bin"


def resolve_memray_output_cli(args: Namespace, feed_urls: Sequence[str]) -> Path:
    """Resolve memray capture path for CLI runs."""
    raw = getattr(args, "memray_output", None)
    if raw is not None and str(raw).strip():
        return Path(str(raw).strip()).expanduser().resolve()
    return _default_memray_path_for_cli(args, feed_urls)


def resolve_memray_output_service(output_dir: Optional[str], memray_output: Optional[str]) -> Path:
    """Resolve memray capture path for service runs."""
    if memray_output is not None and str(memray_output).strip():
        return Path(str(memray_output).strip()).expanduser().resolve()
    return _default_memray_path_for_cfg(output_dir)


def maybe_reexec_memray_cli(args: Namespace, argv: Sequence[str], feed_urls: Sequence[str]) -> None:
    """Re-exec the CLI under ``memray run`` when requested; otherwise no-op.

    Uses :envvar:`PODCAST_SCRAPER_MEMRAY_ACTIVE` to avoid re-exec loops.

    Raises:
        SystemExit: If memray is requested but the ``memray`` executable is not on ``PATH``.
    """
    if os.environ.get(MEMRAY_ACTIVE_ENV) == "1":
        return
    if not bool(getattr(args, "memray", False)):
        return
    memray_bin = shutil.which("memray")
    if not memray_bin:
        print(
            "Error: --memray requires the memray executable on PATH "
            "(install optional extra: pip install -e '.[monitor]').",
            file=sys.stderr,
        )
        raise SystemExit(1)
    out = resolve_memray_output_cli(args, feed_urls)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[MEMRAY_ACTIVE_ENV] = "1"
    cmd = [memray_bin, "run", "-o", str(out), "-f", "-m", _CLI_MODULE, *list(argv)]
    os.execvpe(memray_bin, cmd, env)


def maybe_reexec_memray_service(
    *,
    memray: bool,
    output_dir: Optional[str],
    memray_output: Optional[str],
) -> Optional[str]:
    """Re-exec the service entrypoint under memray, or return an error message.

    Returns:
        Error text when memray was requested but ``memray`` is missing; ``None`` otherwise.
        Does not return if re-exec succeeds.
    """
    if os.environ.get(MEMRAY_ACTIVE_ENV) == "1":
        return None
    if not memray:
        return None
    memray_bin = shutil.which("memray")
    if not memray_bin:
        return (
            "memray not on PATH; install optional extra: pip install -e '.[monitor]' "
            "(RFC-065 Phase 3)."
        )
    out = resolve_memray_output_service(output_dir, memray_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[MEMRAY_ACTIVE_ENV] = "1"
    cmd = [memray_bin, "run", "-o", str(out), "-f", "-m", _SERVICE_MODULE, *sys.argv[1:]]
    os.execvpe(memray_bin, cmd, env)
