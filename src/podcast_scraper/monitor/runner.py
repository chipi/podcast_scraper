"""Monitor subprocess entry point (RFC-065)."""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import psutil
from rich.console import Console
from rich.live import Live

from .dashboard import build_dashboard_panel, build_summary_table, HistoryRow
from .memray_util import MEMRAY_ACTIVE_ENV
from .sampler import CrossProcessSampler
from .status import read_pipeline_status

# Env: force `.monitor.log` instead of `rich.Live` on stderr (profile freeze / RFC-065).
MONITOR_FILE_LOG_ENV = "PODCAST_SCRAPER_MONITOR_FILE_LOG"

SampleSeg = Tuple[float, float, float, float]  # wall_mono, peak_mb, avg_cpu, count


def _monitor_file_log_env_truthy() -> bool:
    val = os.environ.get(MONITOR_FILE_LOG_ENV)
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def monitor_use_rich_live_on_stderr() -> bool:
    """Use Rich Live on stderr when it is a TTY and file logging is not forced."""
    if _monitor_file_log_env_truthy():
        return False
    return sys.stderr.isatty()


def _pid_alive(pid: int) -> bool:
    try:
        return bool(psutil.Process(pid).is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _segment_stats(samples: List[Tuple[float, float, float]], t0: float, t1: float) -> SampleSeg:
    seg = [s for s in samples if t0 <= s[0] <= t1]
    if not seg:
        return (max(0.0, t1 - t0), 0.0, 0.0, 0)
    wall = t1 - t0
    peak = max(s[1] for s in seg)
    avg_cpu = sum(s[2] for s in seg) / len(seg)
    return (wall, peak, avg_cpu, len(seg))


def _plain_tick_line(
    *,
    pipeline_pid: int,
    stage: str,
    episode_idx: Optional[int],
    episode_total: Optional[int],
    rss_mb: float,
    peak_rss_mb: float,
    cpu_pct: float,
    elapsed_s: float,
    stage_elapsed_s: float,
    memray_active: bool,
) -> str:
    ep = ""
    if episode_total is not None:
        idx = episode_idx if episode_idx is not None else 0
        ep = f" episodes={idx}/{episode_total}"
    mem = " memray=1" if memray_active else ""
    return (
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"pid={pipeline_pid} stage={stage}{ep} "
        f"rss_mb={rss_mb:.0f} peak_rss_mb={peak_rss_mb:.0f} cpu_pct={cpu_pct:.1f} "
        f"elapsed_s={elapsed_s:.1f} stage_elapsed_s={stage_elapsed_s:.1f}{mem}"
    )


def _proc_create_time_or_none(pid: int) -> Optional[float]:
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


@dataclass
class _MonitorLoop:
    out: Path
    pipeline_pid: int
    sampler: CrossProcessSampler
    memray_active: bool
    proc_start_time: Optional[float]
    history_rows: List[HistoryRow] = field(default_factory=list)
    current_stage: Optional[str] = None
    stage_mono_start: float = field(default_factory=time.monotonic)
    peak_session_rss: float = 0.0

    def flush_stage(self, end_mono: float, stage_name: str) -> None:
        wall, peak, avg_cpu, n = _segment_stats(
            self.sampler.samples, self.stage_mono_start, end_mono
        )
        if n == 0:
            return
        self.peak_session_rss = max(self.peak_session_rss, peak)
        self.history_rows.append(
            (stage_name, f"{wall:.1f}", f"{peak:.0f}", f"{avg_cpu:.1f}"),
        )

    def tick(self) -> Tuple[bool, Any, str]:
        alive = _pid_alive(self.pipeline_pid)
        st: Optional[Dict[str, Any]] = read_pipeline_status(self.out)
        stage = str((st or {}).get("stage") or "starting")
        now_mono = time.monotonic()

        if stage != self.current_stage:
            if self.current_stage is not None:
                self.flush_stage(now_mono, self.current_stage)
            self.current_stage = stage
            self.stage_mono_start = now_mono

        rss = cpu = 0.0
        if self.sampler.samples:
            _, rss, cpu = self.sampler.samples[-1]
        self.peak_session_rss = max(self.peak_session_rss, rss)

        elapsed = 0.0
        if self.proc_start_time is not None:
            elapsed = max(0.0, time.time() - self.proc_start_time)
        elif st and st.get("started_at"):
            elapsed = max(0.0, time.time() - float(st["started_at"]))

        st_elapsed = 0.0
        if st and st.get("stage_started_at"):
            st_elapsed = max(0.0, time.time() - float(st["stage_started_at"]))

        ep_idx = st.get("episode_idx") if st else None
        ep_tot = st.get("episode_total") if st else None
        ep_i = int(ep_idx) if ep_idx is not None else None
        ep_t = int(ep_tot) if ep_tot is not None else None

        panel = build_dashboard_panel(
            pipeline_pid=self.pipeline_pid,
            stage=stage,
            episode_idx=ep_i,
            episode_total=ep_t,
            rss_mb=rss,
            peak_rss_mb=self.peak_session_rss,
            cpu_pct=cpu,
            elapsed_s=elapsed,
            stage_elapsed_s=st_elapsed,
            history_rows=self.history_rows,
            memray_active=self.memray_active,
        )
        plain = _plain_tick_line(
            pipeline_pid=self.pipeline_pid,
            stage=stage,
            episode_idx=ep_i,
            episode_total=ep_t,
            rss_mb=rss,
            peak_rss_mb=self.peak_session_rss,
            cpu_pct=cpu,
            elapsed_s=elapsed,
            stage_elapsed_s=st_elapsed,
            memray_active=self.memray_active,
        )
        return alive, panel, plain


def _maybe_append_unsegmented_row(loop: _MonitorLoop) -> None:
    if loop.history_rows or not loop.sampler.samples:
        return
    wall, peak, avg_cpu, n = _segment_stats(
        loop.sampler.samples, loop.sampler.samples[0][0], loop.sampler.samples[-1][0]
    )
    if n:
        loop.history_rows.append(("(unsegmented)", f"{wall:.1f}", f"{peak:.0f}", f"{avg_cpu:.1f}"))


def _emit_tty_footer(
    console: Console,
    history_rows: List[HistoryRow],
    peak_session_rss: float,
    pipeline_pid: int,
) -> None:
    total_wall = sum(float(r[1]) for r in history_rows) if history_rows else 0.0
    avg_all = sum(float(r[3]) for r in history_rows) / len(history_rows) if history_rows else 0.0
    console.print()
    if history_rows:
        console.print(
            build_summary_table(history_rows, total_wall, peak_session_rss, avg_all),
        )
    if not _pid_alive(pipeline_pid):
        console.print("[dim]Pipeline process has exited.[/dim]")


def _emit_file_footer(
    log_path: Path,
    history_rows: List[HistoryRow],
    peak_session_rss: float,
    pipeline_pid: int,
) -> None:
    total_wall = sum(float(r[1]) for r in history_rows) if history_rows else 0.0
    avg_all = sum(float(r[3]) for r in history_rows) / len(history_rows) if history_rows else 0.0
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n# summary\n")
        if history_rows:
            lf.write(
                f"total_wall_s={total_wall:.1f} peak_rss_mb={peak_session_rss:.0f} "
                f"avg_cpu_pct={avg_all:.1f}\n",
            )
            for row in history_rows:
                lf.write(f"  stage={row[0]} wall_s={row[1]} peak_mb={row[2]} avg_cpu={row[3]}\n")
        if not _pid_alive(pipeline_pid):
            lf.write("pipeline_exited=1\n")


def _run_tty_loop(loop: _MonitorLoop, console: Console, poll_interval_s: float) -> None:
    rps = min(8, max(1, int(1.0 / max(poll_interval_s, 0.1))))
    with Live(console=console, refresh_per_second=rps) as live:
        while True:
            alive, panel, _ = loop.tick()
            live.update(panel)
            if not alive:
                if loop.current_stage is not None:
                    loop.flush_stage(time.monotonic(), loop.current_stage)
                break
            time.sleep(poll_interval_s)


def _run_file_loop(loop: _MonitorLoop, log_file: TextIO, poll_interval_s: float) -> None:
    while True:
        alive, _, plain = loop.tick()
        log_file.write(plain + "\n")
        log_file.flush()
        if not alive:
            if loop.current_stage is not None:
                loop.flush_stage(time.monotonic(), loop.current_stage)
            break
        time.sleep(poll_interval_s)


def monitor_main(
    pipeline_pid: int,
    output_dir: str,
    *,
    poll_interval_s: float = 0.5,
) -> None:
    """Poll status file and ``psutil``; render ``rich.Live`` or append ``.monitor.log``."""
    out = Path(output_dir)
    sampler = CrossProcessSampler(pipeline_pid, interval_s=poll_interval_s)
    sampler.start()
    use_rich_live = monitor_use_rich_live_on_stderr()
    memray_active = os.environ.get(MEMRAY_ACTIVE_ENV) == "1"
    console = Console(stderr=True)
    log_file: Optional[TextIO] = None
    log_path: Optional[Path] = None
    if not use_rich_live:
        log_path = out / ".monitor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
        reason = (
            "PODCAST_SCRAPER_MONITOR_FILE_LOG"
            if _monitor_file_log_env_truthy()
            else "non-TTY stderr"
        )
        log_file.write(
            f"\n# podcast_scraper monitor ({reason}; append log) "
            f"session_start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        )
        log_file.flush()

    loop = _MonitorLoop(
        out=out,
        pipeline_pid=pipeline_pid,
        sampler=sampler,
        memray_active=memray_active,
        proc_start_time=_proc_create_time_or_none(pipeline_pid),
    )

    try:
        if use_rich_live:
            _run_tty_loop(loop, console, poll_interval_s)
        else:
            assert log_file is not None
            _run_file_loop(loop, log_file, poll_interval_s)
    finally:
        sampler.stop()
        if log_file is not None:
            log_file.close()

    _maybe_append_unsegmented_row(loop)

    if use_rich_live:
        _emit_tty_footer(console, loop.history_rows, loop.peak_session_rss, pipeline_pid)
    elif log_path is not None:
        _emit_file_footer(log_path, loop.history_rows, loop.peak_session_rss, pipeline_pid)


def _monitor_entry(pipeline_pid: int, output_dir: str, *, poll_interval_s: float = 0.5) -> None:
    try:
        monitor_main(pipeline_pid, output_dir, poll_interval_s=poll_interval_s)
    except KeyboardInterrupt:
        pass


def start_monitor_subprocess(*, pipeline_pid: int, output_dir: str) -> BaseProcess:
    """Spawn the monitor (``spawn`` context for macOS/Windows compatibility)."""
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_monitor_entry,
        args=(pipeline_pid, output_dir),
        kwargs={"poll_interval_s": 0.5},
        name="podcast_scraper_monitor",
    )
    proc.start()
    return proc
