"""Rich layout for the live monitor."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HistoryRow = Tuple[str, str, str, str]  # stage, wall_s, peak_mb, cpu_pct


def build_dashboard_panel(
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
    history_rows: Sequence[HistoryRow],
    memray_active: bool = False,
) -> Panel:
    """Single refresh frame for ``rich.Live``."""
    title = Text("podcast_scraper — live monitor", style="bold")
    head = Text.assemble(
        ("PID: ", "dim"),
        (str(pipeline_pid), "cyan"),
        ("  ·  Elapsed: ", "dim"),
        (_fmt_duration(elapsed_s), "white"),
    )
    ep_part = ""
    if episode_total is not None:
        idx = episode_idx if episode_idx is not None else 0
        ep_part = f"  ({idx}/{episode_total} episodes)"
    stage_line = Text.assemble(
        ("Stage: ", "dim"),
        (stage, "bold yellow"),
        (ep_part, "dim"),
    )
    st_el = Text.assemble(
        ("Stage elapsed: ", "dim"),
        (_fmt_duration(stage_elapsed_s), "white"),
    )
    res = Text.assemble(
        ("RSS: ", "dim"),
        (f"{rss_mb:.0f} MB", "white"),
        ("  (peak: ", "dim"),
        (f"{peak_rss_mb:.0f} MB", "white"),
        (")  CPU: ", "dim"),
        (f"{cpu_pct:.1f}%", "white"),
    )
    body_items: List[Any] = [
        title,
        Text(""),
    ]
    if memray_active:
        body_items.append(
            Text("memray capture active (PODCAST_SCRAPER_MEMRAY_ACTIVE=1)", style="dim")
        )
        body_items.append(Text(""))
    body_items.extend(
        [
            head,
            Text(""),
            stage_line,
            st_el,
            Text(""),
            res,
            Text(""),
        ]
    )
    if history_rows:
        body_items.append(Text("Stage history (completed)", style="bold dim"))
        tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        tbl.add_column("Stage", style="cyan", no_wrap=True)
        tbl.add_column("Wall", justify="right")
        tbl.add_column("Peak MB", justify="right")
        tbl.add_column("Avg CPU%", justify="right")
        for row in history_rows:
            tbl.add_row(*row)
        body_items.append(tbl)
    body = Group(*body_items)
    return Panel(body, border_style="blue", padding=(0, 1))


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def build_summary_table(
    rows: Sequence[HistoryRow], total_wall: float, peak_mb: float, avg_cpu: float
) -> Table:
    """Final summary after the pipeline exits."""
    tbl = Table(
        title="Pipeline monitor summary",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    tbl.add_column("Stage", style="cyan")
    tbl.add_column("Wall (s)", justify="right")
    tbl.add_column("Peak RSS (MB)", justify="right")
    tbl.add_column("Avg CPU%", justify="right")
    for r in rows:
        tbl.add_row(*r)
    tbl.add_section()
    tbl.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_wall:.1f}[/bold]",
        f"[bold]{peak_mb:.0f}[/bold]",
        f"[bold]{avg_cpu:.1f}[/bold]",
    )
    return tbl
