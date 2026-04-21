#!/usr/bin/env python3
"""Compare two frozen performance profiles (RFC-064, Issue #510)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:  # pragma: no cover
    raise SystemExit("rich is required for profile diff. Install: pip install -e .") from exc

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RULES = _ROOT / "data" / "profiles" / "regression_rules.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _load_rules(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def _pct_delta(before: float, after: float) -> Optional[float]:
    if before == 0:
        return None
    return round((after - before) / before * 100.0, 2)


def _collect_rows(
    left: Dict[str, Any],
    right: Dict[str, Any],
    rules: Optional[Dict[str, Any]],
) -> List[Tuple[str, str, float, float, Optional[float], str]]:
    """Rows: stage, metric, v_from, v_to, delta_pct, flag."""
    rows: List[Tuple[str, str, float, float, Optional[float], str]] = []
    stages_l = left.get("stages") or {}
    stages_r = right.get("stages") or {}
    all_stages = sorted(set(stages_l.keys()) | set(stages_r.keys()))
    metrics = ("wall_time_s", "peak_rss_mb", "avg_cpu_pct")

    def _threshold_note(
        stage: str,
        metric: str,
        delta_pct: Optional[float],
    ) -> str:
        if delta_pct is None or rules is None:
            return ""
        thr = rules.get("thresholds", {}).get(metric, {})
        if isinstance(thr, dict):
            max_pct = thr.get("max_delta_pct")
            if isinstance(max_pct, (int, float)) and abs(delta_pct) > float(max_pct):
                return " (!)"
        st = rules.get("stages", {}).get(stage, {}).get(metric, {})
        if isinstance(st, dict):
            max_pct = st.get("max_delta_pct")
            if isinstance(max_pct, (int, float)) and abs(delta_pct) > float(max_pct):
                return " (!)"
        return ""

    for st in all_stages:
        sl = stages_l.get(st) or {}
        sr = stages_r.get(st) or {}
        for m in metrics:
            vl = float(sl.get(m, 0) or 0)
            vr = float(sr.get(m, 0) or 0)
            d = _pct_delta(vl, vr) if vl or vr else None
            flag = _threshold_note(st, m, d)
            rows.append((st, m, vl, vr, d, flag))

    tl = left.get("totals") or {}
    tr = right.get("totals") or {}
    for m in ("peak_rss_mb", "wall_time_s", "avg_wall_time_per_episode_s"):
        vl = float(tl.get(m, 0) or 0)
        vr = float(tr.get(m, 0) or 0)
        d = _pct_delta(vl, vr) if vl or vr else None
        flag = _threshold_note("totals", m, d)
        rows.append(("totals", m, vl, vr, d, flag))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two frozen performance profiles.")
    parser.add_argument("from_path", type=Path, help="Left profile YAML")
    parser.add_argument("to_path", type=Path, help="Right profile YAML")
    parser.add_argument(
        "--rules",
        type=Path,
        default=_DEFAULT_RULES,
        help="Optional regression_rules.yaml (annotations when thresholds exceeded)",
    )
    args = parser.parse_args()

    if not args.from_path.is_file() or not args.to_path.is_file():
        print("Both profile paths must exist.", file=sys.stderr)
        sys.exit(1)

    left = _load_yaml(args.from_path)
    right = _load_yaml(args.to_path)
    rules = _load_rules(args.rules)

    rel = left.get("release", "?")
    rer = right.get("release", "?")
    ds = left.get("dataset_id", "?")
    host = (left.get("environment") or {}).get("hostname", "?")

    console = Console()
    console.print(f"Profile diff: {rel} -> {rer} ({ds}, {host})")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Stage")
    table.add_column("Metric")
    table.add_column("From", justify="right")
    table.add_column("To", justify="right")
    table.add_column("Delta", justify="right")

    for stage, metric, vl, vr, dpct, flag in _collect_rows(left, right, rules):
        d_str = "" if dpct is None else f"{dpct:+.1f}%"
        table.add_row(
            stage,
            metric,
            f"{vl:g}",
            f"{vr:g}",
            f"{d_str}{flag}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
