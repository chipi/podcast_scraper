#!/usr/bin/env python3
"""Compare a fresh experiment / benchmark run against the most recent
checked-in baseline under ``data/baselines/``.

Emits Markdown for the GitHub job summary. Per operator directive
(2026-06-25): regressions are **soft warnings**, never hard failures.
Exits 0 always — the caller decides what to do with the output.

Used by ``release.yml`` so the release-notes draft has a quick "what
changed since last week" block. Designed to be safe when no baseline
exists (e.g. first run) — emits an "ℹ️ no prior baseline" block in
that case.

Usage::

    .venv/bin/python scripts/baselines/compare_against_baseline.py \\
        --experiment-run-dir data/eval/runs/<run_id> \\
        --benchmark-run-dir data/eval/runs/<run_id> \\
        --baselines-dir data/baselines \\
        [--output release-comparison.md]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Metric-relative-difference thresholds for "warn" vs "info". Negative
# delta (worse) above warn_pct triggers ⚠️; below info_pct is silent ✓.
_WARN_PCT = 0.02  # 2% regression triggers warning
_INFO_PCT = 0.005  # 0.5% jitter is normal


def _latest_baseline(baselines_dir: Path) -> Optional[Path]:
    if not baselines_dir.is_dir():
        return None
    candidates = sorted(
        baselines_dir.glob("baseline-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_metrics(run_dir: Optional[Path]) -> Dict[str, Any]:
    if run_dir is None or not run_dir.is_dir():
        return {}
    p = run_dir / "metrics.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _flatten_numeric(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    """Recursively pull numeric leaves into ``out`` under dotted keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numeric(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)


def _diff(new: Dict[str, float], old: Dict[str, float]) -> List[Tuple[str, float, float, float]]:
    """Return list of (key, old, new, pct_delta) for keys in both."""
    rows: List[Tuple[str, float, float, float]] = []
    for k in sorted(set(new) & set(old)):
        a = old[k]
        b = new[k]
        if a == 0:
            pct = 0.0 if b == 0 else float("inf")
        else:
            pct = (b - a) / abs(a)
        rows.append((k, a, b, pct))
    return rows


def _classify(pct: float) -> str:
    if pct >= _WARN_PCT:
        return "✅"  # improvement
    if pct <= -_WARN_PCT:
        return "⚠️"
    if abs(pct) <= _INFO_PCT:
        return "·"  # noise
    return "ℹ️"


def render_markdown(
    experiment_new: Dict[str, Any],
    benchmark_new: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("## Release eval comparison vs last baseline\n")
    if baseline is None:
        lines.append(
            "ℹ️ No prior baseline found under `data/baselines/`. "
            "Treat this run as the first data point.\n"
        )
    else:
        lines.append(
            f"- **Baseline week:** `{baseline.get('week_id')}` "
            f"(captured `{baseline.get('captured_at')}`)\n"
            f"- **This run:** `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n"
        )

    def _section(title: str, new_metrics: Dict[str, Any], baseline_run: Dict[str, Any]) -> None:
        lines.append(f"\n### {title}\n")
        if not new_metrics:
            lines.append("_no metrics.json in this run_\n")
            return
        old_metrics = baseline_run.get("metrics") or {} if baseline_run else {}
        new_flat: Dict[str, float] = {}
        old_flat: Dict[str, float] = {}
        _flatten_numeric("", new_metrics, new_flat)
        _flatten_numeric("", old_metrics, old_flat)
        rows = _diff(new_flat, old_flat)
        if not rows:
            # No overlap → just dump the new numbers.
            for k, v in sorted(new_flat.items())[:20]:
                lines.append(f"- `{k}`: {v:.4f}\n")
            return
        lines.append("| metric | baseline | new | Δ% | |\n|---|---:|---:|---:|:--:|\n")
        for k, a, b, pct in rows[:30]:
            mark = _classify(pct)
            lines.append(f"| `{k}` | {a:.4f} | {b:.4f} | " f"{pct*100:+.2f}% | {mark} |\n")

    bl_exp = (baseline or {}).get("experiment_run") or {}
    bl_bm = (baseline or {}).get("benchmark_run") or {}
    _section("AI experiment pipeline", experiment_new, bl_exp)
    _section("ML benchmark", benchmark_new, bl_bm)

    lines.append(
        "\n---\n"
        f"_Soft warnings only ({int(_WARN_PCT*100)}% regression threshold). "
        "No metric blocks release per operator directive 2026-06-25._\n"
    )
    return "".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-run-dir", type=Path, default=None)
    p.add_argument("--benchmark-run-dir", type=Path, default=None)
    p.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "baselines",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    bl_path = _latest_baseline(args.baselines_dir)
    baseline = None
    if bl_path is not None:
        try:
            baseline = json.loads(bl_path.read_text(encoding="utf-8"))
        except Exception:
            baseline = None

    md = render_markdown(
        _load_metrics(args.experiment_run_dir),
        _load_metrics(args.benchmark_run_dir),
        baseline,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
