#!/usr/bin/env python3
"""Validate a metrics directory produced by CI (or fetch_ci_metrics_artifacts.sh).

Checks latest-ci.json / latest-nightly.json structure, optional history JSONL,
and recomputes flaky count from embedded slowest/flaky lists when present.

Usage:
  python scripts/dashboard/validate_metrics_bundle.py artifacts/ci-metrics-runs/run-12345
  python scripts/dashboard/validate_metrics_bundle.py metrics/

Exit code 0 if JSON parses and required keys exist; 1 on error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DASH = Path(__file__).resolve().parent
if str(_DASH) not in sys.path:
    sys.path.insert(0, str(_DASH))

from metrics_jsonl import load_metrics_history  # noqa: E402


def _find_latest_bundle(root: Path) -> Tuple[Path, str]:
    """Return (path_to_latest_json, kind) where kind is ci or nightly."""
    root = root.resolve()
    candidates = [
        (root / "latest-ci.json", "ci"),
        (root / "latest-nightly.json", "nightly"),
        (root / "metrics" / "latest-ci.json", "ci"),
        (root / "metrics" / "latest-nightly.json", "nightly"),
    ]
    for path, kind in candidates:
        if path.is_file():
            return path, kind
    raise FileNotFoundError(
        f"No latest-ci.json or latest-nightly.json under {root} (or metrics/ subdir)"
    )


def _find_history(root: Path, kind: str) -> Optional[Path]:
    root = root.resolve()
    name = "history-ci.jsonl" if kind == "ci" else "history-nightly.jsonl"
    for base in (root, root / "metrics"):
        p = base / name
        if p.is_file():
            return p
    return None


def _summarize_history(path: Path) -> Tuple[int, List[str]]:
    """Return (record_count, errors) using same loader as CI (JSONL + legacy recovery)."""
    errors: List[str] = []
    if not path.is_file():
        return 0, errors
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return 0, errors
    records = load_metrics_history(path)
    if not records:
        errors.append("non-empty file but no JSON objects parsed (corrupt?)")
    return len(records), errors


def _summarize_latest(data: Dict[str, Any]) -> None:
    m = data.get("metrics") or {}
    th = m.get("test_health") or {}
    rt = m.get("runtime") or {}
    cov = m.get("coverage") or {}
    slow = m.get("slowest_tests") or []
    flaky_tests = th.get("flaky_tests") or []

    print("--- latest bundle ---")
    print(f"  timestamp:     {data.get('timestamp')}")
    print(f"  commit:        {data.get('commit')}")
    print(f"  branch:        {data.get('branch')}")
    print(f"  workflow_run:  {data.get('workflow_run')}")
    print(f"  tests total:   {th.get('total')}")
    print(f"  passed/fail:   {th.get('passed')} / {th.get('failed')}")
    print(f"  flaky (count): {th.get('flaky')} (listed: {len(flaky_tests)})")
    print(f"  pytest wall s: {rt.get('total')}")
    print(f"  coverage %:    {cov.get('overall')}")
    print(f"  slowest_tests: {len(slow)} entries")
    if slow:
        print("  top 3 slowest:")
        for row in slow[:3]:
            print(f"    - {row.get('duration')}s  {row.get('name')}")
    cx = m.get("complexity") or {}
    print(
        "  complexity:    cc_avg=%s  mi_avg=%s  docstrings=%s%%  dead=%s  spell=%s"
        % (
            cx.get("cyclomatic_complexity"),
            cx.get("maintainability_index"),
            cx.get("docstring_coverage"),
            cx.get("dead_code_count"),
            cx.get("spelling_errors_count"),
        )
    )
    alerts = data.get("alerts") or []
    print(f"  alerts:        {len(alerts)}")


def _diagnose_dashboard_gaps(data: Dict[str, Any]) -> None:
    """Print hints when JSON matches known stale-CI patterns (not validation failures)."""
    m = data.get("metrics") or {}
    th = m.get("test_health") or {}
    total = int(th.get("total") or 0)
    slow = m.get("slowest_tests") or []
    cx = m.get("complexity") or {}
    hints: List[str] = []

    if total > 100 and len(slow) == 0:
        hints.append(
            "slowest_tests empty — stale metrics before CI emitted junit*.xml + "
            "generate_metrics JUnit fallback, or JSON timings still missing on workers; "
            "re-fetch after a main run with the updated workflow."
        )
    cc = cx.get("cyclomatic_complexity")
    if cc == 0 and total > 100:
        hints.append(
            "complexity/docstrings/dead/spell zeros — radon or capture_quality not in that metrics step."
        )
    if th.get("flaky") == 0 and total > 100:
        hints.append(
            "flaky=0 — may be real; stale JSON may have read only merged pytest.json while "
            "shards kept outcome=rerun + call.passed; re-fetch after generate_metrics aggregates "
            "all pytest-*.json + pytest.json."
        )

    if hints:
        print("--- hints (dashboard gaps) ---")
        for h in hints:
            print(f"  • {h}")


def main() -> int:
    logging.getLogger("metrics_jsonl").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "bundle_dir",
        type=Path,
        help="Directory from gh run download (contains latest-*.json) or repo metrics/",
    )
    args = p.parse_args()

    try:
        latest_path, kind = _find_latest_bundle(args.bundle_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        data: Dict[str, Any] = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {latest_path}: {e}", file=sys.stderr)
        return 1

    if "metrics" not in data:
        print(f"error: missing top-level 'metrics' in {latest_path}", file=sys.stderr)
        return 1

    _summarize_latest(data)
    _diagnose_dashboard_gaps(data)

    hist = _find_history(args.bundle_dir, kind)
    if hist:
        n, errs = _summarize_history(hist)
        print(f"--- {hist.name} (CI-compatible parse) ---")
        print(f"  records: {n}")
        if errs:
            for line in errs:
                print(f"  error: {line}")
            return 1
    else:
        print("--- history JSONL ---")
        print("  (not found next to latest JSON — optional)")

    # If both CI and nightly history files exist, summarize the other one too
    other = "history-nightly.jsonl" if kind == "ci" else "history-ci.jsonl"
    for base in (args.bundle_dir.resolve(), args.bundle_dir.resolve() / "metrics"):
        alt = base / other
        if alt.is_file() and (hist is None or alt.resolve() != hist.resolve()):
            n2, errs2 = _summarize_history(alt)
            print(f"--- {alt.name} (CI-compatible parse) ---")
            print(f"  records: {n2}")
            if errs2:
                for line in errs2:
                    print(f"  error: {line}")
                return 1
            break

    th = (data.get("metrics") or {}).get("test_health") or {}
    flaky_n = th.get("flaky")
    flaky_list = th.get("flaky_tests") or []
    if isinstance(flaky_n, int) and len(flaky_list) != flaky_n:
        print(
            f"warning: flaky count ({flaky_n}) != len(flaky_tests) ({len(flaky_list)})",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
