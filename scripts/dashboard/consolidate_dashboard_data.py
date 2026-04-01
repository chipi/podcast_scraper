#!/usr/bin/env python3
"""Build dashboard-data.json: one JSON file with CI + nightly latest + history arrays.

The unified dashboard prefers this file (single fetch). Individual latest-*.json and
history-*.jsonl remain for workflows, append scripts, and legacy dashboard fallback.

See docs/wip/METRICS_DOCS_AND_DASHBOARD_V2.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_DASH = Path(__file__).resolve().parent
_sdash = str(_DASH)
if _sdash not in sys.path:
    sys.path.insert(0, _sdash)

from metrics_jsonl import load_metrics_history  # noqa: E402

NO_CI_ALERT: Dict[str, Any] = {
    "severity": "info",
    "metric": "ci",
    "message": "No CI snapshot in this bundle (missing latest-ci.json).",
}
NO_NIGHTLY_ALERT: Dict[str, Any] = {
    "severity": "info",
    "metric": "nightly",
    "message": "No nightly snapshot in this bundle (missing latest-nightly.json).",
}


def _physical_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if line.strip())


def _check_pretty_jsonl(
    path: Path, records: List[Dict[str, Any]], strict: bool, warnings: List[str]
) -> None:
    """Warn or exit if a .jsonl file looks like one pretty-printed JSON blob."""
    n_phys = _physical_nonempty_lines(path)
    if n_phys >= 8 and len(records) <= 1:
        msg = (
            f"{path}: parsed {len(records)} history record(s) but file has {n_phys} non-empty "
            "lines — likely pretty-printed JSON in a .jsonl file; use one compact JSON object "
            "per line (see append_metrics_history_line.py)."
        )
        warnings.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
        if strict:
            raise SystemExit(2)


def _load_latest(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not load {path}: {e}", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def build_bundle(input_dir: Path, strict: bool) -> Dict[str, Any]:
    """Return dashboard bundle dict; mutates nothing on disk."""
    warnings: List[str] = []
    input_dir = input_dir.resolve()

    ci_latest_p = input_dir / "latest-ci.json"
    ci_hist_p = input_dir / "history-ci.jsonl"
    ny_latest_p = input_dir / "latest-nightly.json"
    ny_hist_p = input_dir / "history-nightly.jsonl"

    ci_latest = _load_latest(ci_latest_p)
    if ci_latest is None:
        ci_latest = {
            "timestamp": None,
            "commit": None,
            "branch": None,
            "metrics": {},
            "trends": {},
            "alerts": [dict(NO_CI_ALERT)],
        }

    ci_hist = load_metrics_history(ci_hist_p) if ci_hist_p.exists() else []
    _check_pretty_jsonl(ci_hist_p, ci_hist, strict, warnings)

    ny_latest = _load_latest(ny_latest_p)
    if ny_latest is None:
        ny_latest = {
            "timestamp": None,
            "commit": None,
            "branch": None,
            "metrics": {},
            "trends": {},
            "alerts": [dict(NO_NIGHTLY_ALERT)],
        }

    ny_hist = load_metrics_history(ny_hist_p) if ny_hist_p.exists() else []
    _check_pretty_jsonl(ny_hist_p, ny_hist, strict, warnings)

    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "version": 1,
        "generated_at": generated,
        "ci": {"latest": ci_latest, "history": ci_hist},
        "nightly": {"latest": ny_latest, "history": ny_hist},
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write dashboard-data.json for unified dashboard.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing latest-*.json and history-*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input-dir>/dashboard-data.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if a history file looks like pretty-printed JSON, not JSONL.",
    )
    args = parser.parse_args()
    out = args.output if args.output is not None else args.input_dir / "dashboard-data.json"

    bundle = build_bundle(args.input_dir, args.strict)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ Wrote {out}")


if __name__ == "__main__":
    main()
