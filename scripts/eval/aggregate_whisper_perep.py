#!/usr/bin/env python3
"""Aggregate per-episode metrics.json files into a single sweep summary.

Used by the Tailscale-hang workaround pattern in #929 / #963 contention re-test:
each episode is run as an independent harness invocation, then the per-episode
metrics.json files are stitched together into one sweep-level metrics.json
with the same `schema: "metrics_whisper_dgx_vs_cloud_v1"` shape that the
single-shot harness emits.

Usage:
    python scripts/eval/aggregate_whisper_perep.py <perep_dir> <output_path>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: aggregate_whisper_perep.py <perep_dir> <output_path>", file=sys.stderr)
        return 2

    perep_dir = Path(sys.argv[1])
    output = Path(sys.argv[2])

    if not perep_dir.exists():
        print(f"perep_dir does not exist: {perep_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    backend = None
    model = None
    for ep_dir in sorted(perep_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        metrics_path = ep_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  SKIP {ep_dir.name}: no metrics.json (likely timeout)", file=sys.stderr)
            continue
        data = json.loads(metrics_path.read_text())
        for row in data.get("rows", []):
            rows.append(row)
            backend = backend or row.get("backend")
            model = model or row.get("model")

    if not rows:
        print("no rows collected", file=sys.stderr)
        return 1

    def _clean(r: dict) -> bool:
        err = r.get("error")
        return err == "" or err is None

    clean_rows = [r for r in rows if _clean(r)]
    errored_rows = [r for r in rows if not _clean(r)]
    wers = [r["wer"] for r in clean_rows]
    elapseds = [r["elapsed_s"] for r in clean_rows]
    rts = [r["realtime_multiple"] for r in clean_rows if r.get("realtime_multiple") is not None]
    total_audio = sum(r.get("audio_seconds", 0.0) for r in clean_rows)

    summary = [
        {
            "backend": backend,
            "model": model,
            "episodes": len(rows),
            "clean_episodes": len(clean_rows),
            "errored_episodes": len(errored_rows),
            "mean_wer": round(statistics.fmean(wers), 4) if wers else None,
            "max_wer": round(max(wers), 4) if wers else None,
            "min_wer": round(min(wers), 4) if wers else None,
            "mean_elapsed_s": round(statistics.fmean(elapseds), 2) if elapseds else None,
            "stdev_elapsed_s": (
                round(statistics.stdev(elapseds), 2) if len(elapseds) >= 2 else None
            ),
            "mean_realtime_multiple": round(statistics.fmean(rts), 2) if rts else None,
            "total_audio_seconds": round(total_audio, 1),
            "total_cost_usd": 0.0,
        }
    ]

    note = (
        "aggregated from per-episode invocations (Tailscale-hang "
        "workaround) for #963 contention re-test"
    )
    out = {
        "schema": "metrics_whisper_dgx_vs_cloud_v1",
        "backend": backend,
        "note": note,
        "summary": summary,
        "rows": rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {output} ({len(rows)} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
