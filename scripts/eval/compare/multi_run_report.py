#!/usr/bin/env python3
"""Generate a multi-run comparison report (baseline + N runs) with vs-reference metrics.

Part of the evaluation framework. Reads metrics.json from baseline and/or run
directories and writes a markdown table so you can compare quality (ROUGE, BLEU,
embedding, coverage, WER) and latency across runs. Supports any number of runs.

Usage:
    # Default: baseline + tier1 + tier2, vs silver_gpt4o_smoke_v1
    make report-multi-run

    # Custom: one baseline and several runs
    make report-multi-run BASELINE_ID=baseline_ml_prod_authority_smoke_v1 \\
        RUN_IDS=run_a,run_b,run_c REFERENCE_ID=silver_gpt4o_smoke_v1

    # Runs only (no baseline)
    make report-multi-run RUN_IDS=id1,id2,id3 \\
        REFERENCE_ID=silver_gpt4o_smoke_v1 OUTPUT=docs/wip/my_comparison.md

    # With custom labels and title
    make report-multi-run BASELINE_ID=... RUN_IDS=... REFERENCE_ID=... \\
        TITLE="Smoke comparison" LABELS="Prod,Tier1,Tier2"

Direct script usage:
    python scripts/eval/multi_run_report.py \\
        --reference-id silver_gpt4o_smoke_v1 \\
        [--baseline-id baseline_id] [--run-ids run1,run2,...] \\
        [--output path] [--title "Title"] [--labels "A,B,C"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _vs_reference(metrics: dict, reference_id: str) -> dict | None:
    return (metrics.get("vs_reference") or {}).get(reference_id)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.1f}s"


def _resolve_metrics_path(entry_id: str, baselines_dir: Path, runs_dir: Path) -> tuple[Path, str]:
    """Return (path to metrics.json, 'baseline'|'run'). Raises FileNotFoundError."""
    baseline_path = baselines_dir / entry_id / "metrics.json"
    if baseline_path.exists():
        return baseline_path, "baseline"
    run_path = runs_dir / entry_id / "metrics.json"
    if run_path.exists():
        return run_path, "run"
    raise FileNotFoundError(
        f"Metrics not found for '{entry_id}'. " f"Checked: {baseline_path}, {run_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-run comparison report (baseline + N runs) with vs-reference metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference-id",
        type=str,
        required=True,
        help="Reference ID used for vs_reference metrics (e.g. silver_gpt4o_smoke_v1)",
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        default=None,
        help="Optional baseline ID; included as first row (from --baselines-dir)",
    )
    parser.add_argument(
        "--run-ids",
        type=str,
        default=None,
        help="Comma-separated run IDs (from --runs-dir). Need --baseline-id and/or --run-ids.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/wip/multi_run_comparison.md"),
        help="Output markdown path (default: docs/wip/multi_run_comparison.md)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path("data/eval/baselines"),
        help="Baselines directory (default: data/eval/baselines)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/eval/runs"),
        help="Runs directory (default: data/eval/runs)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Multi-Run Comparison",
        help="Report title (default: Multi-Run Comparison)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Row labels (baseline first, then runs). Default: use entry ID.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset ID for report subtitle (default: read from first metrics)",
    )
    args = parser.parse_args()

    if not args.baseline_id and not args.run_ids:
        parser.error("At least one of --baseline-id or --run-ids is required")

    # Build ordered list of (id, path, kind)
    entries: list[tuple[str, Path, str]] = []
    if args.baseline_id:
        path, kind = _resolve_metrics_path(args.baseline_id, args.baselines_dir, args.runs_dir)
        entries.append((args.baseline_id, path, kind))
    if args.run_ids:
        for run_id in (r.strip() for r in args.run_ids.split(",") if r.strip()):
            path, kind = _resolve_metrics_path(run_id, args.baselines_dir, args.runs_dir)
            entries.append((run_id, path, kind))

    labels_list: list[str] | None = None
    if args.labels:
        labels_list = [x.strip() for x in args.labels.split(",") if x.strip()]
        if len(labels_list) != len(entries):
            sys.exit(
                f"Error: --labels has {len(labels_list)} value(s) but there are "
                f"{len(entries)} row(s) (baseline + runs)."
            )

    metrics_list = [_load_metrics(p) for _, p, _ in entries]
    dataset_id = args.dataset_id
    if not dataset_id and metrics_list:
        dataset_id = (metrics_list[0].get("dataset_id")) or ""

    def row(label: str, m: dict) -> list[str]:
        vs = _vs_reference(m, args.reference_id)
        lat = (m.get("intrinsic") or {}).get("performance") or {}
        avg_ms = lat.get("avg_latency_ms")
        return [
            label,
            _fmt_ms(avg_ms),
            _fmt_pct(vs.get("rouge1_f1") if vs else None),
            _fmt_pct(vs.get("rouge2_f1") if vs else None),
            _fmt_pct(vs.get("rougeL_f1") if vs else None),
            _fmt_pct(vs.get("bleu") if vs else None),
            _fmt_pct(vs.get("embedding_cosine") if vs else None),
            _fmt_pct(vs.get("coverage_ratio") if vs else None),
            _fmt_pct(vs.get("wer") if vs else None),
        ]

    headers = [
        "Run",
        "Latency/ep",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "BLEU",
        "Embed",
        "Coverage",
        "WER",
    ]
    row_labels = labels_list if labels_list else [eid for eid, _, _ in entries]
    rows = [row(label, m) for label, m in zip(row_labels, metrics_list)]

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    subtitle = f"Reference: `{args.reference_id}`."
    if dataset_id:
        subtitle = f"Dataset: `{dataset_id}`. {subtitle}"
    lines = [
        f"# {args.title}",
        "",
        subtitle,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Higher ROUGE / BLEU / Embed / Coverage = closer to reference; lower WER = better.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
