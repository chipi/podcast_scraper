#!/usr/bin/env python3
"""Weekly autoresearch drift check — per-candidate, week-over-week.

Reads the two most recent ``data/autoresearch_baselines/autoresearch-*.json``
ledgers; for each candidate present in BOTH weeks, applies thresholds from
``drift_thresholds.yaml`` (when present — the file is deleted at method
boundaries to reset calibration, in which case this script emits a
no_thresholds report and exits 0) to the per-candidate metrics. Emits a structured
report (JSON) on stdout + to ``--output``. The workflow inspects the report
to open/update/close the weekly drift issue.

Mirrors scripts/baselines/check_drift.py's plan-then-act pattern (no
side-effects here; the workflow does the issue management).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_two_ledgers(ledger_dir: Path) -> tuple[Path | None, Path | None]:
    files = sorted(ledger_dir.glob("autoresearch-*.json"))
    if not files:
        return None, None
    if len(files) == 1:
        return files[-1], None
    return files[-1], files[-2]


def _by_model(cohort: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["model"]: row for row in cohort if "model" in row}


# Primary phase for drift comparison. Points at the least vendor-biased
# judge in the cohort (Qwen3-30B-A3B — Alibaba, disjoint from every
# candidate family AND from the silver Anthropic reference). Post-2026-W27
# refactor: Ollama used to run a scalar judge here but was retired
# because the available Ollama judges (gemma3:27b, mistral-small:24b)
# are same-tier as the candidates — peer review, not authoritative.
# If the cohort ever picks up Alibaba candidates, switch primary here.
_PRIMARY_PHASE = "judge_qwen_next_scalar"


def _primary_scores_and_latency(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (scores, latency_ms) for the primary drift-check phase.

    v2 ledgers (schema_version=2) key scores by phase name:
        row.scores_by_phase[phase].scores.final / ...
        row.scores_by_phase[phase].latency_ms.p95

    v1 ledgers (pre-refactor, flat ``scores`` at the row root) are still
    read for backward-compat while the last of them expire from the
    committed history; new ledgers are v2-only. Pre-refactor v2 ledgers
    that used ``ollama`` as the primary phase name will produce an
    empty dict here — they were tainted anyway (same-tier judges) and
    the operator opted to reset drift thresholds on the boundary.
    """
    if "scores" in row:
        return row.get("scores") or {}, row.get("latency_ms") or {}
    phase = (row.get("scores_by_phase") or {}).get(_PRIMARY_PHASE) or {}
    return phase.get("scores") or {}, phase.get("latency_ms") or {}


def _drop_ratio(prev: float, this: float) -> float | None:
    if prev is None or this is None or prev <= 0:
        return None
    return (prev - this) / prev


def _growth_ratio(prev: float, this: float) -> float | None:
    if prev is None or this is None or prev <= 0:
        return None
    return this / prev


def check_candidate(
    model: str, this_row: dict[str, Any], prev_row: dict[str, Any], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply per-metric thresholds. Returns a list of breach dicts."""
    breaches: list[dict[str, Any]] = []

    if this_row.get("status") != "ok":
        breaches.append(
            {
                "category": "candidate_failed",
                "model": model,
                "severity": "high",
                "message": (
                    f"Candidate `{model}` did not produce a successful score "
                    f"this week (status: {this_row.get('status')}). "
                    f"Check the workflow logs."
                ),
                "this_value": this_row.get("status"),
                "prev_value": prev_row.get("status"),
            }
        )
        # Skip metric comparisons on failed candidates.
        return breaches

    # v1 ledger: row.scores / row.latency_ms at the top.
    # v2 ledger (multi-phase): row.scores_by_phase[phase].{scores,latency_ms}.
    # For drift, compare the primary judge phase (judge_qwen — least vendor-biased,
    # see _PRIMARY_PHASE). v1 ledger rows fall through to the top-of-row scores branch above.
    this_scores, this_perf = _primary_scores_and_latency(this_row)
    prev_scores, prev_perf = _primary_scores_and_latency(prev_row)

    # Headline scalar.
    fs = cfg.get("final_score") or {}
    if (md := fs.get("max_drop_ratio")) is not None:
        drop = _drop_ratio(prev_scores.get("final"), this_scores.get("final"))
        if drop is not None and drop > md:
            breaches.append(
                {
                    "category": "final_score",
                    "model": model,
                    "severity": "high",
                    "message": (
                        f"`{model}` final score dropped from "
                        f"{prev_scores.get('final'):.4f} to "
                        f"{this_scores.get('final'):.4f} "
                        f"({drop * 100:.1f}% drop; threshold {md * 100:.0f}%)."
                    ),
                    "this_value": this_scores.get("final"),
                    "prev_value": prev_scores.get("final"),
                    "drop_ratio": round(drop, 4),
                }
            )

    # ROUGE-L F1.
    rl = cfg.get("rouge_l_f1") or {}
    if (md := rl.get("max_drop_ratio")) is not None:
        drop = _drop_ratio(prev_scores.get("rougeL_f1"), this_scores.get("rougeL_f1"))
        if drop is not None and drop > md:
            breaches.append(
                {
                    "category": "rouge_l_f1",
                    "model": model,
                    "severity": "medium",
                    "message": (
                        f"`{model}` rougeL_f1 dropped from "
                        f"{prev_scores.get('rougeL_f1'):.4f} to "
                        f"{this_scores.get('rougeL_f1'):.4f} "
                        f"({drop * 100:.1f}% drop; threshold {md * 100:.0f}%)."
                    ),
                    "this_value": this_scores.get("rougeL_f1"),
                    "prev_value": prev_scores.get("rougeL_f1"),
                    "drop_ratio": round(drop, 4),
                }
            )

    # judge_mean drop.
    jm = cfg.get("judge_mean") or {}
    if (md := jm.get("max_drop_ratio")) is not None:
        drop = _drop_ratio(prev_scores.get("judge_mean"), this_scores.get("judge_mean"))
        if drop is not None and drop > md:
            breaches.append(
                {
                    "category": "judge_mean",
                    "model": model,
                    "severity": "medium",
                    "message": (
                        f"`{model}` judge_mean dropped from "
                        f"{prev_scores.get('judge_mean'):.4f} to "
                        f"{this_scores.get('judge_mean'):.4f} "
                        f"({drop * 100:.1f}% drop; threshold {md * 100:.0f}%)."
                    ),
                    "this_value": this_scores.get("judge_mean"),
                    "prev_value": prev_scores.get("judge_mean"),
                    "drop_ratio": round(drop, 4),
                }
            )

    # p95 latency growth.
    lat = cfg.get("p95_latency_ms") or {}
    if (mr := lat.get("max_growth_ratio")) is not None:
        ratio = _growth_ratio(prev_perf.get("p95"), this_perf.get("p95"))
        if ratio is not None and ratio > mr:
            breaches.append(
                {
                    "category": "p95_latency_ms",
                    "model": model,
                    "severity": "medium",
                    "message": (
                        f"`{model}` p95 latency grew "
                        f"{prev_perf.get('p95'):.0f} ms → "
                        f"{this_perf.get('p95'):.0f} ms "
                        f"({(ratio - 1) * 100:.1f}% slower; threshold "
                        f"{(mr - 1) * 100:.0f}%)."
                    ),
                    "this_value": this_perf.get("p95"),
                    "prev_value": prev_perf.get("p95"),
                    "growth_ratio": round(ratio, 3),
                }
            )

    # Contested rate increase.
    cr = cfg.get("contested_rate") or {}
    if (mi := cr.get("max_absolute_increase")) is not None:
        prev_cr = prev_scores.get("contested_rate") or 0.0
        this_cr = this_scores.get("contested_rate") or 0.0
        if this_cr - prev_cr > mi:
            breaches.append(
                {
                    "category": "contested_rate",
                    "model": model,
                    "severity": "medium",
                    "message": (
                        f"`{model}` contested rate rose from "
                        f"{prev_cr * 100:.0f}% to {this_cr * 100:.0f}% "
                        f"(+{(this_cr - prev_cr) * 100:.0f}pp; "
                        f"threshold +{mi * 100:.0f}pp). Judges disagree more "
                        f"often than before — prompt may be producing "
                        f"inconsistent outputs."
                    ),
                    "this_value": this_cr,
                    "prev_value": prev_cr,
                }
            )

    # Quality gates went dirty.
    if (cfg.get("quality_gates") or {}).get("fail_on_gates_not_clean") and not (
        this_row.get("intrinsic") or {}
    ).get("gates_clean", True):
        breaches.append(
            {
                "category": "quality_gates",
                "model": model,
                "severity": "high",
                "message": (
                    f"`{model}` intrinsic quality gates are no longer clean "
                    f"(leak rates / truncation / failed episodes). Check "
                    f"the run's metrics.json."
                ),
                "this_value": (this_row.get("intrinsic") or {}),
                "prev_value": (prev_row.get("intrinsic") or {}),
            }
        )

    return breaches


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly autoresearch drift check.")
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=Path("data/autoresearch_baselines"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("data/autoresearch_baselines/drift_thresholds.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/autoresearch-drift-report.json"),
    )
    args = parser.parse_args()

    # Thresholds file may be absent on purpose — operator resets it after
    # methodology changes (e.g. mode/judge swap) that invalidate the
    # calibrated per-candidate drop thresholds. Skip drift check cleanly
    # when absent so the nightly workflow does not fail; drift will
    # re-arm once the file is reintroduced.
    if not args.thresholds.is_file():
        report_missing = {
            "status": "no_thresholds",
            "informational": [
                f"Thresholds file not found at {args.thresholds}. "
                "Drift check SKIPPED — bootstrap needed. "
                "Re-add data/autoresearch_baselines/drift_thresholds.yaml "
                "to re-enable per-candidate breach detection."
            ],
            "breaches": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report_missing, indent=2), encoding="utf-8")
        print(
            f"Drift check skipped: no thresholds file at {args.thresholds}; "
            f"report written to {args.output}",
            file=sys.stderr,
        )
        return 0

    thresholds = load_yaml(args.thresholds)
    this_path, prev_path = latest_two_ledgers(args.ledger_dir)

    report: dict[str, Any]
    if this_path is None:
        report = {"status": "no_ledger", "breaches": []}
    elif prev_path is None:
        this = load_json(this_path)
        report = {
            "status": "first_week",
            "this_path": str(this_path),
            "this_week_id": this.get("week_id"),
            "breaches": [],
            "informational": [
                f"First autoresearch ledger in the series ({this.get('week_id')}). "
                f"Next week will produce the first drift diff."
            ],
        }
    else:
        this = load_json(this_path)
        prev = load_json(prev_path)
        this_by_model = _by_model(this.get("cohort") or [])
        prev_by_model = _by_model(prev.get("cohort") or [])

        breaches: list[dict[str, Any]] = []
        informational: list[str] = []
        for model, this_row in this_by_model.items():
            prev_row = prev_by_model.get(model)
            if prev_row is None:
                informational.append(
                    f"New candidate `{model}` in cohort — no prev-week comparison."
                )
                continue
            breaches.extend(check_candidate(model, this_row, prev_row, thresholds))

        for model in prev_by_model:
            if model not in this_by_model:
                informational.append(
                    f"Candidate `{model}` was in last week's cohort but not this "
                    f"week — intentional removal?"
                )

        report = {
            "status": "breaches" if breaches else "no_breaches",
            "this_path": str(this_path),
            "this_week_id": this.get("week_id"),
            "prev_path": str(prev_path),
            "prev_week_id": prev.get("week_id"),
            "breaches": breaches,
            "informational": informational,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
