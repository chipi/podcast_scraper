#!/usr/bin/env python3
"""Weekly baseline drift check — emits a structured breach report.

Reads the two most recent ``data/baselines/baseline-*.json`` snapshots
and compares this-week's run against prev-week's, gated by thresholds
loaded from ``data/baselines/drift_thresholds.yaml`` (tunable without
touching this script).

Output: JSON report on stdout AND ``--output`` file. Exit 0 always —
the workflow inspects the JSON to decide whether to open / update /
close the weekly drift issue. Mirrors infra-drift.yml's plan-then-act
pattern.
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


def latest_two_snapshots(baselines_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (this_week, prev_week) — both may be None on a cold start."""
    files = sorted(baselines_dir.glob("baseline-*.json"))
    if not files:
        return None, None
    if len(files) == 1:
        return files[-1], None
    return files[-1], files[-2]


def _get(d: dict, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def check_quality_gates(this: dict, prev: dict, cfg: dict) -> list[dict]:
    """A watched gate going from 0 → >0 is a breach. Failed episodes non-empty is a breach."""
    breaches: list[dict] = []
    this_gates = _get(this, "metrics", "intrinsic", "gates", default={}) or {}
    prev_gates = _get(prev, "metrics", "intrinsic", "gates", default={}) or {}
    this_warn = _get(this, "metrics", "intrinsic", "warnings", default={}) or {}
    prev_warn = _get(prev, "metrics", "intrinsic", "warnings", default={}) or {}

    for field in cfg.get("watch", []) or []:
        tv = float(this_gates.get(field, 0.0) or 0.0)
        pv = float(prev_gates.get(field, 0.0) or 0.0)
        if tv > 0.0 and pv == 0.0:
            breaches.append(
                {
                    "category": "quality_gates",
                    "signal": field,
                    "severity": "high",
                    "message": (
                        f"Gate `{field}` went from {pv} (clean) to {tv} this week. "
                        f"Expected to stay at zero. Either the corpus introduced "
                        f"problematic content, a model behaviour changed, or a "
                        f"prompt/post-processing regression slipped through."
                    ),
                    "this_value": tv,
                    "prev_value": pv,
                }
            )

    for field in cfg.get("watch_warnings", []) or []:
        tv = float(this_warn.get(field, 0.0) or 0.0)
        pv = float(prev_warn.get(field, 0.0) or 0.0)
        if tv > 0.0 and pv == 0.0:
            breaches.append(
                {
                    "category": "quality_gates",
                    "signal": f"warnings.{field}",
                    "severity": "medium",
                    "message": (
                        f"Warning gate `{field}` went from 0 → {tv}. Lower severity "
                        f"than hard gates, but worth checking."
                    ),
                    "this_value": tv,
                    "prev_value": pv,
                }
            )

    if cfg.get("fail_on_failed_episodes_non_empty"):
        failed = this_gates.get("failed_episodes") or []
        if failed:
            breaches.append(
                {
                    "category": "quality_gates",
                    "signal": "failed_episodes",
                    "severity": "high",
                    "message": (
                        f"`failed_episodes` is non-empty this week: {failed}. "
                        f"Episodes are dropping out of the run pipeline — check "
                        f"per-episode error logs in the run dir."
                    ),
                    "this_value": failed,
                    "prev_value": prev_gates.get("failed_episodes") or [],
                }
            )

    return breaches


def check_performance(this: dict, prev: dict, cfg: dict) -> list[dict]:
    breaches: list[dict] = []
    this_perf = _get(this, "metrics", "intrinsic", "performance", default={}) or {}
    prev_perf = _get(prev, "metrics", "intrinsic", "performance", default={}) or {}

    for field, ratio_key in (
        ("p95_latency_ms", "p95_latency_max_ratio"),
        ("median_latency_ms", "median_latency_max_ratio"),
    ):
        max_ratio = cfg.get(ratio_key)
        if max_ratio is None:
            continue
        tv = this_perf.get(field)
        pv = prev_perf.get(field)
        if not (isinstance(tv, (int, float)) and isinstance(pv, (int, float)) and pv > 0):
            continue
        ratio = tv / pv
        if ratio > max_ratio:
            breaches.append(
                {
                    "category": "performance",
                    "signal": field,
                    "severity": "high" if ratio > max_ratio * 1.5 else "medium",
                    "message": (
                        f"`{field}` went from {pv:.0f} ms to {tv:.0f} ms — "
                        f"{(ratio - 1) * 100:.1f}% slower than last week (threshold: "
                        f"{(max_ratio - 1) * 100:.0f}%). Investigate: ml provider "
                        f"warm-up cost, torch/transformers version pins, GH runner "
                        f"capacity (the workflow uses ubuntu-latest CPU)."
                    ),
                    "this_value": tv,
                    "prev_value": pv,
                    "ratio": round(ratio, 3),
                    "threshold_ratio": max_ratio,
                }
            )
    return breaches


def check_length(this: dict, prev: dict, cfg: dict) -> list[dict]:
    breaches: list[dict] = []
    this_len = _get(this, "metrics", "intrinsic", "length", default={}) or {}
    prev_len = _get(prev, "metrics", "intrinsic", "length", default={}) or {}

    max_drift = cfg.get("avg_tokens_max_drift_ratio")
    if max_drift is None:
        return breaches

    tv = this_len.get("avg_tokens")
    pv = prev_len.get("avg_tokens")
    if not (isinstance(tv, (int, float)) and isinstance(pv, (int, float)) and pv > 0):
        return breaches

    drift = abs(tv - pv) / pv
    if drift > max_drift:
        direction = "longer" if tv > pv else "shorter"
        breaches.append(
            {
                "category": "length",
                "signal": "avg_tokens",
                "severity": "medium",
                "message": (
                    f"`avg_tokens` moved from {pv:.0f} to {tv:.0f} — "
                    f"{drift * 100:.1f}% {direction} (threshold: {max_drift * 100:.0f}%). "
                    f"Either a prompt change, model swap, or dataset shift. "
                    f"Combined with truncation gate movement = likely truncation."
                ),
                "this_value": tv,
                "prev_value": pv,
                "drift_ratio": round(drift, 3),
                "threshold_ratio": max_drift,
            }
        )
    return breaches


def check_datasets(this_probe: dict, prev_probe: dict, cfg: dict) -> list[dict]:
    breaches: list[dict] = []
    if not cfg.get("episodes_must_not_decrease"):
        return breaches

    this_ds = this_probe.get("datasets") or {}
    prev_ds = prev_probe.get("datasets") or {}
    for ds_id, prev_entry in prev_ds.items():
        if ds_id not in this_ds:
            breaches.append(
                {
                    "category": "datasets",
                    "signal": f"dataset:{ds_id}",
                    "severity": "high",
                    "message": (
                        f"Dataset `{ds_id}` disappeared from the source tree "
                        f"between last week ({prev_entry.get('episodes')} episodes) "
                        f"and this week. If intentional, fine — confirm and close "
                        f"the issue. Otherwise: data loss."
                    ),
                    "this_value": None,
                    "prev_value": prev_entry,
                }
            )
            continue
        prev_n = int(prev_entry.get("episodes") or 0)
        this_n = int((this_ds[ds_id] or {}).get("episodes") or 0)
        if this_n < prev_n:
            breaches.append(
                {
                    "category": "datasets",
                    "signal": f"dataset:{ds_id}",
                    "severity": "medium",
                    "message": (
                        f"Dataset `{ds_id}` lost episodes: {prev_n} → {this_n}. "
                        f"Sources under `data/eval/sources/{ds_id}/` shrank. Was "
                        f"this intentional (eg. dropping flaky episodes)?"
                    ),
                    "this_value": this_n,
                    "prev_value": prev_n,
                }
            )
    return breaches


def collect_informational(this: dict, prev: dict, cfg: dict) -> list[str]:
    """Soft signals — explanatory context for the issue body, not breaches."""
    notes: list[str] = []
    flags = set(cfg.get("informational") or [])

    if "fingerprint_hash_change" in flags:
        a = _get(this, "experiment_run", "fingerprint_hash")
        b = _get(prev, "experiment_run", "fingerprint_hash")
        if a and b and a != b:
            notes.append(f"`experiment_run.fingerprint_hash`: {b[:12]}… → {a[:12]}…")

    if "model_registry_hash_change" in flags:
        a = _get(this, "setup_probe", "model_registry", "preset_hash")
        b = _get(prev, "setup_probe", "model_registry", "preset_hash")
        if a and b and a != b:
            notes.append(f"`model_registry.preset_hash`: {b} → {a}")

    if "git_commit_change" in flags:
        a = _get(this, "setup_probe", "git", "commit", default="?")
        b = _get(prev, "setup_probe", "git", "commit", default="?")
        notes.append(f"git: {b[:7]} → {a[:7]}")

    if "new_profile_names" in flags:
        a = set(_get(this, "setup_probe", "profiles", "names", default=[]) or [])
        b = set(_get(prev, "setup_probe", "profiles", "names", default=[]) or [])
        new = sorted(a - b)
        if new:
            notes.append(f"new profiles since prev: {', '.join(new)}")
        gone = sorted(b - a)
        if gone:
            notes.append(f"profiles removed since prev: {', '.join(gone)}")

    if "new_dataset_ids" in flags:
        a = set((_get(this, "setup_probe", "datasets", default={}) or {}).keys())
        b = set((_get(prev, "setup_probe", "datasets", default={}) or {}).keys())
        new = sorted(a - b)
        if new:
            notes.append(f"new datasets since prev: {', '.join(new)}")

    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly baseline drift check.")
    parser.add_argument(
        "--baselines-dir",
        default="data/baselines",
        type=Path,
        help="Directory containing baseline-*.json files",
    )
    parser.add_argument(
        "--thresholds",
        default="data/baselines/drift_thresholds.yaml",
        type=Path,
        help="YAML thresholds config",
    )
    parser.add_argument(
        "--output",
        default="/tmp/drift-report.json",
        type=Path,
        help="Write structured report here",
    )
    args = parser.parse_args()

    thresholds = load_yaml(args.thresholds)
    this_path, prev_path = latest_two_snapshots(args.baselines_dir)

    report: dict[str, Any]
    if this_path is None:
        report = {
            "status": "no_snapshot",
            "breaches": [],
            "informational": ["No baseline snapshots present in data/baselines/."],
        }
    elif prev_path is None:
        this = load_json(this_path)
        report = {
            "status": "first_snapshot",
            "this_path": str(this_path),
            "this_week_id": this.get("week_id"),
            "breaches": [],
            "informational": [
                f"First snapshot in the series ({this.get('week_id')}). "
                f"Nothing to compare against yet — next week's run will produce the "
                f"first real drift diff."
            ],
        }
    else:
        this = load_json(this_path)
        prev = load_json(prev_path)
        this_exp = this.get("experiment_run") or {}
        prev_exp = prev.get("experiment_run") or {}
        this_probe = this.get("setup_probe") or {}
        prev_probe = prev.get("setup_probe") or {}

        breaches: list[dict] = []
        breaches += check_quality_gates(this_exp, prev_exp, thresholds.get("quality_gates") or {})
        breaches += check_performance(this_exp, prev_exp, thresholds.get("performance") or {})
        breaches += check_length(this_exp, prev_exp, thresholds.get("length") or {})
        breaches += check_datasets(this_probe, prev_probe, thresholds.get("datasets") or {})

        informational = collect_informational(this, prev, thresholds)

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
