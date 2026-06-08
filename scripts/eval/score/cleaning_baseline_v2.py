"""Cleaning hit-rate baseline (issue #903 AC).

For each episode in a v2 source dataset:
1. Count SPONSOR_PATTERNS hits in the raw transcript.
2. Run the commercial-block detector and remove sponsor blocks.
3. Count SPONSOR_PATTERNS hits in the cleaned transcript.

Acceptance per #903: raw hit-rate >80% of episodes carry sponsor content,
cleaned hit-rate <5% retained after cleaning.

Usage:
    python scripts/eval/score/cleaning_baseline_v2.py \\
        --source data/eval/sources/curated_5feeds_raw_v2 \\
        --output data/eval/runs/cleaning_curated_5feeds_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.cleaning.commercial.detector import CommercialDetector
from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS


def count_sponsor_hits(text: str) -> dict[str, int]:
    """Return per-pattern hit counts plus total."""
    total = 0
    per_pattern: dict[str, int] = {}
    for pat in SPONSOR_PATTERNS:
        n = sum(1 for _ in pat.pattern.finditer(text))
        if n:
            per_pattern[pat.pattern.pattern] = n
            total += n
    return {"total": total, "per_pattern": per_pattern}


def score_episode(transcript_path: Path) -> dict[str, Any]:
    raw = transcript_path.read_text(encoding="utf-8")
    raw_hits = count_sponsor_hits(raw)

    detector = CommercialDetector()
    candidates = detector.detect(raw)
    cleaned = detector.remove(raw)
    cleaned_hits = count_sponsor_hits(cleaned)

    abs_path = transcript_path.resolve()
    try:
        rel = str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(abs_path)

    return {
        "episode_id": transcript_path.stem,
        "transcript_path": rel,
        "raw_chars": len(raw),
        "cleaned_chars": len(cleaned),
        "chars_removed": len(raw) - len(cleaned),
        "chars_removed_pct": round(100 * (len(raw) - len(cleaned)) / len(raw), 2) if raw else 0,
        "blocks_detected": len(candidates),
        "sponsor_pattern_hits_raw": raw_hits["total"],
        "sponsor_pattern_hits_cleaned": cleaned_hits["total"],
        "sponsor_pattern_hits_per_pattern_raw": raw_hits["per_pattern"],
        "sponsor_pattern_hits_per_pattern_cleaned": cleaned_hits["per_pattern"],
        "has_sponsor_raw": raw_hits["total"] > 0,
        "has_sponsor_cleaned": cleaned_hits["total"] > 0,
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    if not n:
        return {}
    raw_pos = sum(1 for e in episodes if e["has_sponsor_raw"])
    cleaned_pos = sum(1 for e in episodes if e["has_sponsor_cleaned"])
    raw_total = sum(e["sponsor_pattern_hits_raw"] for e in episodes)
    cleaned_total = sum(e["sponsor_pattern_hits_cleaned"] for e in episodes)
    return {
        "episode_count": n,
        "raw_episode_hit_rate": round(100 * raw_pos / n, 1),
        "cleaned_episode_hit_rate": round(100 * cleaned_pos / n, 1),
        "raw_pattern_hits_total": raw_total,
        "cleaned_pattern_hits_total": cleaned_total,
        "pattern_hits_retained_pct": (
            round(100 * cleaned_total / raw_total, 2) if raw_total else 0
        ),
        "blocks_detected_total": sum(e["blocks_detected"] for e in episodes),
        "chars_removed_total": sum(e["chars_removed"] for e in episodes),
        "chars_removed_pct_mean": round(sum(e["chars_removed_pct"] for e in episodes) / n, 2),
        "ac_targets": {
            "raw_episode_hit_rate_target_gt": 80.0,
            "cleaned_episode_hit_rate_target_lt": 5.0,
        },
        "ac_pass": {
            "raw_episode_hit_rate": (100 * raw_pos / n) > 80.0,
            "cleaned_episode_hit_rate": (100 * cleaned_pos / n) < 5.0,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, required=True, help="Source directory (curated_5feeds_raw_v2)"
    )
    p.add_argument(
        "--output", type=Path, required=True, help="Output directory for baseline artifacts"
    )
    p.add_argument("--baseline-id", default="baseline_cleaning_curated_5feeds_v2")
    args = p.parse_args()

    if not args.source.is_dir():
        print(f"Source dir not found: {args.source}", file=sys.stderr)
        return 1

    txt_files = sorted(args.source.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files in {args.source}", file=sys.stderr)
        return 1

    episodes = [score_episode(t) for t in txt_files]
    agg = aggregate(episodes)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "baseline_id": args.baseline_id,
        "task": "commercial_cleaning",
        "source_id": args.source.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episodes),
        "aggregate": agg,
        "episodes": episodes,
        "schema": "metrics_cleaning_baseline_v1",
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    source_rel = args.source.resolve()
    try:
        source_rel = source_rel.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    raw_verdict = "PASS" if agg["ac_pass"]["raw_episode_hit_rate"] else "FAIL"
    clean_verdict = "PASS" if agg["ac_pass"]["cleaned_episode_hit_rate"] else "FAIL"
    raw_rate = agg["raw_episode_hit_rate"]
    clean_rate = agg["cleaned_episode_hit_rate"]
    report = [
        f"# Cleaning Baseline — {args.baseline_id}",
        "",
        f"**Source:** `{source_rel}`  ",
        f"**Episodes:** {agg['episode_count']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value | AC target |",
        "| --- | ---: | --- |",
        f"| Raw episode hit-rate | {raw_rate}% | >80% ({raw_verdict}) |",
        f"| Cleaned episode hit-rate | {clean_rate}% | <5% ({clean_verdict}) |",
        f"| Raw pattern hits (total) | {agg['raw_pattern_hits_total']} | — |",
        f"| Cleaned pattern hits (total) | {agg['cleaned_pattern_hits_total']} | — |",
        f"| Pattern hits retained | {agg['pattern_hits_retained_pct']}% | — |",
        f"| Blocks detected (total) | {agg['blocks_detected_total']} | — |",
        f"| Chars removed (mean) | {agg['chars_removed_pct_mean']}% | — |",
        "",
        "## Per-episode",
        "",
        "| Episode | Blocks | Raw hits | Cleaned hits | Chars removed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ep in episodes:
        report.append(
            f"| {ep['episode_id']} | {ep['blocks_detected']} | "
            f"{ep['sponsor_pattern_hits_raw']} | {ep['sponsor_pattern_hits_cleaned']} | "
            f"{ep['chars_removed_pct']}% |"
        )
    (args.output / "metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(
        f"  raw_episode_hit_rate={agg['raw_episode_hit_rate']}% "
        f"cleaned_episode_hit_rate={agg['cleaned_episode_hit_rate']}% "
        f"retained={agg['pattern_hits_retained_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
