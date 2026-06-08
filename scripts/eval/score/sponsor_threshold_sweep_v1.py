"""Sponsor-detector threshold sweep — #904 Phase B.

Sweeps the two main `cleaning/commercial/detector.py` constants against the
v2 source corpus + a 3-episode real-prod sample:

- `_HIGH_CONFIDENCE_FOR_LARGE_BLOCK`: threshold above which a low-confidence
  span is allowed to remove a large fraction of the transcript.
- `_INLINE_STANDALONE_CONFIDENCE`: threshold an inline CTA (bare URL / sign-up
  phrase) must clear without corroboration before the detector will remove it.

For each cell: cleaned char-count delta, sponsor pattern residual on cleaned
output (excluding `block_end` boundary patterns, which are transition markers
not content), block count detected.

Usage:
    python scripts/eval/score/sponsor_threshold_sweep_v1.py \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --output  data/eval/runs/baseline_sponsor_thresholds_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.cleaning.commercial.detector import CommercialDetector
from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS

# Block-boundary patterns ARE NOT content — they're transition markers
# ("welcome back to our show", "back to our show"). Counting them in residual
# is a measurement bug that #594 exposed.
_CONTENT_PATTERNS = [p for p in SPONSOR_PATTERNS if p.boundary_hint != "block_end"]


def _content_sponsor_hits(text: str) -> int:
    return sum(sum(1 for _ in pat.pattern.finditer(text)) for pat in _CONTENT_PATTERNS)


def _all_sponsor_hits(text: str) -> int:
    return sum(sum(1 for _ in pat.pattern.finditer(text)) for pat in SPONSOR_PATTERNS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    txt_files = sorted(args.sources.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt under {args.sources}", file=sys.stderr)
        return 1

    transcripts = [(t.stem, t.read_text(encoding="utf-8")) for t in txt_files]

    # Knob grid
    high_conf_grid = [0.75, 0.80, 0.85, 0.90]  # default 0.85
    inline_conf_grid = [0.5, 0.6, 0.7, 0.8]  # default 0.7

    rows: list[dict[str, Any]] = []
    # Patch module-level constants per cell — restore after the sweep
    from podcast_scraper.cleaning.commercial import detector as det_mod

    orig_high = det_mod._HIGH_CONFIDENCE_FOR_LARGE_BLOCK
    orig_inline = det_mod._INLINE_STANDALONE_CONFIDENCE
    try:
        for high_conf in high_conf_grid:
            for inline_conf in inline_conf_grid:
                det_mod._HIGH_CONFIDENCE_FOR_LARGE_BLOCK = high_conf
                det_mod._INLINE_STANDALONE_CONFIDENCE = inline_conf
                detector = CommercialDetector()
                for ep_id, raw in transcripts:
                    candidates = detector.detect(raw)
                    cleaned = detector.remove(raw)
                    rows.append(
                        {
                            "high_conf": high_conf,
                            "inline_conf": inline_conf,
                            "episode_id": ep_id,
                            "raw_chars": len(raw),
                            "cleaned_chars": len(cleaned),
                            "chars_removed_pct": round(
                                100 * (len(raw) - len(cleaned)) / max(len(raw), 1), 2
                            ),
                            "blocks_detected": len(candidates),
                            "residual_content_hits": _content_sponsor_hits(cleaned),
                            "residual_all_hits": _all_sponsor_hits(cleaned),
                            "raw_content_hits": _content_sponsor_hits(raw),
                            "raw_all_hits": _all_sponsor_hits(raw),
                        }
                    )
    finally:
        det_mod._HIGH_CONFIDENCE_FOR_LARGE_BLOCK = orig_high
        det_mod._INLINE_STANDALONE_CONFIDENCE = orig_inline

    # Aggregate per cell
    from collections import defaultdict

    cells: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["high_conf"], r["inline_conf"])].append(r)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for (high_conf, inline_conf), cell_rows in sorted(cells.items()):
        n = len(cell_rows)
        mean_pct = sum(r["chars_removed_pct"] for r in cell_rows) / n
        mean_blocks = sum(r["blocks_detected"] for r in cell_rows) / n
        ep_with_residual = sum(1 for r in cell_rows if r["residual_content_hits"] > 0)
        total_content_removed = sum(
            r["raw_content_hits"] - r["residual_content_hits"] for r in cell_rows
        )
        total_content_raw = sum(r["raw_content_hits"] for r in cell_rows)
        recall = (
            round(100 * total_content_removed / total_content_raw, 1) if total_content_raw else 0.0
        )
        summary.append(
            {
                "high_conf": high_conf,
                "inline_conf": inline_conf,
                "mean_chars_removed_pct": round(mean_pct, 2),
                "mean_blocks_detected": round(mean_blocks, 2),
                "episodes_with_residual": ep_with_residual,
                "total_episodes": n,
                "content_recall_pct": recall,
            }
        )

    metrics = {
        "schema": "metrics_sponsor_threshold_sweep_v1",
        "default_high_conf": orig_high,
        "default_inline_conf": orig_inline,
        "summary": summary,
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"{'high':>5} {'inline':>6} {'chars%':>6} {'blocks':>6} {'recall%':>7} eps_with_residual")
    for s in summary:
        marker = " *" if s["high_conf"] == orig_high and s["inline_conf"] == orig_inline else "  "
        print(
            f"{s['high_conf']:>5} {s['inline_conf']:>6} "
            f"{s['mean_chars_removed_pct']:>6.2f} {s['mean_blocks_detected']:>6.2f} "
            f"{s['content_recall_pct']:>7.1f} {s['episodes_with_residual']}/{s['total_episodes']}"
            f"{marker}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
