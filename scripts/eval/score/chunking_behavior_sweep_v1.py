"""Chunking-behavior sweep for #905 Phase B.

Doesn't run the full ML summarizer (which is expensive on CPU and would need
a silver summary for each long ep). Instead reports the **chunking strategy
log** the ticket asks for: at each (chunk_words, chunk_overlap) cell, how
many chunks does each long-context v2 episode produce, what's the chunk-size
distribution, what's the total overlap-derived overhead.

This surfaces whether a different cell would meaningfully change the
chunking strategy. If chunk counts are similar across cells, the defaults
don't matter much for the v2 corpus and a deeper summarizer sweep isn't
worth the compute. If counts diverge, the deeper sweep is warranted.

Usage:
    python scripts/eval/score/chunking_behavior_sweep_v1.py \\
        --transcripts tests/fixtures/transcripts/v2/p07_e01.txt \\
                      tests/fixtures/transcripts/v2/p08_e01.txt \\
                      tests/fixtures/transcripts/v2/p09_e01.txt \\
        --output data/eval/runs/baseline_chunking_behavior_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.providers.ml.summarizer import chunk_text_words

CHUNK_WORDS_GRID = [600, 750, 900, 1050, 1200]
OVERLAP_GRID = [100, 150, 200]

# Module-level defaults (record for the "vs default" comparison)
DEFAULT_CHUNK_WORDS = 900
DEFAULT_OVERLAP = 150


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    transcripts = []
    for tpath in args.transcripts:
        if not tpath.exists():
            print(f"  SKIP {tpath}: missing", file=sys.stderr)
            continue
        text = tpath.read_text(encoding="utf-8")
        transcripts.append((tpath.stem, text, len(text.split())))

    rows: list[dict[str, Any]] = []
    for chunk_words in CHUNK_WORDS_GRID:
        for overlap in OVERLAP_GRID:
            for ep_id, text, word_count in transcripts:
                chunks = chunk_text_words(text, chunk_size=chunk_words, overlap=overlap)
                chunk_sizes = [len(c.split()) for c in chunks]
                # Effective unique words (no double-count) ≈ word_count
                # Total emitted (with overlap) = sum(chunk_sizes)
                overlap_overhead_pct = round(
                    100 * (sum(chunk_sizes) - word_count) / max(word_count, 1), 1
                )
                rows.append(
                    {
                        "chunk_words": chunk_words,
                        "overlap": overlap,
                        "episode_id": ep_id,
                        "word_count": word_count,
                        "n_chunks": len(chunks),
                        "min_chunk_words": min(chunk_sizes) if chunk_sizes else 0,
                        "max_chunk_words": max(chunk_sizes) if chunk_sizes else 0,
                        "mean_chunk_words": (
                            round(sum(chunk_sizes) / len(chunk_sizes), 1) if chunk_sizes else 0
                        ),
                        "overlap_overhead_pct": overlap_overhead_pct,
                    }
                )

    # Per-cell aggregation
    from collections import defaultdict

    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["chunk_words"], r["overlap"])].append(r)

    summary = []
    for (cw, ov), cell_rows in sorted(cells.items()):
        summary.append(
            {
                "chunk_words": cw,
                "overlap": ov,
                "is_default": cw == DEFAULT_CHUNK_WORDS and ov == DEFAULT_OVERLAP,
                "mean_n_chunks": round(sum(r["n_chunks"] for r in cell_rows) / len(cell_rows), 2),
                "max_n_chunks": max(r["n_chunks"] for r in cell_rows),
                "mean_chunk_size": round(
                    sum(r["mean_chunk_words"] for r in cell_rows) / len(cell_rows), 1
                ),
                "mean_overlap_overhead_pct": round(
                    sum(r["overlap_overhead_pct"] for r in cell_rows) / len(cell_rows), 1
                ),
                "per_episode_chunks": {r["episode_id"]: r["n_chunks"] for r in cell_rows},
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "metrics_chunking_behavior_sweep_v1",
        "default_chunk_words": DEFAULT_CHUNK_WORDS,
        "default_overlap": DEFAULT_OVERLAP,
        "summary": summary,
        "rows": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"{'words':>6} {'overlap':>7} {'mean_chunks':>11} {'max_chunks':>10} "
        f"{'mean_size':>9} {'overlap_overhead':>16}"
    )
    for s in summary:
        marker = " *" if s["is_default"] else "  "
        print(
            f"{s['chunk_words']:>6} {s['overlap']:>7} {s['mean_n_chunks']:>11.2f} "
            f"{s['max_n_chunks']:>10} {s['mean_chunk_size']:>9.1f} "
            f"{s['mean_overlap_overhead_pct']:>15.1f}%{marker}"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
