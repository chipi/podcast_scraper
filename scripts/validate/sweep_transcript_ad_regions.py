"""Sweep window size and hit-threshold params for transcript-level ad detection.

Reads raw ``transcripts/*.txt`` files from a corpus, slides a char-window,
counts distinct ``_AD_PATTERNS`` hits per window, flags windows above a
threshold, merges adjacent flagged windows into regions. Reports aggregate
flagging rates per (window_size, threshold) combo and spot-checks random
flagged regions so the operator can eyeball precision/recall before deciding
whether to wire a real excision step (#663 option a).

Usage:
    .venv/bin/python scripts/validate/sweep_transcript_ad_regions.py \
        --corpus /path/to/my-manual-run4 \
        --spot-check-regions 6
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Reuse the production ad patterns so the sweep reflects real wiring.
sys.path.insert(0, "src")
from podcast_scraper.gi.filters import _AD_PATTERNS  # type: ignore  # noqa: E402

WINDOW_GRID = [1000, 2000, 3000]
THRESHOLD_GRID = [2, 3, 4]


def distinct_pattern_hits(text: str) -> int:
    """Count how many *distinct* _AD_PATTERNS match anywhere in ``text``.

    Matches the production threshold semantics in
    ``filters.insight_looks_like_ad`` (≥ 2 distinct patterns).
    """
    return sum(1 for p in _AD_PATTERNS if p.search(text))


def find_flagged_regions(text: str, window_size: int, threshold: int) -> List[Tuple[int, int, int]]:
    """Slide a window with 50% overlap; flag where hits ≥ threshold; merge
    overlapping flagged windows into ``(start, end, max_hits)`` regions.
    """
    if not text or window_size <= 0:
        return []
    step = max(1, window_size // 2)
    raw: List[Tuple[int, int, int]] = []
    for start in range(0, max(1, len(text) - window_size + 1), step):
        end = min(len(text), start + window_size)
        hits = distinct_pattern_hits(text[start:end])
        if hits >= threshold:
            raw.append((start, end, hits))
    # Also consider a trailing window if the last step didn't cover EOF
    if raw and raw[-1][1] < len(text) and len(text) > window_size:
        tail_start = len(text) - window_size
        hits = distinct_pattern_hits(text[tail_start:])
        if hits >= threshold:
            raw.append((tail_start, len(text), hits))

    # Merge overlapping / adjacent flagged windows
    merged: List[Tuple[int, int, int]] = []
    for start, end, hits in raw:
        if merged and start <= merged[-1][1]:
            m_start, m_end, m_hits = merged[-1]
            merged[-1] = (m_start, max(m_end, end), max(m_hits, hits))
        else:
            merged.append((start, end, hits))
    return merged


def collect_transcripts(corpus: Path) -> List[Path]:
    return sorted(corpus.glob("**/transcripts/*.txt"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--spot-check-regions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    transcripts = collect_transcripts(args.corpus)
    if not transcripts:
        print(f"No transcripts found under {args.corpus}", file=sys.stderr)
        return 1
    print(f"Found {len(transcripts)} transcripts under {args.corpus}")
    print(f"Loaded {len(_AD_PATTERNS)} ad patterns from gi.filters")
    print()

    # Pre-load all transcripts (they're small, ~60KB each)
    loaded: List[Tuple[Path, str]] = []
    for p in transcripts:
        try:
            loaded.append((p, p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    total_chars = sum(len(t) for _, t in loaded)
    print(f"Total transcript bytes: {total_chars:,}")
    print()

    # Aggregate sweep
    print("=" * 92)
    print("SWEEP RESULTS — (window_size, threshold)")
    print("=" * 92)
    header = (
        f"{'window':>7} {'thresh':>6}  {'eps_flagged':>11} {'regions':>8} "
        f"{'chars_flagged':>13} {'pct_corpus':>10} {'avg_hits/region':>15}"
    )
    print(header)
    print("-" * len(header))

    # Store per-combo flagged regions for spot-checks
    best_combo_regions = {}

    for win in WINDOW_GRID:
        for thr in THRESHOLD_GRID:
            eps_flagged = 0
            total_regions = 0
            total_flagged_chars = 0
            hit_scores: List[int] = []
            regions_for_spotcheck: List[Tuple[Path, str, Tuple[int, int, int]]] = []
            for path, txt in loaded:
                regions = find_flagged_regions(txt, win, thr)
                if regions:
                    eps_flagged += 1
                    total_regions += len(regions)
                    for s, e, h in regions:
                        total_flagged_chars += e - s
                        hit_scores.append(h)
                        regions_for_spotcheck.append((path, txt, (s, e, h)))
            pct = 100.0 * total_flagged_chars / max(total_chars, 1)
            avg_hits = sum(hit_scores) / max(len(hit_scores), 1)
            print(
                f"{win:>7d} {thr:>6d}  {eps_flagged:>11d} {total_regions:>8d} "
                f"{total_flagged_chars:>13,d} {pct:>9.2f}% {avg_hits:>15.2f}"
            )
            best_combo_regions[(win, thr)] = regions_for_spotcheck

    print()
    print("=" * 92)
    print(
        f"SPOT-CHECK — {args.spot_check_regions} random flagged regions at "
        f"window=2000, threshold=2"
    )
    print("=" * 92)
    rng = random.Random(args.seed)
    spot_source = best_combo_regions.get((2000, 2), [])
    if not spot_source:
        print("  No flagged regions at window=2000, threshold=2.")
    else:
        picks = rng.sample(spot_source, min(args.spot_check_regions, len(spot_source)))
        for i, (path, txt, (s, e, h)) in enumerate(picks, start=1):
            print()
            print(f"[{i}] {path.name}")
            print(f"    chars {s:,}–{e:,} / {len(txt):,}  (hits: {h})")
            # Show the region with a little surrounding context
            ctx_before = max(0, s - 100)
            ctx_after = min(len(txt), e + 100)
            snippet = txt[ctx_before:ctx_after]
            # Annotate region boundaries inline
            region_start_in_snippet = s - ctx_before
            region_end_in_snippet = e - ctx_before
            annotated = (
                snippet[:region_start_in_snippet]
                + ">>>FLAG>>>"
                + snippet[region_start_in_snippet:region_end_in_snippet]
                + "<<<END<<<"
                + snippet[region_end_in_snippet:]
            )
            # Pretty-print (one sentence per line-ish)
            printable = annotated.replace("Speaker ", "\n      Speaker ")
            print("    " + printable[:1800])

    # Episodes never flagged at any combo — candidates for "content-only" episodes.
    print()
    print("=" * 92)
    print("NEVER-FLAGGED EPISODES (at window=2000, threshold=2) — recall check")
    print("=" * 92)
    flagged_paths_at_2000_2 = {p for p, _, _ in best_combo_regions[(2000, 2)]}
    never_flagged = [p for p, _ in loaded if p not in flagged_paths_at_2000_2]
    print(f"Episodes never flagged at (2000, 2): {len(never_flagged)} / {len(loaded)}")
    for p in never_flagged[:10]:
        print(f"  {p.name}")
    if len(never_flagged) > 10:
        print(f"  ... +{len(never_flagged) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
