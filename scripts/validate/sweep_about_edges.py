"""Sweep K and floor params for the ABOUT-edge semantic fix (#664).

Replays existing gi.json artifacts from a corpus, recomputes insight-topic
cosine similarity via sentence-transformers, and reports edge counts for each
(K, floor) combo in the sweep grid. No pipeline changes — pure analysis.

Usage:
    .venv/bin/python scripts/validate/sweep_about_edges.py \
        --corpus /path/to/my-manual-run4 \
        --sample-size 10 \
        --spot-check-eps 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

K_GRID = [1, 2, 3]
FLOOR_GRID = [0.0, 0.20, 0.25, 0.30]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def collect_gi_files(corpus: Path) -> List[Path]:
    return sorted(corpus.glob("**/*.gi.json"))


def load_episode(gi_path: Path) -> Dict[str, Any]:
    with gi_path.open() as f:
        gi = json.load(f)

    insights = [n for n in gi.get("nodes", []) if n.get("type") == "Insight"]
    topics = [n for n in gi.get("nodes", []) if n.get("type") == "Topic"]

    insight_rows = [
        {"id": n["id"], "text": (n.get("properties") or {}).get("text") or ""} for n in insights
    ]
    topic_rows = [
        {"id": n["id"], "label": (n.get("properties") or {}).get("label") or ""} for n in topics
    ]

    about_edges = sum(1 for e in gi.get("edges", []) if e.get("type") == "ABOUT")

    return {
        "path": gi_path,
        "episode_id": gi.get("episode_id", ""),
        "insights": insight_rows,
        "topics": topic_rows,
        "about_edges_orig": about_edges,
    }


def select_edges(sim: np.ndarray, k: int, floor: float) -> List[List[Tuple[int, float]]]:
    """For each insight (row), pick top-k topic columns with cosine >= floor."""
    picks: List[List[Tuple[int, float]]] = []
    for row in sim:
        scored = [(j, float(row[j])) for j in range(len(row)) if row[j] >= floor]
        scored.sort(key=lambda t: t[1], reverse=True)
        picks.append(scored[:k])
    return picks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--spot-check-eps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Score every episode, not just sample")
    args = parser.parse_args()

    gi_files = collect_gi_files(args.corpus)
    if not gi_files:
        print(f"No gi.json files found under {args.corpus}", file=sys.stderr)
        return 1

    print(f"Found {len(gi_files)} gi.json files under {args.corpus}")

    rng = random.Random(args.seed)
    if args.all:
        sample = gi_files
    else:
        sample = rng.sample(gi_files, min(args.sample_size, len(gi_files)))

    print(f"Sweeping on {len(sample)} episodes (seed={args.seed})")

    # Load model once
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # Per-combo aggregates
    combo_edges: Dict[Tuple[int, float], List[int]] = defaultdict(list)
    chosen_cosines: Dict[Tuple[int, float], List[float]] = defaultdict(list)
    total_orig_edges = 0

    # Per-episode record (for spot-checks and correlation)
    episodes_data: List[Dict[str, Any]] = []

    for idx, gi_path in enumerate(sample, start=1):
        ep = load_episode(gi_path)
        if not ep["insights"] or not ep["topics"]:
            continue

        texts_insights = [r["text"] or r["id"] for r in ep["insights"]]
        texts_topics = [r["label"] or r["id"] for r in ep["topics"]]

        emb_i = encoder.encode(texts_insights, normalize_embeddings=True)
        emb_t = encoder.encode(texts_topics, normalize_embeddings=True)
        sim = np.asarray(emb_i) @ np.asarray(emb_t).T  # (n_ins, n_top)

        n_ins = len(ep["insights"])
        n_top = len(ep["topics"])
        cross = n_ins * n_top
        total_orig_edges += ep["about_edges_orig"]

        for k in K_GRID:
            for floor in FLOOR_GRID:
                picks = select_edges(sim, k, floor)
                edge_count = sum(len(p) for p in picks)
                combo_edges[(k, floor)].append(edge_count)
                for p in picks:
                    for _, cos in p:
                        chosen_cosines[(k, floor)].append(cos)

        episodes_data.append(
            {
                "path": gi_path,
                "episode_id": ep["episode_id"],
                "n_insights": n_ins,
                "n_topics": n_top,
                "cross": cross,
                "orig_edges": ep["about_edges_orig"],
                "sim": sim,
                "insights": ep["insights"],
                "topics": ep["topics"],
            }
        )

        if idx % 10 == 0 or idx == len(sample):
            print(f"  {idx}/{len(sample)} episodes processed")

    # Report
    print("")
    print("=" * 88)
    print("SWEEP RESULTS — edge count per combo (sum across sampled episodes)")
    print("=" * 88)
    n_eps_scored = len(episodes_data)
    print(f"Sampled episodes: {n_eps_scored}")
    print(
        f"Original ABOUT edges total (sampled): {total_orig_edges} "
        f"({total_orig_edges / max(n_eps_scored, 1):.1f} avg/ep)"
    )
    print("")

    header = (
        f"{'K':>3} {'floor':>6}  {'edges_total':>12} {'edges/ep':>10} "
        f"{'vs_orig_%':>10} {'cos_p25':>8} {'cos_p50':>8} {'cos_p75':>8}"
    )
    print(header)
    print("-" * len(header))
    for k in K_GRID:
        for floor in FLOOR_GRID:
            totals = combo_edges[(k, floor)]
            edge_total = sum(totals)
            per_ep = edge_total / max(n_eps_scored, 1)
            pct_of_orig = 100.0 * edge_total / max(total_orig_edges, 1)
            cos_list = chosen_cosines[(k, floor)]
            p25 = float(np.percentile(cos_list, 25)) if cos_list else 0.0
            p50 = float(np.percentile(cos_list, 50)) if cos_list else 0.0
            p75 = float(np.percentile(cos_list, 75)) if cos_list else 0.0
            print(
                f"{k:>3} {floor:>6.2f}  {edge_total:>12d} {per_ep:>10.1f} "
                f"{pct_of_orig:>9.1f}% {p25:>8.3f} {p50:>8.3f} {p75:>8.3f}"
            )

    # Cosine distribution of the full similarity matrix (what the LLM would see without filter)
    print("")
    print("=" * 88)
    print("COSINE DISTRIBUTION — all (insight, topic) pairs across sample")
    print("=" * 88)
    all_cos = np.concatenate([ep["sim"].flatten() for ep in episodes_data])
    print(f"  count = {len(all_cos)}")
    print(
        f"  min = {all_cos.min():.3f}  p10 = {np.percentile(all_cos, 10):.3f}  "
        f"p25 = {np.percentile(all_cos, 25):.3f}  p50 = {np.percentile(all_cos, 50):.3f}"
    )
    print(
        f"  p75 = {np.percentile(all_cos, 75):.3f}  p90 = {np.percentile(all_cos, 90):.3f}  "
        f"max = {all_cos.max():.3f}"
    )
    # Buckets
    buckets = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.8),
        (0.8, 1.01),
    ]
    print("  bucket counts (share):")
    for lo, hi in buckets:
        count = int(((all_cos >= lo) & (all_cos < hi)).sum())
        share = count / len(all_cos)
        bar = "#" * max(1, int(share * 80))
        print(f"    [{lo:.2f}, {hi:.2f})  {count:>7d}  {share*100:>5.1f}%  {bar}")

    # Spot checks
    print("")
    print("=" * 88)
    print(f"SPOT-CHECK — {args.spot_check_eps} random episodes (K=3, no floor)")
    print("=" * 88)
    spot = rng.sample(episodes_data, min(args.spot_check_eps, len(episodes_data)))
    for ep in spot:
        print("")
        print(f"Episode: {ep['path'].name}")
        print(
            f"  {ep['n_insights']} insights × {ep['n_topics']} topics "
            f"= {ep['cross']} cross-product (orig ABOUT: {ep['orig_edges']})"
        )
        print("  Topics:")
        for t in ep["topics"]:
            print(f"    {t['id']}")
            print(f"      label: {t['label'][:100]}")
        print("  Top-3 topics per insight:")
        for i, ins in enumerate(ep["insights"]):
            row = ep["sim"][i]
            ranked = sorted(enumerate(row), key=lambda t: t[1], reverse=True)[:3]
            print(f"    [{i}] {ins['text'][:90]}")
            for j, cos in ranked:
                print(f"        cos={cos:.3f}  →  {ep['topics'][j]['label'][:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
