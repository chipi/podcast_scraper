"""Harvest stance-level speaker pairs for the #1144 disagreement-detector feasibility spike.

#1106 proved cross-Person *atomic-insight* contradictions are near-absent — but real
disagreement lives at the **stance** level: does speaker A's overall position on a topic
oppose speaker B's? This script builds the input for that spike.

For each topic where ≥2 speakers each contribute ≥``--min-insights`` insights (enough to
*have* a stance), it forms every cross-speaker pair and aggregates each speaker's insights
on the topic. A strong LLM (see ``disagreement_stance_silver_v1.py``) then judges whether
the two stances disagree — stance-vs-stance, not atomic pair NLI.

Deterministic given ``--seed``. No model loaded here (pure GI walk), so it runs on a
``.[dev]`` install.

Usage:
    python scripts/eval/score/disagreement_stance_harvest_v1.py \\
        --corpus .test_outputs/manual/prod-v2/corpus \\
        --out data/eval/enrichment/disagreement/harvest_prodv2_v1.jsonl \\
        --n 40 --min-insights 2 --seed 1144
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers.nli_contradiction import (
    _episode_topic_insight_speaker_index,
)
from podcast_scraper.enrichment.paths import discover_episode_bundles


def _topic_speaker_insights(
    corpus_root: Path,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    """topic_id -> speaker_id -> [insight_text]; plus speaker_id -> name."""
    bundles = discover_episode_bundles(corpus_root)
    by_topic, labels = _episode_topic_insight_speaker_index(bundles)
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for tid, entries in by_topic.items():
        for _iid, pid, text in entries:
            if text.strip():
                grouped[tid][pid].append(text)
    return grouped, labels


def _viable_pairs(
    grouped: dict[str, dict[str, list[str]]],
    labels: dict[str, str],
    min_insights: int,
) -> list[dict[str, Any]]:
    """Every cross-speaker pair on a topic where both speakers have >= min_insights."""
    pairs: list[dict[str, Any]] = []
    for tid, spk in grouped.items():
        rich = sorted(p for p, txts in spk.items() if len(txts) >= min_insights)
        for i in range(len(rich)):
            for j in range(i + 1, len(rich)):
                pa, pb = rich[i], rich[j]
                pairs.append(
                    {
                        "topic_id": tid,
                        "speaker_a_id": pa,
                        "speaker_a_name": labels.get(pa, pa),
                        "speaker_a_insights": spk[pa],
                        "speaker_b_id": pb,
                        "speaker_b_name": labels.get(pb, pb),
                        "speaker_b_insights": spk[pb],
                    }
                )
    return pairs


def _topic_diverse_sample(
    rng: random.Random, rows: list[dict[str, Any]], n: int, per_topic_cap: int = 3
) -> list[dict[str, Any]]:
    shuffled = rows[:]
    rng.shuffle(shuffled)
    taken: list[dict[str, Any]] = []
    per_topic: dict[str, int] = {}
    for r in shuffled:
        if len(taken) >= n:
            break
        t = r["topic_id"]
        if per_topic.get(t, 0) >= per_topic_cap:
            continue
        per_topic[t] = per_topic.get(t, 0) + 1
        taken.append(r)
    if len(taken) < n:
        taken.extend([r for r in shuffled if r not in taken][: n - len(taken)])
    return taken


def main() -> int:
    """Harvest stance-level speaker pairs; write JSONL for LLM disagreement judging."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--min-insights", type=int, default=2)
    p.add_argument("--seed", type=int, default=1144)
    args = p.parse_args()

    if not args.corpus.is_dir():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    grouped, labels = _topic_speaker_insights(args.corpus)
    pairs = _viable_pairs(grouped, labels, args.min_insights)
    rng = random.Random(args.seed)
    sample = _topic_diverse_sample(rng, pairs, args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(sample):
            r["pair_id"] = f"{args.seed}-{i:04d}"
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"harvested {len(sample)} stance pairs → {args.out}\n"
        f"  viable pairs in corpus: {len(pairs)} across "
        f"{len({r['topic_id'] for r in pairs})} topics",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
