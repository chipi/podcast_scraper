"""Harvest topic-similarity candidate pools for silver gold labeling (#1105).

For a sample of topics, builds a candidate-neighbour pool that lets a strong LLM (see
``enrichment_topic_similarity_silver_v1.py``) pick the *genuinely related* topics — the gold
``expected_neighbours`` the recall@K scorer needs. The pool per topic:

* the enricher's predicted **top_k** neighbours (measures precision — are predictions real),
* **cluster siblings** from ``search/topic_clusters.json`` (plausible neighbours the enricher
  may have *missed* — the real recall signal),
* a few **random distractors** (non-neighbours for silver to reject).

All candidate ids are constrained to the enricher's own topic vocabulary (the ids in
``topic_similarity.json``) so the scorer's id-intersection is well-defined. Deterministic
given ``--seed``. No model loaded here.

Usage:
    python scripts/eval/score/enrichment_topic_similarity_harvest_v1.py \\
        --corpus .test_outputs/manual/prod-v2/corpus \\
        --out data/eval/enrichment/topic_similarity/harvest_prodv2_v1.jsonl \\
        --n 24 --seed 1105
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _load(corpus: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    """Return (topic -> {label, top_k}), (id -> label), (topic -> [sibling_ids in vocab])."""
    sim = json.loads((corpus / "enrichments" / "topic_similarity.json").read_text("utf-8"))
    topics = {t["topic_id"]: t for t in (sim.get("data") or {}).get("topics") or []}
    label_of: dict[str, str] = {}
    for tid, t in topics.items():
        label_of[tid] = str(t.get("topic_label") or _slug(tid.split(":", 1)[-1]))
        for n in t.get("top_k") or []:
            label_of.setdefault(str(n.get("topic_id")), str(n.get("topic_label") or ""))
    vocab = set(topics)  # the enricher's id space

    siblings: dict[str, list[str]] = {}
    clusters_path = corpus / "search" / "topic_clusters.json"
    if clusters_path.is_file():
        clusters = json.loads(clusters_path.read_text("utf-8")).get("clusters") or []
        for c in clusters:
            members = c.get("members") or []
            ids = []
            for m in members:
                mid = m.get("topic_id") or m.get("id")
                if not mid and m.get("label"):
                    mid = "topic:" + _slug(m["label"])
                mid = str(mid) if mid else ""
                if mid:
                    ids.append(mid)
                    label_of.setdefault(mid, str(m.get("label") or ""))
            for mid in ids:
                if mid in vocab:
                    siblings.setdefault(mid, []).extend(x for x in ids if x != mid and x in vocab)
    return topics, label_of, siblings


def main() -> int:
    """Sample topics + build candidate pools; write JSONL for silver labeling."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--distractors", type=int, default=5)
    p.add_argument("--seed", type=int, default=1105)
    args = p.parse_args()

    if not args.corpus.is_dir():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    topics, label_of, siblings = _load(args.corpus)
    vocab = list(topics)
    rng = random.Random(args.seed)

    # Prefer topics that have BOTH predictions and >=1 cluster sibling (so precision AND
    # recall are measurable); backfill with prediction-only topics if short.
    rich = [t for t in vocab if (topics[t].get("top_k")) and siblings.get(t)]
    rng.shuffle(rich)
    chosen = rich[: args.n]
    if len(chosen) < args.n:
        rest = [t for t in vocab if t not in set(chosen) and topics[t].get("top_k")]
        rng.shuffle(rest)
        chosen.extend(rest[: args.n - len(chosen)])

    with args.out.open("w", encoding="utf-8") as fh:
        for i, tid in enumerate(chosen):
            pred = [str(n["topic_id"]) for n in (topics[tid].get("top_k") or [])[: args.top_k]]
            sib = list(dict.fromkeys(siblings.get(tid, [])))[:6]
            pool_ids = list(dict.fromkeys(pred + sib))
            distractor_src = [x for x in vocab if x != tid and x not in set(pool_ids)]
            rng.shuffle(distractor_src)
            pool_ids += distractor_src[: args.distractors]
            rng.shuffle(pool_ids)
            row = {
                "pair_id": f"{args.seed}-{i:04d}",
                "topic_id": tid,
                "topic_label": label_of.get(tid, tid),
                "predicted_top_k": pred,
                "candidates": [
                    {"id": c, "label": label_of.get(c, c.split(":", 1)[-1])} for c in pool_ids
                ],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"harvested {len(chosen)} topics → {args.out}  "
        f"(rich topics with siblings available: {len(rich)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
