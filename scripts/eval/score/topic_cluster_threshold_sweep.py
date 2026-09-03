#!/usr/bin/env python
"""Sweep ``topic_cluster_threshold`` and score each value by CROSS-EPISODE reach.

Why this exists
---------------
`topic_cluster_threshold` ships at **0.75**, and `config.py` says why: *"Pareto-optimal at 0.75 on
v2 fixtures per EVAL_FIXTURES_V2_TIER1_TUNING_2026_06_08."* Fixtures — not the corpus.
`capability_audit.py` already flags the gap: *"0.75, tuned on v2 fixtures in June and never
re-measured on real data."* Meanwhile RFC-075's own production sweep (1,178 topics, 2026-04)
recommended **0.70**, and its status line claims that became the default. It did not.

At 9,512 topics the shipped value produces **85.7% singletons** against the 69% RFC-075 measured
at 0.70 on a corpus 8x smaller. The threshold is demonstrably binding: converting
`similarity_to_centroid` to pairwise for the 492 size-2 clusters (83% of all), the minimum is
exactly 0.750 and **38.6% merged within 0.03 of the threshold** — mass piled against the wall.

.. warning::

   **The topic count is not consistent across this branch's write-ups and none of them should be
   quoted as "the" corpus size.** Four figures appear for what all claim to be the 1,066-episode
   corpus:

   =========  ===========================================================
   9,263      ``topic_cooccurrence_corpus`` — canonicalisation measurement
   9,345      ``temporal_velocity`` — its topic universe (8,743 singletons)
   9,512      this file
   9,594      ``config.py`` / ``model_registry.py``, citing THIS script
   =========  ===========================================================

   The last two are a direct contradiction: they attribute different totals to the same sweep.
   The likeliest explanation is that they were taken at different points while the Batch A
   ingestion was still running — the corpus grew to 1,066 episodes during this work — but that is
   an explanation, not a verification, and no one has re-derived them since. Some may also count
   different populations (all KG topics vs. topics reaching the velocity window). Treat every one
   as approximate, and re-run this script before quoting a number in a decision.

Scoring by the right number
---------------------------
Cluster COUNT is the wrong success metric, and `config.py` says so: *"Lower values surface
near-singleton parents without adding cross-feed value."* A merge of two near-synonyms inside one
episode (`ai-safety` + `ai-alignment`) inflates the count and connects nothing.
`capability_audit.py` names the discriminator: *"Size cannot tell those apart. Feed span can."*

So this sweep reports, per threshold, the metric that actually matters:

* ``clusters``            — total (the number that misleads)
* ``xepisode_clusters``   — clusters whose members span >= 2 DISTINCT episodes
* ``xfeed_clusters``      — clusters whose members span >= 2 DISTINCT feeds  <- the product claim
* ``intra_episode``       — clusters confined to ONE episode (the failure mode above)

At the shipped 0.75 the live corpus scores 596 clusters, of which **26.7% are confined to a single
episode** and only 30% span three or more. A candidate threshold is better only if it grows
``xfeed_clusters`` faster than ``intra_episode``.

Faithfulness
------------
Uses the repo's own ``cluster_indices_by_threshold`` — the same average-linkage merge production
runs — so a sweep result is comparable to the shipped artifact rather than to a re-implementation.
Topic vectors are rebuilt the way ``search/indexer.py`` builds them
(``_kg_embed_text_topic`` = label + optional description) and then averaged per ``topic_id`` and
re-normalised exactly as ``collect_topic_rows_from_lance`` does.

**This reproduces the vectors; it does not read them.** The production vectors live in the LanceDB
index on the box. If the indexed embedding ever stops being a pure function of label+description,
this harness silently drifts from production — verify with ``--verify-against`` before trusting a
recommendation.

Input
-----
``topics.jsonl``, one row per episode::

    {"ep": "<metadata relpath>", "topics": [{"id": "topic:x", "label": "...", "desc": "..."}]}

Usage
-----
    python scripts/eval/score/topic_cluster_threshold_sweep.py topics.jsonl \\
        --thresholds 0.60,0.65,0.70,0.75,0.80 \\
        --verify-against /path/to/topic_clusters.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _embed_text(label: str, desc: str) -> str:
    """Mirror ``search/indexer.py::_kg_embed_text_topic`` exactly."""
    parts = []
    if label and label.strip():
        parts.append(label.strip())
    if desc and desc.strip():
        parts.append(desc.strip())
    return " ".join(parts)


def load_rows(path: Path) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (ordered topic_ids, texts_by_topic, episodes_by_topic)."""
    texts: Dict[str, List[str]] = defaultdict(list)
    episodes: Dict[str, List[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = str(row.get("ep") or "")
            for t in row.get("topics") or []:
                tid = t.get("id")
                if not isinstance(tid, str) or not tid:
                    continue
                txt = _embed_text(str(t.get("label") or ""), str(t.get("desc") or ""))
                if not txt:
                    continue
                texts[tid].append(txt)
                if ep:
                    episodes[tid].append(ep)
    return sorted(texts), texts, episodes


def feed_of(ep_relpath: str) -> str:
    """Feed directory from a corpus-relative metadata path (``feeds/<feed>/run_.../...``)."""
    parts = ep_relpath.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "feeds" else ep_relpath


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("topics_jsonl", type=Path)
    ap.add_argument("--thresholds", default="0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Live topic_clusters.json — checks this harness reproduces production at its "
        "own threshold before any recommendation is believed.",
    )
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from podcast_scraper.search.topic_clusters import (
        cluster_indices_by_threshold,
        cosine_similarity_matrix,
    )

    topic_ids, texts, episodes = load_rows(args.topics_jsonl)
    if not topic_ids:
        print("no topics found", file=sys.stderr)
        return 1
    print(f"topics: {len(topic_ids)}", file=sys.stderr)

    # One embedding per OCCURRENCE, then mean per topic_id + renormalise — the same aggregation
    # collect_topic_rows_from_lance performs over the indexed rows.
    model = SentenceTransformer(args.model)
    flat: List[str] = []
    owner: List[int] = []
    for i, tid in enumerate(topic_ids):
        for txt in texts[tid]:
            flat.append(txt)
            owner.append(i)
    print(f"embedding {len(flat)} topic occurrences ...", file=sys.stderr)
    embs = model.encode(flat, batch_size=256, show_progress_bar=False, convert_to_numpy=True)

    dim = embs.shape[1]
    acc = np.zeros((len(topic_ids), dim), dtype=np.float64)
    cnt = np.zeros(len(topic_ids), dtype=np.float64)
    for vec, idx in zip(embs, owner):
        acc[idx] += vec
        cnt[idx] += 1
    mean = acc / np.maximum(cnt[:, None], 1.0)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    vectors = np.asarray(mean / np.maximum(norms, 1e-12), dtype=np.float32)

    sim = cosine_similarity_matrix(vectors)

    eps_by_topic = {t: set(episodes[t]) for t in topic_ids}
    feeds_by_topic = {t: {feed_of(e) for e in episodes[t]} for t in topic_ids}

    if args.verify_against and args.verify_against.is_file():
        live = json.loads(args.verify_against.read_text(encoding="utf-8"))
        lt = float(live.get("threshold", 0.75))
        labels = cluster_indices_by_threshold(sim, lt)
        groups = defaultdict(list)
        for i, g in enumerate(labels):
            groups[int(g)].append(i)
        got_clusters = sum(1 for m in groups.values() if len(m) > 1)
        print(
            f"\nVERIFY at production threshold {lt}: "
            f"harness={got_clusters} clusters / {len(topic_ids)} topics  |  "
            f"live={live.get('cluster_count')} clusters / {live.get('topic_count')} topics",
            file=sys.stderr,
        )
        print(
            "  (a large gap means the harness is NOT reproducing production — do not trust the "
            "sweep below until it is explained)",
            file=sys.stderr,
        )

    print(
        f"\n{'thresh':>7} {'clusters':>9} {'singletons':>11} {'sing%':>7} "
        f"{'x-episode':>10} {'x-feed':>8} {'intra-ep':>9} {'max':>5}"
    )
    for raw in args.thresholds.split(","):
        t = float(raw)
        labels = cluster_indices_by_threshold(sim, t)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, g in enumerate(labels):
            groups[int(g)].append(i)
        multi = [m for m in groups.values() if len(m) > 1]
        singles = len(groups) - len(multi)
        xep = xfeed = intra = 0
        for m in multi:
            eps: set = set()
            fds: set = set()
            for i in m:
                eps |= eps_by_topic[topic_ids[i]]
                fds |= feeds_by_topic[topic_ids[i]]
            if len(eps) >= 2:
                xep += 1
            if len(fds) >= 2:
                xfeed += 1
            if len(eps) <= 1:
                intra += 1
        biggest = max((len(m) for m in multi), default=0)
        pct = singles / len(topic_ids) * 100
        print(
            f"{t:7.2f} {len(multi):9d} {singles:11d} {pct:6.1f}% "
            f"{xep:10d} {xfeed:8d} {intra:9d} {biggest:5d}"
        )

    print(
        "\nPick on x-feed, not on clusters: a threshold is better only if it grows x-feed "
        "faster than intra-ep.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
