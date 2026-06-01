#!/usr/bin/env python3
"""Eval two-tier hybrid retrieval vs the FAISS baseline (RFC-090 Stage 4 / #858).

The Stage-4 gate: does hybrid (BM25 + dense + RRF over the two-tier LanceDB index,
#855/#856) beat the current vector-only FAISS store on a real corpus? Because the
migration (``search/migration.py``) reuses the FAISS embeddings verbatim, the dense
signal is held constant across both systems — so the measured delta isolates the
value of adding BM25 + RRF fusion, which is precisely the RFC-090 hypothesis.

Eval design — known-item retrieval, no human judgments:
  For each sampled insight, build a *term-subset* query from its salient content
  words (stopwords dropped, capped at --query-terms). This is neither a verbatim
  slice (so BM25 gets no free exact-phrase hit) nor the full text (so the dense
  vector gets no free self-similarity ~1.0). Gold = that insight. Both systems are
  scored on the insight tier only, apples-to-apples.

  Limitation: a term-subset known-item proxy correlates with, but is not, a
  human-judged relevance eval. Treat the numbers as a *relative* A/B, not absolute
  retrieval quality. A labeled query set (also feeds #860's ML router) supersedes
  this when it exists.

Metrics (k=10): recall@k, MRR@k, nDCG@k (single relevant doc).

Decision rule (RFC-090 Phase 3): FAISS removal is gated on hybrid ≥ FAISS on MRR
*and* recall here. This script reports the verdict; it does not remove anything.

Usage:
  python scripts/eval_two_tier_retrieval.py \
      --faiss-dir .test_outputs/manual/my-manual-run-10/search \
      --lance-path /tmp/lance_eval --sample 200 --k 10
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo src on path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from podcast_scraper.providers.ml.embedding_loader import encode  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from podcast_scraper.search.faiss_store import FaissVectorStore  # noqa: E402
from podcast_scraper.search.migration import migrate_faiss_to_lance  # noqa: E402
from podcast_scraper.search.retrieval import RetrievalLayer  # noqa: E402

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "our",
    "their",
    "his",
    "her",
    "so",
    "like",
    "just",
    "about",
    "which",
    "what",
    "when",
    "where",
    "how",
    "why",
    "who",
    "from",
    "by",
    "at",
    "into",
    "than",
    "then",
    "there",
    "here",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
    "can",
    "will",
    "would",
    "could",
    "should",
    "more",
    "most",
    "very",
    "really",
}


def _term_subset_query(text: str, *, max_terms: int) -> str:
    """Salient content terms from *text* (stopwords dropped, order preserved)."""
    seen: set[str] = set()
    terms: List[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z'\-]+", text):
        low = tok.lower()
        if low in _STOPWORDS or len(low) < 3 or low in seen:
            continue
        seen.add(low)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def _ndcg_at(rank: Optional[int]) -> float:
    return 1.0 / math.log2(rank + 1) if rank else 0.0


def _rank_of(doc_ids: List[str], gold: str) -> Optional[int]:
    for i, did in enumerate(doc_ids):
        if did == gold:
            return i + 1
    return None


def _faiss_insight_ranks(store: FaissVectorStore, embedding: List[float], k: int) -> List[str]:
    # Overfetch then filter to insights (FAISS holds 6 doc types).
    hits = store.search(embedding, top_k=k * 6)
    out = [h.doc_id for h in hits if h.metadata.get("doc_type") == "insight"]
    return out[:k]


def _hybrid_insight_ranks(
    layer: RetrievalLayer, query: str, embedding: List[float], k: int
) -> List[str]:
    results = layer.retrieve(query, embedding, k=k * 2, tier="insight", signals="hybrid")
    return [r.doc_id for r in results][:k]


def _evaluate(
    queries: List[Tuple[str, str]],  # (query_text, gold_insight_id)
    store: FaissVectorStore,
    layer: RetrievalLayer,
    k: int,
) -> Dict[str, Dict[str, float]]:
    agg = {sys: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for sys in ("faiss", "hybrid")}
    n = len(queries)
    for query, gold in queries:
        emb = encode(query, "minilm-l6", allow_download=True)
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        for name, ranks in (
            ("faiss", _faiss_insight_ranks(store, emb_list, k)),
            ("hybrid", _hybrid_insight_ranks(layer, query, emb_list, k)),
        ):
            rank = _rank_of(ranks, gold)
            agg[name]["recall"] += 1.0 if rank else 0.0
            agg[name]["mrr"] += 1.0 / rank if rank else 0.0
            agg[name]["ndcg"] += _ndcg_at(rank)
    for name in agg:
        for metric in agg[name]:
            agg[name][metric] /= max(n, 1)
    return agg


def main() -> int:
    """Run the two-tier-vs-FAISS eval and print the verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faiss-dir", required=True, help="dir with vectors.faiss")
    ap.add_argument("--lance-path", default=None, help="LanceDB dir (default: temp)")
    ap.add_argument("--sample", type=int, default=200, help="insight queries to eval")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query-terms", type=int, default=8)
    args = ap.parse_args()

    lance_path = args.lance_path or tempfile.mkdtemp(prefix="lance_eval_")
    print(f"Migrating FAISS → LanceDB at {lance_path} ...")
    stats = migrate_faiss_to_lance(args.faiss_dir, lance_path)
    print(f"  segments={stats.segments} insights={stats.insights} skipped={stats.skipped}")

    store = FaissVectorStore.load(Path(args.faiss_dir))
    backend = LanceDBBackend(lance_path, embed_dim=store.embedding_dim)
    layer = RetrievalLayer(backend)

    # Build the known-item query set: deterministic sample of insights with usable text.
    md = store.metadata_by_doc_id
    insights = sorted(
        (did, m) for did, m in md.items() if m.get("doc_type") == "insight" and m.get("text")
    )
    insights = insights[:: max(1, len(insights) // max(args.sample, 1))][: args.sample]
    queries: List[Tuple[str, str]] = []
    for did, m in insights:
        q = _term_subset_query(m["text"], max_terms=args.query_terms)
        if len(q.split()) >= 3:  # need a non-degenerate query
            queries.append((q, did))
    print(f"Evaluating {len(queries)} known-item queries (k={args.k}) ...")

    agg = _evaluate(queries, store, layer, args.k)
    print("\n  system   recall@{k}   MRR@{k}   nDCG@{k}".format(k=args.k))
    for name in ("faiss", "hybrid"):
        a = agg[name]
        print(f"  {name:7s}  {a['recall']:8.3f}  {a['mrr']:7.3f}  {a['ndcg']:7.3f}")

    f, h = agg["faiss"], agg["hybrid"]
    wins = h["mrr"] >= f["mrr"] and h["recall"] >= f["recall"]
    verdict = (
        "PASS — hybrid ≥ FAISS (FAISS removal unblocked)"
        if wins
        else ("HOLD — hybrid did not beat FAISS; keep FAISS, investigate")
    )
    print(f"\n  Δ MRR={h['mrr'] - f['mrr']:+.3f}  Δ recall={h['recall'] - f['recall']:+.3f}")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
