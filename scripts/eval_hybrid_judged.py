#!/usr/bin/env python3
"""Human-judged hybrid-vs-FAISS eval (RFC-057 / wire-live follow-up C).

The discriminating eval the saturated known-item proxy (eval_two_tier_retrieval.py)
could not provide — it gates FAISS removal (#858) and ML-router promotion (#860).

Two modes:

  template  Run a query set through FAISS + hybrid, emit a judgment template JSONL
            (one record per query, each candidate awaiting a `relevance` grade).
              python scripts/eval_hybrid_judged.py template \
                  --corpus-dir CORPUS --queries queries.txt --out template.jsonl

  score     Read the human-graded JSONL back and print mean nDCG@k / recall@k per
            backend + the verdict.
              python scripts/eval_hybrid_judged.py score --judged template.jsonl

Between the two, a human opens the template and fills `relevance` (0=irrelevant,
1=related, 2=on-point) for the candidates that matter. The same queries carry an
`intent` label (seeds #860 training data).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from podcast_scraper.search.faiss_store import FaissVectorStore  # noqa: E402
from podcast_scraper.search.hybrid_search import hybrid_candidates, lance_index_dir  # noqa: E402
from podcast_scraper.search.judged_eval import (  # noqa: E402
    build_judgment_template,
    JudgmentRecord,
    score_from_judgments,
)
from podcast_scraper.search.router import classify_query  # noqa: E402

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _faiss_ranks(corpus_dir: Path, k: int):
    from podcast_scraper.providers.ml import embedding_loader

    store = FaissVectorStore.load(corpus_dir / "search")

    def ranks(query: str) -> Sequence[Tuple[str, str]]:
        emb = embedding_loader.encode(
            query, store.stats().embedding_model, return_numpy=False, allow_download=True
        )
        hits = store.search(list(emb), top_k=k * 4)
        out = [(h.doc_id, str(h.metadata.get("text") or "")) for h in hits]
        return out[:k]

    return ranks


def _hybrid_ranks(corpus_dir: Path, k: int):
    def ranks(query: str) -> Sequence[Tuple[str, str]]:
        rows = hybrid_candidates(corpus_dir, query, top_k=k, embedding_model=_MODEL) or []
        return [(r.doc_id, str(r.metadata.get("text") or "")) for r in rows][:k]

    return ranks


def _cmd_template(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus_dir)
    if not lance_index_dir(corpus).exists():
        print(f"No LanceDB index at {lance_index_dir(corpus)} — run `make index-two-tier` first.")
        return 1
    queries = [
        ln.strip()
        for ln in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    records = build_judgment_template(
        queries,
        faiss_ranks=_faiss_ranks(corpus, args.k),
        hybrid_ranks=_hybrid_ranks(corpus, args.k),
        intent_of=classify_query,
        k=args.k,
    )
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")
    print(f"Wrote {len(records)} judgment records → {args.out}. Fill `relevance`, then `score`.")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    records: List[JudgmentRecord] = []
    for line in Path(args.judged).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            d["relevance"] = {k: int(v) for k, v in (d.get("relevance") or {}).items()}
            records.append(
                JudgmentRecord(
                    **{
                        f: d.get(f)
                        for f in (
                            "query",
                            "intent",
                            "candidates",
                            "faiss_ranking",
                            "hybrid_ranking",
                            "relevance",
                        )
                    }
                )
            )
    scores = score_from_judgments(records, k=args.k)
    n = int(scores["_judged_count"]["n"])
    print(f"Judged queries with signal: {n}/{len(records)}")
    print(f"\n  backend   nDCG@{args.k}   recall@{args.k}")
    for b in ("faiss", "hybrid"):
        print(f"  {b:7s}  {scores[b]['ndcg']:8.3f}  {scores[b]['recall']:8.3f}")
    dn = scores["hybrid"]["ndcg"] - scores["faiss"]["ndcg"]
    if dn > 0.02 and n >= 30:
        verdict = "PASS — hybrid clearly beats FAISS (unblocks #858 removal + #860 promotion)"
    else:
        verdict = f"INCONCLUSIVE — need >=30 judged queries (have {n}) and a clear margin"
    print(f"\n  Δ nDCG = {dn:+.3f}\n  VERDICT: {verdict}")
    return 0


def main() -> int:
    """Dispatch the template / score subcommand."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="build a judgment template")
    t.add_argument("--corpus-dir", required=True)
    t.add_argument("--queries", required=True, help="one query per line")
    t.add_argument("--out", default="judgment_template.jsonl")
    t.add_argument("--k", type=int, default=10)
    s = sub.add_parser("score", help="score graded judgments")
    s.add_argument("--judged", required=True)
    s.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    return _cmd_template(args) if args.cmd == "template" else _cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
