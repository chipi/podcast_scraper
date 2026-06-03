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

  auto      Synthesize queries (or read --queries), run both backends, grade with an
            LLM-as-judge, score, and write a graded JSONL for a human spot-check.
              python scripts/eval_hybrid_judged.py auto \
                  --corpus-dir CORPUS --synthesize 40 --model gpt-4o-mini

Between template/score, a human fills `relevance` (0=irrelevant, 1=related,
2=on-point); `auto` does it with an LLM and you validate a sample. The same queries
carry an `intent` label (seeds #860 training data).

Caveat: synthesized queries derive from the corpus's own kg surfaces, so they retain
home-field bias toward FAISS even with question templates. A trustworthy FAISS-removal
/ ML-promotion verdict needs REAL user queries (`--queries`) + a human spot-check of
the LLM grades. Treat synthesized verdicts as directional only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
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


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_TOPIC_TEMPLATES = [
    "how does {x} affect markets",
    "what are the risks of {x}",
    "explain the debate around {x}",
    "why does {x} matter for investors",
]
_ENTITY_TEMPLATES = [
    "what is {x}'s view",
    "what did experts say about {x}",
    "how is {x} positioned",
]


def _synthesize_queries(corpus_dir: Path, n: int) -> List[str]:
    """Bootstrap queries from corpus kg_entity/kg_topic surfaces (no live traffic).

    Uses natural-question TEMPLATES rather than the bare surface label — a bare label
    is a known-item gift to FAISS (it exact-matches its own kg node). This reduces, but
    does not remove, the home-field bias: a truly fair eval uses REAL user queries via
    ``--queries``. Treat synthesized verdicts as directional only.
    """
    store = FaissVectorStore.load(corpus_dir / "search")
    people, topics = [], []
    for meta in store.metadata_by_doc_id.values():
        txt = (meta.get("text") or "").strip()
        if not txt or len(txt) > 50:
            continue
        if meta.get("doc_type") == "kg_entity":
            people.append(txt)
        elif meta.get("doc_type") == "kg_topic":
            topics.append(txt)
    people = list(dict.fromkeys(people))
    topics = list(dict.fromkeys(topics))
    out: List[str] = []
    for i in range(n):
        if topics and (i % 2 == 0 or not people):
            tmpl = _TOPIC_TEMPLATES[i % len(_TOPIC_TEMPLATES)]
            out.append(tmpl.format(x=topics[i % len(topics)]))
        elif people:
            tmpl = _ENTITY_TEMPLATES[i % len(_ENTITY_TEMPLATES)]
            out.append(tmpl.format(x=people[i % len(people)]))
    return out[:n]


def _openai_complete(model: str):
    from openai import OpenAI

    client = OpenAI()

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return complete


def _cmd_auto(args: argparse.Namespace) -> int:
    from podcast_scraper.search.llm_judge import grade_records

    corpus = Path(args.corpus_dir)
    if not lance_index_dir(corpus).exists():
        print(f"No LanceDB index at {lance_index_dir(corpus)} — run `make index-two-tier` first.")
        return 1
    _load_dotenv()
    if args.queries:
        queries = [ln.strip() for ln in Path(args.queries).read_text().splitlines() if ln.strip()]
    else:
        queries = _synthesize_queries(corpus, args.synthesize)
    print(f"Queries: {len(queries)} ({'file' if args.queries else 'synthesized'})")

    records = build_judgment_template(
        queries,
        faiss_ranks=_faiss_ranks(corpus, args.k),
        hybrid_ranks=_hybrid_ranks(corpus, args.k),
        intent_of=classify_query,
        k=args.k,
    )
    print(f"Grading {len(records)} records with {args.model} (LLM-as-judge) ...")
    grade_records(records, _openai_complete(args.model))

    with Path(args.out).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")

    scores = score_from_judgments(records, k=args.k)
    n = int(scores["_judged_count"]["n"])
    print(f"\nGraded → {args.out}  (spot-check a sample to validate the judge)")
    print(f"Judged with signal: {n}/{len(records)}")
    print(f"\n  backend   nDCG@{args.k}   recall@{args.k}")
    for b in ("faiss", "hybrid"):
        print(f"  {b:7s}  {scores[b]['ndcg']:8.3f}  {scores[b]['recall']:8.3f}")
    dn = scores["hybrid"]["ndcg"] - scores["faiss"]["ndcg"]
    verdict = (
        "PASS — hybrid >= FAISS (unblocks #858 + #860)"
        if dn > 0.02 and n >= 30
        else f"INCONCLUSIVE — Δ nDCG={dn:+.3f}, n={n} (want clear margin over >=30)"
    )
    print(f"\n  Δ nDCG = {dn:+.3f}\n  VERDICT: {verdict}")
    return 0


def main() -> int:
    """Dispatch the template / score / auto subcommand."""
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
    a = sub.add_parser("auto", help="synthesize queries, run both backends, LLM-grade, score")
    a.add_argument("--corpus-dir", required=True)
    a.add_argument("--queries", default=None, help="query file (else synthesize from corpus)")
    a.add_argument("--synthesize", type=int, default=40, help="N queries when --queries absent")
    a.add_argument("--model", default="gpt-4o-mini")
    a.add_argument("--out", default="judged_auto.jsonl")
    a.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "template":
        return _cmd_template(args)
    if args.cmd == "score":
        return _cmd_score(args)
    return _cmd_auto(args)


if __name__ == "__main__":
    raise SystemExit(main())
