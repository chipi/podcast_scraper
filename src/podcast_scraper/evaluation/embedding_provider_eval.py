"""Embedding-provider A/B eval (ADR-098 / #897).

Compares retrieval quality of two ``vector_embedding_provider`` choices on
the operator's own corpus. Uses gi.json ``SUPPORTED_BY`` edges as ground
truth — every Insight has supporting Quote(s) linked by the pipeline's
grounding stage. We treat each insight as a query and ask each provider:
"does the supporting quote rank high in your nearest-neighbour search?"

Why this is honest as ground truth:
- The pipeline's grounding choice is LLM-driven (text↔text reasoning), not
  embedding-driven, so neither provider gets an unfair advantage.
- Insights ARE the kind of thing users semantically search for; quotes ARE
  the kind of evidence they want surfaced. Domain-realistic.
- Scales for free as the corpus grows.

Outputs ``data/eval/embedding_provider_comparison/<timestamp>/report.{json,md}``
matching the autoresearch report shape (drops into the existing ledger).

Run: ``make embedding-provider-eval CORPUS=./output``
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple

PairsFn = Callable[[Path], List["GroundTruthPair"]]

from podcast_scraper.providers.ml import embedding_loader
from podcast_scraper.search.chunker import chunk_transcript
from podcast_scraper.search.corpus_scope import discover_metadata_files

logger = logging.getLogger(__name__)

# Recall/nDCG cutoffs surfaced in the report (small first → larger).
K_VALUES: Tuple[int, ...] = (1, 5, 10, 20)


@dataclass
class GroundTruthPair:
    """One (insight, supporting-quote) pair extracted from gi.json."""

    episode_id: str
    insight_id: str
    insight_text: str
    quote_id: str
    quote_text: str


@dataclass
class ProviderConfig:
    """One row of the A/B matrix."""

    label: str
    provider: str  # "sentence_transformers" | "ollama"
    model_id: str
    endpoint: Optional[str] = None  # base URL for ollama; None for in-process


@dataclass
class ProviderMetrics:
    """Per-provider results aggregated over all queries."""

    label: str
    provider: str
    model_id: str
    n_queries: int
    recall_at: Dict[int, float] = field(default_factory=dict)
    ndcg_at: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    embed_latency_ms_p50: float = 0.0
    embed_latency_ms_p95: float = 0.0
    dim: int = 0


# ----- Ground truth extraction -------------------------------------------------


def extract_pairs_from_corpus(corpus_root: Path) -> List[GroundTruthPair]:
    """Walk ``corpus_root`` and yield every ``Insight -SUPPORTED_BY-> Quote`` pair.

    Standard pipeline output: each episode has a ``*.metadata.json`` next to
    its ``*.gi.json``. Flat fixture layouts (no metadata sidecar) are also
    supported via a fallback rglob.
    """
    pairs: List[GroundTruthPair] = []
    seen_gi: set[Path] = set()

    # Standard layout: anchor on .metadata.json and find sibling .gi.json.
    for meta_path in discover_metadata_files(corpus_root):
        gi_path = meta_path.parent / meta_path.name.replace(".metadata.json", ".gi.json")
        if not gi_path.is_file():
            continue
        seen_gi.add(gi_path.resolve())
        pairs.extend(_load_pairs(gi_path))

    # Fallback for flat fixture layouts (production-shaped, e2e bundles, etc.).
    for gi_path in corpus_root.rglob("*.gi.json"):
        if gi_path.resolve() in seen_gi:
            continue
        pairs.extend(_load_pairs(gi_path))

    return pairs


def _load_pairs(gi_path: Path) -> List[GroundTruthPair]:
    try:
        doc = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("skip unreadable gi.json %s: %s", gi_path, exc)
        return []
    return _pairs_from_gi(doc)


def extract_transcript_pairs_from_corpus(corpus_root: Path) -> List[GroundTruthPair]:
    """Yield (insight, full-episode-transcript) pairs.

    Tests the long-context hypothesis directly: a 2k–3k token transcript is
    orders of magnitude larger than MiniLM's 256-token window. The
    ``quote_id`` / ``quote_text`` fields are reused for the transcript role
    (avoids dataclass churn — same harness mechanics either way).

    The transcript path comes from ``metadata.json.content.transcript_file_path``;
    if absent, that episode is skipped.
    """
    pairs: List[GroundTruthPair] = []
    for meta_path in corpus_root.rglob("*.metadata.json"):
        gi_path = meta_path.parent / meta_path.name.replace(".metadata.json", ".gi.json")
        if not gi_path.is_file():
            continue
        try:
            meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
            gi_doc = json.loads(gi_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skip unreadable artifact next to %s: %s", meta_path, exc)
            continue
        rel_tx_path = (meta_doc.get("content") or {}).get("transcript_file_path")
        if not rel_tx_path:
            continue
        tx_path = _resolve_transcript_path(meta_path, rel_tx_path)
        if tx_path is None or not tx_path.is_file():
            logger.warning("transcript missing for %s: %s", meta_path.name, rel_tx_path)
            continue
        try:
            transcript_text = tx_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("transcript unreadable %s: %s", tx_path, exc)
            continue
        if not transcript_text:
            continue
        episode_id = str(gi_doc.get("episode_id") or tx_path.stem)
        transcript_id = f"transcript:{episode_id}"
        for node in gi_doc.get("nodes", []):
            if not isinstance(node, dict) or node.get("type") != "Insight":
                continue
            insight_text = ((node.get("properties") or {}).get("text") or "").strip()
            if not insight_text:
                continue
            pairs.append(
                GroundTruthPair(
                    episode_id=episode_id,
                    insight_id=str(node.get("id", "")),
                    insight_text=insight_text,
                    quote_id=transcript_id,
                    quote_text=transcript_text,
                )
            )
    return pairs


def _resolve_transcript_path(meta_path: Path, rel: str) -> Optional[Path]:
    """Find transcripts/<name>.txt relative to the corpus root or its parents."""
    # Try parent of metadata/ (run dir) first, then walk up.
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    for candidate_root in (meta_path.parent.parent, meta_path.parent.parent.parent):
        candidate = candidate_root / rel
        if candidate.exists():
            return candidate
    return None


def _pairs_from_gi(doc: Dict) -> List[GroundTruthPair]:
    nodes_by_id = {n["id"]: n for n in doc.get("nodes", []) if isinstance(n, dict) and "id" in n}
    episode_id = str(doc.get("episode_id", ""))
    out: List[GroundTruthPair] = []
    for edge in doc.get("edges", []):
        if not isinstance(edge, dict) or edge.get("type") != "SUPPORTED_BY":
            continue
        src = nodes_by_id.get(edge.get("from", ""))
        dst = nodes_by_id.get(edge.get("to", ""))
        if not src or not dst:
            continue
        if src.get("type") != "Insight" or dst.get("type") != "Quote":
            continue
        i_text = (src.get("properties", {}) or {}).get("text") or ""
        q_text = (dst.get("properties", {}) or {}).get("text") or ""
        if not i_text.strip() or not q_text.strip():
            continue
        out.append(
            GroundTruthPair(
                episode_id=episode_id,
                insight_id=str(src["id"]),
                insight_text=i_text.strip(),
                quote_id=str(dst["id"]),
                quote_text=q_text.strip(),
            )
        )
    return out


# ----- Embedding + scoring -----------------------------------------------------


# nomic-embed-text strictly enforces 8192 tokens. Real podcast transcripts with
# "Speaker 1: ..." formatting tokenize at ~1.5 chars/token (dense vs synthetic test
# text), so 8000 chars ≈ 5333 tokens is a safe ceiling. Both providers see only
# the truncated head — fair comparison, both biased toward the start of the
# document. MiniLM's silent ~256-token (≈1024 char) cut applies on top.
_EMBED_INPUT_MAX_CHARS = 8_000


def _embed_one(text: str, cfg: ProviderConfig) -> Tuple[List[float], float]:
    """Embed one text; return (vector, elapsed_ms). Truncates to a context-safe length."""
    safe_text = text[:_EMBED_INPUT_MAX_CHARS]
    t0 = time.perf_counter()
    vec = embedding_loader.encode(
        safe_text,
        cfg.model_id,
        return_numpy=False,
        allow_download=False,
        remote_endpoint=cfg.endpoint,
        provider=cfg.provider,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if not isinstance(vec, list) or not vec or not isinstance(vec[0], float):
        raise RuntimeError(f"provider {cfg.label} returned malformed embedding")
    return cast(List[float], vec), dt_ms


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = max(0, min(len(sorted_v) - 1, int(round(q * (len(sorted_v) - 1)))))
    return float(sorted_v[k])


def _dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def _ndcg_at(ranked_ids: Sequence[str], target_id: str, k: int) -> float:
    grades = [1.0 if doc_id == target_id else 0.0 for doc_id in ranked_ids[:k]]
    idcg = _dcg([1.0])  # one relevant item; ideal = it ranks first
    return _dcg(grades) / idcg if idcg > 0 else 0.0


def _recall_at(ranked_ids: Sequence[str], target_id: str, k: int) -> float:
    return 1.0 if target_id in ranked_ids[:k] else 0.0


def _reciprocal_rank(ranked_ids: Sequence[str], target_id: str) -> float:
    for i, doc_id in enumerate(ranked_ids):
        if doc_id == target_id:
            return 1.0 / (i + 1)
    return 0.0


def score_provider(pairs: Sequence[GroundTruthPair], cfg: ProviderConfig) -> ProviderMetrics:
    """Embed all quotes + insights once with *cfg*, then rank quotes per insight query."""
    quote_ids: List[str] = []
    quote_vecs: List[List[float]] = []
    embed_latencies_ms: List[float] = []

    # 1. Embed every unique quote once (corpus-level pass).
    seen_quotes: Dict[str, int] = {}
    for p in pairs:
        if p.quote_id in seen_quotes:
            continue
        seen_quotes[p.quote_id] = len(quote_ids)
        vec, dt = _embed_one(p.quote_text, cfg)
        quote_vecs.append(vec)
        quote_ids.append(p.quote_id)
        embed_latencies_ms.append(dt)

    dim = len(quote_vecs[0]) if quote_vecs else 0
    recall_sum: Dict[int, float] = {k: 0.0 for k in K_VALUES}
    ndcg_sum: Dict[int, float] = {k: 0.0 for k in K_VALUES}
    mrr_sum = 0.0
    n_queries = 0

    # 2. Per insight: embed query, cosine-rank quotes, score.
    for p in pairs:
        try:
            q_vec, dt = _embed_one(p.insight_text, cfg)
        except RuntimeError as exc:
            logger.warning("skip query %s: %s", p.insight_id, exc)
            continue
        embed_latencies_ms.append(dt)
        scored = [(quote_ids[i], _cosine(q_vec, quote_vecs[i])) for i in range(len(quote_vecs))]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        ranked_ids = [doc_id for doc_id, _ in scored]
        for k in K_VALUES:
            recall_sum[k] += _recall_at(ranked_ids, p.quote_id, k)
            ndcg_sum[k] += _ndcg_at(ranked_ids, p.quote_id, k)
        mrr_sum += _reciprocal_rank(ranked_ids, p.quote_id)
        n_queries += 1

    n = max(n_queries, 1)
    return ProviderMetrics(
        label=cfg.label,
        provider=cfg.provider,
        model_id=cfg.model_id,
        n_queries=n_queries,
        recall_at={k: recall_sum[k] / n for k in K_VALUES},
        ndcg_at={k: ndcg_sum[k] / n for k in K_VALUES},
        mrr=mrr_sum / n,
        embed_latency_ms_p50=_percentile(embed_latencies_ms, 0.50),
        embed_latency_ms_p95=_percentile(embed_latencies_ms, 0.95),
        dim=dim,
    )


def score_provider_chunked(
    pairs: Sequence[GroundTruthPair],
    cfg: ProviderConfig,
    *,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> ProviderMetrics:
    """Production-realistic transcript retrieval: chunk first, then embed.

    Mirrors what the live indexer does (``vector_chunk_size_tokens=300``,
    ``vector_chunk_overlap_tokens=50``). Each transcript becomes N chunks;
    each chunk gets its own vector. For each insight query, we rank ALL
    chunks across ALL transcripts and take the max chunk score per
    transcript as the transcript's score. This is exactly how production
    search retrieves at the transcript level via the per-chunk index.

    Fair to both providers — same chunking strategy; the model competes
    only on within-chunk embedding quality, not on whole-doc handling.
    """
    chunk_transcript_ids: List[str] = []
    chunk_vecs: List[List[float]] = []
    embed_latencies_ms: List[float] = []

    # 1. Per unique transcript: chunk, embed each chunk.
    seen_transcripts: set[str] = set()
    for p in pairs:
        if p.quote_id in seen_transcripts:
            continue
        seen_transcripts.add(p.quote_id)
        chunks = chunk_transcript(
            p.quote_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens
        )
        for ch in chunks:
            try:
                vec, dt = _embed_one(ch.text, cfg)
            except RuntimeError as exc:
                logger.warning("skip chunk in %s: %s", p.quote_id, exc)
                continue
            chunk_vecs.append(vec)
            chunk_transcript_ids.append(p.quote_id)
            embed_latencies_ms.append(dt)

    dim = len(chunk_vecs[0]) if chunk_vecs else 0
    recall_sum: Dict[int, float] = {k: 0.0 for k in K_VALUES}
    ndcg_sum: Dict[int, float] = {k: 0.0 for k in K_VALUES}
    mrr_sum = 0.0
    n_queries = 0

    # 2. Per insight: embed query, score all chunks, max-pool by transcript.
    for p in pairs:
        try:
            q_vec, dt = _embed_one(p.insight_text, cfg)
        except RuntimeError as exc:
            logger.warning("skip query %s: %s", p.insight_id, exc)
            continue
        embed_latencies_ms.append(dt)
        per_transcript_max: Dict[str, float] = {}
        for i, ch_vec in enumerate(chunk_vecs):
            tid = chunk_transcript_ids[i]
            score = _cosine(q_vec, ch_vec)
            if score > per_transcript_max.get(tid, -1e9):
                per_transcript_max[tid] = score
        ranked = sorted(per_transcript_max.items(), key=lambda kv: kv[1], reverse=True)
        ranked_ids = [tid for tid, _ in ranked]
        for k in K_VALUES:
            recall_sum[k] += _recall_at(ranked_ids, p.quote_id, k)
            ndcg_sum[k] += _ndcg_at(ranked_ids, p.quote_id, k)
        mrr_sum += _reciprocal_rank(ranked_ids, p.quote_id)
        n_queries += 1

    n = max(n_queries, 1)
    return ProviderMetrics(
        label=cfg.label,
        provider=cfg.provider,
        model_id=cfg.model_id,
        n_queries=n_queries,
        recall_at={k: recall_sum[k] / n for k in K_VALUES},
        ndcg_at={k: ndcg_sum[k] / n for k in K_VALUES},
        mrr=mrr_sum / n,
        embed_latency_ms_p50=_percentile(embed_latencies_ms, 0.50),
        embed_latency_ms_p95=_percentile(embed_latencies_ms, 0.95),
        dim=dim,
    )


# ----- Report writers ----------------------------------------------------------


def write_report_json(
    metrics: Sequence[ProviderMetrics],
    pairs: Sequence[GroundTruthPair],
    dest: Path,
) -> None:
    """Write the machine-readable A/B report (schema-versioned) to *dest*."""
    payload = {
        "schema_version": 1,
        "experiment": "embedding_provider_comparison",
        "ground_truth": {
            "source": "gi.json SUPPORTED_BY edges",
            "n_pairs": len(pairs),
            "n_unique_quotes": len({p.quote_id for p in pairs}),
            "n_unique_insights": len({p.insight_id for p in pairs}),
        },
        "providers": [asdict(m) for m in metrics],
        "k_values": list(K_VALUES),
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_md(
    metrics: Sequence[ProviderMetrics], pairs: Sequence[GroundTruthPair], dest: Path
) -> None:
    """Write the human-readable Markdown A/B report to *dest*."""
    lines: List[str] = []
    lines.append("# Embedding-provider comparison")
    lines.append("")
    n_ins = len({p.insight_id for p in pairs})
    n_qts = len({p.quote_id for p in pairs})
    lines.append("- Ground truth: gi.json `SUPPORTED_BY` edges (Insight → Quote)")
    lines.append(f"- Pairs: **{len(pairs)}** ({n_ins} insights × {n_qts} unique quotes)")
    lines.append("")
    lines.append("## Retrieval metrics (higher is better)")
    lines.append("")
    header = ["Metric"] + [m.label for m in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for k in K_VALUES:
        row = [f"Recall@{k}"] + [f"{m.recall_at.get(k, 0.0):.3f}" for m in metrics]
        lines.append("| " + " | ".join(row) + " |")
    for k in K_VALUES:
        row = [f"nDCG@{k}"] + [f"{m.ndcg_at.get(k, 0.0):.3f}" for m in metrics]
        lines.append("| " + " | ".join(row) + " |")
    mrr_row = ["MRR"] + [f"{m.mrr:.3f}" for m in metrics]
    lines.append("| " + " | ".join(mrr_row) + " |")
    lines.append("")
    lines.append("## Operational characteristics")
    lines.append("")
    op_header = ["Field"] + [m.label for m in metrics]
    lines.append("| " + " | ".join(op_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(op_header)) + " |")
    lines.append("| Provider | " + " | ".join(m.provider for m in metrics) + " |")
    lines.append("| Model | " + " | ".join(m.model_id for m in metrics) + " |")
    lines.append("| Vector dim | " + " | ".join(str(m.dim) for m in metrics) + " |")
    lines.append(
        "| Embed latency p50 (ms) | "
        + " | ".join(f"{m.embed_latency_ms_p50:.1f}" for m in metrics)
        + " |"
    )
    lines.append(
        "| Embed latency p95 (ms) | "
        + " | ".join(f"{m.embed_latency_ms_p95:.1f}" for m in metrics)
        + " |"
    )
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- **Recall@1**: how often the model nails the exact supporting quote first try.")
    lines.append("- **Recall@10**: how often it's in the user's top-10 (the practical UX bar).")
    lines.append("- **MRR**: average rank-quality across all queries.")
    lines.append("- A 5+ point Recall@10 delta is materially felt in retrieval UX.")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----- Orchestration / CLI -----------------------------------------------------


def run_comparison(
    corpus_root: Path,
    output_root: Path,
    matrix: Sequence[ProviderConfig],
    *,
    timestamp: str,
    max_pairs: Optional[int] = None,
    pairs_fn: Optional["PairsFn"] = None,
    chunked: bool = False,
) -> Path:
    """Run the A/B sweep + write reports. Returns the run dir path."""
    extractor = pairs_fn or extract_pairs_from_corpus
    pairs = extractor(corpus_root)
    if not pairs:
        raise RuntimeError(
            f"No SUPPORTED_BY pairs found under {corpus_root}. Run the pipeline first "
            "so gi.json artifacts exist with grounding edges."
        )
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    logger.info("Found %d insight→quote pairs across corpus", len(pairs))

    scorer = score_provider_chunked if chunked else score_provider
    metrics: List[ProviderMetrics] = []
    for cfg in matrix:
        logger.info("Scoring provider: %s (%s / %s)", cfg.label, cfg.provider, cfg.model_id)
        metrics.append(scorer(pairs, cfg))

    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(metrics, pairs, run_dir / "report.json")
    write_report_md(metrics, pairs, run_dir / "report.md")
    return run_dir
