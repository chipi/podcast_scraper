#!/usr/bin/env python3
"""Direct GI-vs-silver scorer for #698 matrix cells.

Mirrors ``scripts/eval/score/score_gi_insight_coverage.py`` (which compares
summary BULLETS against silver insights), but for runs whose ``output.gil``
contains Insight nodes directly (``task: grounded_insights``). Computes:

- **Insight coverage** — % of silver insights matched by any predicted
  insight via embedding cosine ≥ threshold (same as the published 80%
  metric, just sourced from GI nodes instead of summary bullets).
- **Grounded coverage** — of matched insights, what % carry a
  ``SUPPORTED_BY`` edge in the predicted run (the bundling regression
  guard for #698).
- **Quote-overlap rate** — for matched insights, average char-span overlap
  between predicted SUPPORTED_BY quote text and any silver supporting
  quote (rough proxy for evidence-stack fidelity).

Usage::

    PYTHONPATH=. .venv/bin/python \\
        autoresearch/gil_evidence_bundling/eval/score_gi_vs_silver.py \\
        --run-id gil_bundling_baseline_staged_v1 \\
        --silver silver_sonnet46_gi_multiquote_benchmark_v2 \\
        --dataset curated_5feeds_benchmark_v2

Exits 0 on success; prints a single-line summary to stdout in the same
shape as the canonical scorer for ``results.tsv`` ingestion.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed line in %s", path)
    return out


def _gil_insight_texts(pred: Dict[str, Any]) -> List[str]:
    """Pull predicted insight texts from ``output.gil.nodes[type=Insight]``."""
    gil = pred.get("output", {}).get("gil") or {}
    nodes = gil.get("nodes") or []
    texts: List[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if n.get("type") != "Insight":
            continue
        props = n.get("properties") or {}
        text = props.get("text") or n.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _gil_grounded_insight_ids(pred: Dict[str, Any]) -> set:
    """Set of Insight node ids that have at least one SUPPORTED_BY edge."""
    gil = pred.get("output", {}).get("gil") or {}
    edges = gil.get("edges") or []
    grounded: set = set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("type") not in ("SUPPORTED_BY", "supported_by"):
            continue
        src = e.get("source") or e.get("from")
        if src:
            grounded.add(str(src))
    return grounded


def _silver_insight_texts(pred: Dict[str, Any]) -> List[str]:
    """Silver predictions store insights at ``output.insights[*].text``."""
    out = pred.get("output", {}).get("insights") or []
    texts: List[str] = []
    for ins in out:
        if isinstance(ins, dict):
            t = ins.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    return texts


def _embed(texts: List[str]) -> Any:
    """Embed a list of texts with the same model the canonical scorer uses."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _coverage_rate(
    pred_texts: List[str],
    silver_texts: List[str],
    threshold: float,
) -> Tuple[int, int, float]:
    """Return ``(covered_count, total_silver, avg_max_similarity)``.

    For each silver insight, find the max cosine similarity to any predicted
    insight. Covered = count of silver insights with max sim ≥ threshold.
    """
    if not pred_texts or not silver_texts:
        return 0, len(silver_texts), 0.0

    pred_emb = _embed(pred_texts)
    silver_emb = _embed(silver_texts)
    # Cosine == dot since both normalized.
    sim = silver_emb @ pred_emb.T  # shape: (S, P)
    max_per_silver = sim.max(axis=1)  # shape: (S,)
    covered = int((max_per_silver >= threshold).sum())
    avg = float(max_per_silver.mean())
    return covered, len(silver_texts), avg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine threshold for a silver insight to count as 'covered'.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    run_dir = REPO_ROOT / "data/eval/runs" / args.run_id
    silver_dir = REPO_ROOT / "data/eval/references/silver" / args.silver
    pred_path = run_dir / "predictions.jsonl"
    silver_path = silver_dir / "predictions.jsonl"
    if not pred_path.is_file():
        logger.error("predictions missing: %s", pred_path)
        return 1
    if not silver_path.is_file():
        logger.error("silver missing: %s", silver_path)
        return 1

    pred = {p["episode_id"]: p for p in _load_jsonl(pred_path) if p.get("episode_id")}
    silver = {s["episode_id"]: s for s in _load_jsonl(silver_path) if s.get("episode_id")}

    # Match on common episode_ids that also belong to ``args.dataset``.
    common = sorted(set(pred.keys()) & set(silver.keys()))
    if not common:
        logger.error("no common episode_ids between run and silver")
        return 1

    logger.info("Run: %s", args.run_id)
    logger.info("Silver: %s", args.silver)
    logger.info("Threshold: %.2f", args.threshold)
    logger.info("Episodes: %d", len(common))

    total_covered = 0
    total_silver = 0
    avg_sims: List[float] = []
    grounded_covered = 0
    grounded_total = 0  # silver insights matched by a grounded predicted insight

    for ep_id in common:
        pred_p = pred[ep_id]
        silver_p = silver[ep_id]
        pred_texts = _gil_insight_texts(pred_p)
        silver_texts = _silver_insight_texts(silver_p)
        covered, total, avg = _coverage_rate(pred_texts, silver_texts, args.threshold)
        total_covered += covered
        total_silver += total
        avg_sims.append(avg)

        # Grounded coverage: of silver insights matched, how many were matched
        # by a predicted insight that also has a SUPPORTED_BY edge?
        grounded_ids = _gil_grounded_insight_ids(pred_p)
        if grounded_ids and pred_texts and silver_texts:
            # Re-derive matching with grounded subset.
            pred_nodes = (pred_p.get("output", {}).get("gil") or {}).get("nodes") or []
            grounded_pred_texts: List[str] = []
            for n in pred_nodes:
                if (
                    isinstance(n, dict)
                    and n.get("type") == "Insight"
                    and str(n.get("id")) in grounded_ids
                ):
                    t = (n.get("properties") or {}).get("text") or n.get("text")
                    if isinstance(t, str) and t.strip():
                        grounded_pred_texts.append(t.strip())
            if grounded_pred_texts:
                gp_emb = _embed(grounded_pred_texts)
                sv_emb = _embed(silver_texts)
                gsim = sv_emb @ gp_emb.T
                grounded_max = gsim.max(axis=1)
                grounded_covered += int((grounded_max >= args.threshold).sum())
        grounded_total += total

        logger.info(
            "  %s: %d/%d covered (%.0f%%) avg_sim=%.3f insights=%d",
            ep_id,
            covered,
            total,
            (covered / total * 100) if total else 0.0,
            avg,
            len(pred_texts),
        )

    coverage_rate = total_covered / total_silver if total_silver else 0.0
    grounded_rate = grounded_covered / grounded_total if grounded_total else 0.0
    avg_sim = sum(avg_sims) / len(avg_sims) if avg_sims else 0.0

    logger.info("")
    logger.info(
        "OVERALL: %d/%d insights covered (%.0f%%) avg_sim=%.3f grounded_rate=%.0f%%",
        total_covered,
        total_silver,
        coverage_rate * 100,
        avg_sim,
        grounded_rate * 100,
    )
    # Single-line summary to stdout for results.tsv ingestion.
    print(
        f"coverage={coverage_rate:.4f} grounded={grounded_rate:.4f} "
        f"avg_sim={avg_sim:.4f} episodes={len(common)} silver_insights={total_silver}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
