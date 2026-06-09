"""Finale-tier runner — stratify, promote, judge, synthesize (#932).

Inputs:
    - A finale config (YAML) that defines the strata and how runs map into them.
    - A glob of qualifier run dirs (each with ``predictions.jsonl`` and
      ``metrics_vs_silver_opus47_smoke_v1.json``).

Output (under ``data/eval/runs/finale/<tag>/``):
    - ``promotion.json`` — top-3-per-stratum + capped finalists.
    - ``finalists.jsonl`` — one row per (finalist run, judge, episode, dim).
    - ``finale_report.json`` — per-finalist aggregate (mean score per dim,
      overall mean, contested flags vs. cross-check judge, total cost).
    - ``finale_report.md`` — human-readable summary with the top-3 verdicts
      per stratum and the methodology footer.

Cost guard: the runner tracks running USD against ``cost_cap_usd`` from the
config (default $50) and aborts mid-run if exceeded — partial results are
still persisted for the report so a budget-blown sweep leaves a trail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from podcast_scraper.evaluation.g_eval import (
    DIMENSIONS,
    score_summary,
    SummaryScore,
)
from podcast_scraper.evaluation.judges.base import JudgeUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types


@dataclass
class RunCandidate:
    """One qualifier run dir, with its metrics resolved for promotion."""

    run_id: str
    run_dir: Path
    stratum: str
    rouge_l_f1: float
    embedding_cosine: float
    coverage_ratio: float
    metrics_vs_silver_path: Path


@dataclass
class StratumPromotion:
    """Per-stratum promotion summary."""

    name: str
    leader_rouge_l: float
    floor: float  # 0.8 * leader_rouge_l
    promoted: List[str] = field(default_factory=list)  # run ids
    rejected: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FinalistAggregate:
    """Aggregate G-Eval result for one finalist across all episodes/judges."""

    run_id: str
    stratum: str
    primary_judge: str
    cross_judge: Optional[str]
    n_episodes: int
    primary_mean_per_dim: Dict[str, float] = field(default_factory=dict)
    cross_mean_per_dim: Dict[str, float] = field(default_factory=dict)
    primary_overall_mean: float = 0.0
    cross_overall_mean: Optional[float] = None
    agreement_rate: Optional[float] = None  # primary vs cross, on the same pairs
    contested: bool = False
    total_cost_usd: float = 0.0
    errors: int = 0


# ---------------------------------------------------------------------------
# Stratification & promotion


def _classify_stratum(run_id: str, stratum_map: List[Dict[str, Any]]) -> Optional[str]:
    """Match ``run_id`` against the first stratum whose ``match`` substring fits.

    The stratum config is intentionally minimal: an ordered list of
    ``{name, match: [substrings]}`` entries. First match wins so callers can
    put "cloud" rules before generic ollama rules without arbitration logic.

    Returns ``None`` if no stratum matches; the caller drops the run.
    """
    for s in stratum_map:
        for needle in s.get("match", []):
            if needle in run_id:
                return str(s["name"])
    return None


def load_run_candidate(
    run_dir: Path,
    *,
    stratum_map: List[Dict[str, Any]],
    metrics_filename: str = "metrics_vs_silver_opus47_smoke_v1.json",
) -> Optional[RunCandidate]:
    """Load one run dir into a :class:`RunCandidate`.

    Returns ``None`` if the run is missing ``predictions.jsonl``, missing the
    metrics file, or has no matching stratum. (The promotion stage filters
    these out silently — they show in the run log.)
    """
    preds = run_dir / "predictions.jsonl"
    metrics_path = run_dir / metrics_filename
    if not preds.is_file() or not metrics_path.is_file():
        logger.debug("Skip %s: missing predictions or metrics", run_dir.name)
        return None
    stratum = _classify_stratum(run_dir.name, stratum_map)
    if stratum is None:
        logger.debug("Skip %s: no matching stratum", run_dir.name)
        return None
    try:
        blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Skip %s: metrics JSON unparsable: %s", run_dir.name, exc)
        return None
    vs = blob.get("vs_reference") or {}
    rouge_l = vs.get("rougeL_f1")
    if rouge_l is None:
        logger.debug("Skip %s: no rougeL_f1 in metrics", run_dir.name)
        return None
    return RunCandidate(
        run_id=run_dir.name,
        run_dir=run_dir,
        stratum=stratum,
        rouge_l_f1=float(rouge_l),
        embedding_cosine=float(vs.get("embedding_cosine") or 0.0),
        coverage_ratio=float(vs.get("coverage_ratio") or 0.0),
        metrics_vs_silver_path=metrics_path,
    )


def promote_finalists(
    candidates: Iterable[RunCandidate],
    *,
    per_stratum_top_k: int = 3,
    floor_fraction: float = 0.8,
    overall_cap: int = 12,
    carte_blanche: Iterable[str] = (),
) -> tuple[List[RunCandidate], List[StratumPromotion]]:
    """Apply the #932 promotion rule.

    Per stratum:
        1. Rank by ``rouge_l_f1`` descending.
        2. Take the top ``per_stratum_top_k``.
        3. Drop any whose ``rouge_l_f1`` < ``floor_fraction * stratum_leader``.

    Then trim the union to ``overall_cap`` candidates by global ``rouge_l_f1``
    descending. Returns ``(finalists, per_stratum_promotion)``.

    ``carte_blanche``: list of substrings; any candidate whose ``run_id``
    contains one of these is force-promoted regardless of floor / top_k /
    overall_cap. Use for "include the current prod champion even if its
    ROUGE on the new silver fell below the floor" — since the whole reason
    for G-Eval is to bypass ROUGE bias, excluding a model on that biased
    metric is exactly the bias the finale is supposed to escape.
    """
    carte_blanche_terms = tuple(s for s in carte_blanche if s)
    by_stratum: Dict[str, List[RunCandidate]] = {}
    for c in candidates:
        by_stratum.setdefault(c.stratum, []).append(c)

    promoted: List[RunCandidate] = []
    summary: List[StratumPromotion] = []
    for name, runs in by_stratum.items():
        runs_sorted = sorted(runs, key=lambda r: r.rouge_l_f1, reverse=True)
        if not runs_sorted:
            continue
        leader = runs_sorted[0].rouge_l_f1
        floor = floor_fraction * leader
        stratum_promo = StratumPromotion(name=name, leader_rouge_l=leader, floor=floor)
        for rank, run in enumerate(runs_sorted[:per_stratum_top_k]):
            if run.rouge_l_f1 >= floor:
                promoted.append(run)
                stratum_promo.promoted.append(run.run_id)
            else:
                stratum_promo.rejected.append(
                    {
                        "run_id": run.run_id,
                        "reason": f"below_floor (rougeL={run.rouge_l_f1:.4f} < floor={floor:.4f})",
                        "rouge_l_f1": run.rouge_l_f1,
                    }
                )
        # Note runs ranked below per_stratum_top_k as 'not_top_k' for traceability.
        for rank, run in enumerate(runs_sorted[per_stratum_top_k:], start=per_stratum_top_k + 1):
            stratum_promo.rejected.append(
                {
                    "run_id": run.run_id,
                    "reason": f"not_top_{per_stratum_top_k} (rank={rank})",
                    "rouge_l_f1": run.rouge_l_f1,
                }
            )
        summary.append(stratum_promo)

    # Apply global cap.
    if len(promoted) > overall_cap:
        promoted = sorted(promoted, key=lambda r: r.rouge_l_f1, reverse=True)[:overall_cap]
        promoted_ids = {r.run_id for r in promoted}
        for s in summary:
            s.promoted = [r for r in s.promoted if r in promoted_ids]

    # Carte blanche — force-include candidates whose run_id matches any of the
    # configured substrings, even if they were dropped by floor / top_k / cap.
    # Bypasses overall_cap deliberately: the operator's intent is "I want to
    # see this model's G-Eval score regardless of ROUGE."
    if carte_blanche_terms:
        promoted_ids = {r.run_id for r in promoted}
        # Walk all candidates again (need full pool, not just the per-stratum
        # winners) — by_stratum has it.
        all_candidates = [c for cs in by_stratum.values() for c in cs]
        for cand in all_candidates:
            if cand.run_id in promoted_ids:
                continue
            if any(term in cand.run_id for term in carte_blanche_terms):
                promoted.append(cand)
                promoted_ids.add(cand.run_id)
                # Record on its stratum_promo for traceability.
                for s in summary:
                    if s.name == cand.stratum:
                        s.promoted.append(cand.run_id)
                        break
                # Remove the corresponding rejected entry if any (so the report
                # doesn't show both 'rejected: below_floor' and 'promoted'
                # at once).
                for s in summary:
                    if s.name == cand.stratum:
                        s.rejected = [r for r in s.rejected if r.get("run_id") != cand.run_id]
                        break

    return promoted, summary


# ---------------------------------------------------------------------------
# Episode-level orchestration


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Read a ``predictions.jsonl`` file into a list of dicts (one per episode)."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed prediction line in %s: %s", path, exc)
    return out


def _extract_summary(pred: Dict[str, Any]) -> str:
    """Pull the final summary string from a prediction dict.

    Mirrors ``autoresearch_track_a.summary_text_from_prediction`` so both
    tiers see the same text. Bundled mode where ``summary_final`` is a JSON
    blob is left as-is — the caller's prompt frames it explicitly.
    """
    out = pred.get("output") or {}
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for key in ("summary_final", "summary_long", "summary"):
            v = out.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def load_transcript(*, dataset_id: str, episode_id: str, eval_root: Path) -> Optional[str]:
    """Load one materialized transcript; return ``None`` if not found.

    Materialized transcripts live under
    ``<eval_root>/materialized/<dataset_id>/<episode_id>.txt``. The qualifier
    tier already uses these — we reuse them so the finale judge sees the
    same evidence the qualifier did.
    """
    candidate = eval_root / "materialized" / dataset_id / f"{episode_id}.txt"
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8")


def judge_finalist(
    *,
    finalist: RunCandidate,
    judge: Any,
    eval_root: Path,
    max_episodes: Optional[int] = None,
    cost_budget_remaining: Optional[float] = None,
) -> tuple[List[SummaryScore], float, int]:
    """Score every episode in ``finalist`` with ``judge``.

    Returns ``(per_episode_scores, total_cost_usd, n_errors)``. Stops early
    if ``cost_budget_remaining`` drops to <= 0 (preserves partial results).
    """
    predictions = load_predictions(finalist.run_dir / "predictions.jsonl")
    if max_episodes is not None:
        predictions = predictions[:max_episodes]

    results: List[SummaryScore] = []
    total_cost = 0.0
    errors = 0
    remaining = cost_budget_remaining
    for pred in predictions:
        episode_id = pred.get("episode_id")
        dataset_id = pred.get("dataset_id") or ""
        if not episode_id or not dataset_id:
            logger.warning(
                "Skipping prediction without episode_id/dataset_id in %s",
                finalist.run_id,
            )
            continue
        transcript = load_transcript(
            dataset_id=dataset_id, episode_id=episode_id, eval_root=eval_root
        )
        if transcript is None:
            logger.warning(
                "Skipping %s/%s: materialized transcript missing",
                finalist.run_id,
                episode_id,
            )
            continue
        summary = _extract_summary(pred)
        if not summary.strip():
            logger.warning("Skipping %s/%s: empty candidate summary", finalist.run_id, episode_id)
            continue
        try:
            score = score_summary(
                run_id=finalist.run_id,
                episode_id=str(episode_id),
                transcript=transcript,
                summary=summary,
                judge=judge,
            )
        except JudgeUnavailableError as exc:
            logger.error(
                "Judge fully unavailable on %s/%s: %s — aborting finalist",
                finalist.run_id,
                episode_id,
                exc,
            )
            errors += 1
            break
        results.append(score)
        total_cost += score.total_cost_usd
        errors += len(score.errors)
        if remaining is not None:
            remaining -= score.total_cost_usd
            if remaining <= 0:
                logger.warning(
                    "Cost budget exhausted mid-finalist %s after %d episodes",
                    finalist.run_id,
                    len(results),
                )
                break
    return results, total_cost, errors


# ---------------------------------------------------------------------------
# Aggregation


def aggregate_finalist(
    *,
    finalist: RunCandidate,
    primary_judge_model: str,
    primary_scores: List[SummaryScore],
    cross_judge_model: Optional[str] = None,
    cross_scores: Optional[List[SummaryScore]] = None,
) -> FinalistAggregate:
    """Reduce per-episode scores to a single :class:`FinalistAggregate`.

    Per-dimension means are computed on the episodes where the judge returned
    a parseable score (errors are excluded from the numerator AND
    denominator). The ``contested`` flag is set if the cross-judge's overall
    mean differs from primary's by > 0.5 on the 1-5 scale (mirrors the
    spirit of the track_a 0.25 threshold on the 0-1 scale).
    """

    def _per_dim_mean(scores: List[SummaryScore], dim: str) -> Optional[float]:
        vals = [s.per_dimension[dim].score for s in scores if dim in s.per_dimension]
        if not vals:
            return None
        return sum(vals) / len(vals)

    primary_per_dim = {d: _per_dim_mean(primary_scores, d) for d in DIMENSIONS}
    primary_overall_means = [m for m in primary_per_dim.values() if m is not None]
    primary_overall_mean = (
        sum(primary_overall_means) / len(primary_overall_means) if primary_overall_means else 0.0
    )
    total_cost = sum(s.total_cost_usd for s in primary_scores)
    n_err = sum(len(s.errors) for s in primary_scores)

    cross_per_dim: Dict[str, float] = {}
    cross_overall: Optional[float] = None
    agree_rate: Optional[float] = None
    contested = False
    if cross_scores is not None:
        cross_per_dim_raw = {d: _per_dim_mean(cross_scores, d) for d in DIMENSIONS}
        cross_per_dim = {d: v for d, v in cross_per_dim_raw.items() if v is not None}
        if cross_per_dim:
            cross_overall = sum(cross_per_dim.values()) / len(cross_per_dim)
            contested = abs(primary_overall_mean - cross_overall) > 0.5
        # Pair-level agreement on (episode, dim) pairs where both judges scored.
        agree_rate = _pairwise_agreement_rate(primary_scores, cross_scores)
        total_cost += sum(s.total_cost_usd for s in cross_scores)
        n_err += sum(len(s.errors) for s in cross_scores)

    return FinalistAggregate(
        run_id=finalist.run_id,
        stratum=finalist.stratum,
        primary_judge=primary_judge_model,
        cross_judge=cross_judge_model,
        n_episodes=len(primary_scores),
        primary_mean_per_dim={d: v for d, v in primary_per_dim.items() if v is not None},
        cross_mean_per_dim=cross_per_dim,
        primary_overall_mean=primary_overall_mean,
        cross_overall_mean=cross_overall,
        agreement_rate=agree_rate,
        contested=contested,
        total_cost_usd=total_cost,
        errors=n_err,
    )


def _pairwise_agreement_rate(
    primary: List[SummaryScore],
    cross: List[SummaryScore],
    *,
    tolerance: int = 1,
) -> Optional[float]:
    """Episode-by-episode, dimension-by-dimension agreement (exact-or-adjacent).

    Falls back to ``None`` when no (episode, dim) pair has both scores.
    """
    cross_by_ep = {s.episode_id: s for s in cross}
    total = 0
    agree = 0
    for ps in primary:
        cs = cross_by_ep.get(ps.episode_id)
        if cs is None:
            continue
        for dim, p_dim in ps.per_dimension.items():
            c_dim = cs.per_dimension.get(dim)
            if c_dim is None:
                continue
            total += 1
            if abs(p_dim.score - c_dim.score) <= tolerance:
                agree += 1
    if total == 0:
        return None
    return agree / total


# ---------------------------------------------------------------------------
# Report rendering


def render_report_markdown(
    *,
    tag: str,
    aggregates: List[FinalistAggregate],
    promotion: List[StratumPromotion],
    total_cost_usd: float,
    cost_cap_usd: float,
) -> str:
    """Render the finale Markdown report (top-3 per stratum + cost summary)."""
    lines: List[str] = []
    lines.append(f"# Finale sweep — `{tag}`")
    lines.append("")
    lines.append(f"_Total spend: **${total_cost_usd:.2f}** (cap ${cost_cap_usd:.2f})_")
    lines.append("")
    lines.append("## Verdicts by stratum")
    lines.append("")
    by_stratum: Dict[str, List[FinalistAggregate]] = {}
    for a in aggregates:
        by_stratum.setdefault(a.stratum, []).append(a)
    for stratum_name, items in by_stratum.items():
        items_sorted = sorted(items, key=lambda a: a.primary_overall_mean, reverse=True)
        lines.append(f"### {stratum_name}")
        lines.append("")
        lines.append("| Rank | Run | Faith | Cov | Coh | Flu | Mean | Contested? |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |")
        for rank, agg in enumerate(items_sorted[:3], start=1):
            d = agg.primary_mean_per_dim
            lines.append(
                "| {rank} | `{run}` | {f:.2f} | {c:.2f} | {h:.2f} | {l:.2f} "
                "| **{m:.2f}** | {x} |".format(
                    rank=rank,
                    run=agg.run_id,
                    f=d.get("faithfulness", 0.0),
                    c=d.get("coverage", 0.0),
                    h=d.get("coherence", 0.0),
                    l=d.get("fluency", 0.0),
                    m=agg.primary_overall_mean,
                    x="!" if agg.contested else "",
                )
            )
        lines.append("")
    lines.append("## Promotion details")
    lines.append("")
    for p in promotion:
        lines.append(f"### {p.name}")
        lines.append(f"- Leader ROUGE-L: {p.leader_rouge_l:.4f} — floor: {p.floor:.4f}")
        lines.append(f"- Promoted: {', '.join(p.promoted) if p.promoted else '(none)'}")
        if p.rejected:
            lines.append("- Rejected:")
            for r in p.rejected:
                lines.append(f"  - `{r['run_id']}` — {r['reason']}")
        lines.append("")
    return "\n".join(lines)


def write_finale_artifacts(
    *,
    output_dir: Path,
    tag: str,
    promotion: List[StratumPromotion],
    aggregates: List[FinalistAggregate],
    per_episode_records: List[Dict[str, Any]],
    total_cost_usd: float,
    cost_cap_usd: float,
) -> Dict[str, Path]:
    """Persist promotion.json, finalists.jsonl, finale_report.{json,md}.

    Returns a dict of artifact name → path for the caller to log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    promotion_path = output_dir / "promotion.json"
    promotion_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "strata": [
                    {
                        "name": p.name,
                        "leader_rouge_l": p.leader_rouge_l,
                        "floor": p.floor,
                        "promoted": p.promoted,
                        "rejected": p.rejected,
                    }
                    for p in promotion
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    finalists_path = output_dir / "finalists.jsonl"
    with finalists_path.open("w", encoding="utf-8") as fh:
        for rec in per_episode_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report_json_path = output_dir / "finale_report.json"
    report_json_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "total_cost_usd": total_cost_usd,
                "cost_cap_usd": cost_cap_usd,
                "aggregates": [
                    {
                        "run_id": a.run_id,
                        "stratum": a.stratum,
                        "primary_judge": a.primary_judge,
                        "cross_judge": a.cross_judge,
                        "n_episodes": a.n_episodes,
                        "primary_mean_per_dim": a.primary_mean_per_dim,
                        "cross_mean_per_dim": a.cross_mean_per_dim,
                        "primary_overall_mean": a.primary_overall_mean,
                        "cross_overall_mean": a.cross_overall_mean,
                        "agreement_rate": a.agreement_rate,
                        "contested": a.contested,
                        "total_cost_usd": a.total_cost_usd,
                        "errors": a.errors,
                    }
                    for a in aggregates
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_md_path = output_dir / "finale_report.md"
    report_md_path.write_text(
        render_report_markdown(
            tag=tag,
            aggregates=aggregates,
            promotion=promotion,
            total_cost_usd=total_cost_usd,
            cost_cap_usd=cost_cap_usd,
        ),
        encoding="utf-8",
    )
    return {
        "promotion": promotion_path,
        "finalists": finalists_path,
        "report_json": report_json_path,
        "report_md": report_md_path,
    }


__all__ = [
    "FinalistAggregate",
    "RunCandidate",
    "StratumPromotion",
    "aggregate_finalist",
    "judge_finalist",
    "load_predictions",
    "load_run_candidate",
    "load_transcript",
    "promote_finalists",
    "render_report_markdown",
    "write_finale_artifacts",
]
