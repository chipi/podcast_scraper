"""Unit tests for the finale runner (#932).

Covers:

- ``_classify_stratum`` via ``load_run_candidate`` (first match wins, ordered)
- ``promote_finalists`` applies top-K, floor, and global cap correctly
- ``aggregate_finalist`` computes per-dim means and the contested flag
- ``_pairwise_agreement_rate`` handles missing-pair and mismatched dim cases
- Markdown report renders top-3 per stratum + cost line
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.evaluation.finale_runner import (
    _pairwise_agreement_rate,
    aggregate_finalist,
    FinalistAggregate,
    load_run_candidate,
    promote_finalists,
    render_report_markdown,
    RunCandidate,
    StratumPromotion,
)
from podcast_scraper.evaluation.g_eval import DimensionScore, SummaryScore

# ---------------------------------------------------------------------------
# load_run_candidate / classify


def _write_run(
    tmp_path: Path,
    name: str,
    *,
    rouge_l: float = 0.5,
    coverage: float = 0.95,
    cosine: float = 0.8,
    write_metrics: bool = True,
    write_preds: bool = True,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    if write_preds:
        (run_dir / "predictions.jsonl").write_text(
            '{"episode_id":"p01_e01","dataset_id":"curated_5feeds_smoke_v1",'
            '"output":{"summary_final":"s"}}\n',
            encoding="utf-8",
        )
    if write_metrics:
        (run_dir / "metrics_vs_silver_opus47_smoke_v1.json").write_text(
            json.dumps(
                {
                    "reference_id": "silver_opus47_smoke_v1",
                    "run_id": name,
                    "vs_reference": {
                        "rougeL_f1": rouge_l,
                        "coverage_ratio": coverage,
                        "embedding_cosine": cosine,
                    },
                }
            ),
            encoding="utf-8",
        )
    return run_dir


_STRATA = [
    {"name": "cloud", "match": ["autoresearch_prompt_anthropic_"]},
    {"name": "dgx_le_40b", "match": ["autoresearch_prompt_ollama_qwen35_35b_"]},
    {"name": "mbp_le_14b", "match": ["autoresearch_prompt_ollama_"]},
]


def test_load_run_candidate_assigns_first_matching_stratum(tmp_path: Path) -> None:
    """Strata are ordered: anthropic_ -> cloud (not the ollama_ catch-all)."""
    rd = _write_run(tmp_path, "autoresearch_prompt_anthropic_smoke_paragraph_v1_x")
    c = load_run_candidate(rd, stratum_map=_STRATA)
    assert c is not None
    assert c.stratum == "cloud"


def test_load_run_candidate_falls_through_to_catchall(tmp_path: Path) -> None:
    """A small ollama model hits the mbp catch-all, not dgx."""
    rd = _write_run(tmp_path, "autoresearch_prompt_ollama_qwen35_9b_smoke_paragraph_v1_x")
    c = load_run_candidate(rd, stratum_map=_STRATA)
    assert c is not None
    assert c.stratum == "mbp_le_14b"


def test_load_run_candidate_specific_stratum_wins_over_catchall(tmp_path: Path) -> None:
    """qwen35_35b matches dgx_le_40b before the generic ollama catch-all."""
    rd = _write_run(tmp_path, "autoresearch_prompt_ollama_qwen35_35b_smoke_paragraph_v1_x")
    c = load_run_candidate(rd, stratum_map=_STRATA)
    assert c is not None
    assert c.stratum == "dgx_le_40b"


def test_load_run_candidate_skips_when_no_stratum_matches(tmp_path: Path) -> None:
    rd = _write_run(tmp_path, "unknown_run_dir")
    assert load_run_candidate(rd, stratum_map=_STRATA) is None


def test_load_run_candidate_skips_when_metrics_missing(tmp_path: Path) -> None:
    rd = _write_run(tmp_path, "autoresearch_prompt_anthropic_x", write_metrics=False)
    assert load_run_candidate(rd, stratum_map=_STRATA) is None


def test_load_run_candidate_skips_when_predictions_missing(tmp_path: Path) -> None:
    rd = _write_run(tmp_path, "autoresearch_prompt_anthropic_x", write_preds=False)
    assert load_run_candidate(rd, stratum_map=_STRATA) is None


# ---------------------------------------------------------------------------
# promote_finalists


def _candidate(name: str, rouge: float, stratum: str = "cloud") -> RunCandidate:
    return RunCandidate(
        run_id=name,
        run_dir=Path("/nonexistent"),
        stratum=stratum,
        rouge_l_f1=rouge,
        embedding_cosine=0.0,
        coverage_ratio=0.0,
        metrics_vs_silver_path=Path("/nonexistent.json"),
    )


def test_promote_finalists_top_k_and_floor() -> None:
    """Top-3 promoted; the 0.4-leader candidate falls below 0.8*0.9 = 0.72 floor."""
    cands = [
        _candidate("a", 0.9),
        _candidate("b", 0.8),
        _candidate("c", 0.75),
        _candidate("d", 0.4),  # below floor 0.72
    ]
    promoted, summary = promote_finalists(
        cands, per_stratum_top_k=3, floor_fraction=0.8, overall_cap=10
    )
    promoted_ids = {p.run_id for p in promoted}
    assert promoted_ids == {"a", "b", "c"}
    assert summary[0].leader_rouge_l == 0.9
    assert summary[0].floor == pytest.approx(0.72)
    # d shouldn't appear as promoted; it's beyond top_k=3 so "not_top_3"
    assert "d" not in summary[0].promoted


def test_promote_finalists_drops_top_k_below_floor() -> None:
    """A top-3 candidate that falls below floor is rejected, not promoted."""
    cands = [
        _candidate("a", 1.0),
        _candidate("b", 0.9),
        _candidate("c", 0.5),  # 0.5 < 0.8*1.0 = 0.8 floor → rejected
    ]
    promoted, summary = promote_finalists(
        cands, per_stratum_top_k=3, floor_fraction=0.8, overall_cap=10
    )
    assert {p.run_id for p in promoted} == {"a", "b"}
    rejected_ids = {r["run_id"] for r in summary[0].rejected}
    assert "c" in rejected_ids


def test_promote_finalists_overall_cap_trims_by_global_rouge() -> None:
    """Two strata × 3 promoted each = 6; cap at 4 keeps the global top 4."""
    cands = [
        _candidate("c1", 0.95, stratum="cloud"),
        _candidate("c2", 0.93, stratum="cloud"),
        _candidate("c3", 0.91, stratum="cloud"),
        _candidate("d1", 0.90, stratum="dgx_le_40b"),
        _candidate("d2", 0.88, stratum="dgx_le_40b"),
        _candidate("d3", 0.86, stratum="dgx_le_40b"),
    ]
    promoted, _ = promote_finalists(cands, per_stratum_top_k=3, floor_fraction=0.5, overall_cap=4)
    # Top 4 globally: c1, c2, c3, d1
    assert {p.run_id for p in promoted} == {"c1", "c2", "c3", "d1"}


# ---------------------------------------------------------------------------
# aggregate_finalist + agreement rate


def _summary(run_id: str, ep: str, scores: dict[str, int], cost: float = 0.01) -> SummaryScore:
    s = SummaryScore(run_id=run_id, episode_id=ep, judge_model="fake")
    for dim, val in scores.items():
        s.per_dimension[dim] = DimensionScore(
            dimension=dim,
            score=val,
            explanation="",
            judge_model="fake",
            cost_usd=cost / len(scores),
            prompt_tokens=100,
            completion_tokens=20,
        )
    s.total_cost_usd = cost
    return s


def _runc() -> RunCandidate:
    return RunCandidate(
        run_id="run-1",
        run_dir=Path("/tmp/run-1"),
        stratum="cloud",
        rouge_l_f1=0.5,
        embedding_cosine=0.7,
        coverage_ratio=0.9,
        metrics_vs_silver_path=Path("/tmp/metrics.json"),
    )


def test_aggregate_finalist_per_dim_means_and_overall() -> None:
    """Per-dim means + overall mean across episodes from the primary judge."""
    primary = [
        _summary(
            "run-1",
            "p01_e01",
            {"faithfulness": 5, "coverage": 4, "coherence": 3, "fluency": 2},
        ),
        _summary(
            "run-1",
            "p02_e01",
            {"faithfulness": 4, "coverage": 4, "coherence": 4, "fluency": 4},
        ),
    ]
    agg = aggregate_finalist(
        finalist=_runc(),
        primary_judge_model="claude-sonnet-4-6",
        primary_scores=primary,
    )
    assert agg.n_episodes == 2
    assert agg.primary_mean_per_dim["faithfulness"] == pytest.approx(4.5)
    assert agg.primary_mean_per_dim["fluency"] == pytest.approx(3.0)
    # Overall mean is the mean of the four per-dim means
    assert agg.primary_overall_mean == pytest.approx((4.5 + 4 + 3.5 + 3) / 4)
    assert agg.cross_judge is None
    assert agg.cross_overall_mean is None


def test_aggregate_finalist_contested_when_judges_diverge_more_than_half_point() -> None:
    """primary overall=4.0, cross overall=4.6 → diff > 0.5 → contested."""
    primary = [
        _summary("r", "p01_e01", {"faithfulness": 4, "coverage": 4, "coherence": 4, "fluency": 4})
    ]
    cross = [
        _summary("r", "p01_e01", {"faithfulness": 5, "coverage": 5, "coherence": 4, "fluency": 5})
    ]
    agg = aggregate_finalist(
        finalist=_runc(),
        primary_judge_model="claude-sonnet-4-6",
        primary_scores=primary,
        cross_judge_model="gemini-2.5-pro",
        cross_scores=cross,
    )
    assert agg.contested is True
    # Agreement on the 4 pairs: 4-5(adj OK), 4-5(adj OK), 4-4(exact), 4-5(adj OK) → 4/4 at tol=1
    assert agg.agreement_rate == pytest.approx(1.0)


def test_aggregate_finalist_uncontested_within_half_point() -> None:
    primary = [
        _summary("r", "p01_e01", {"faithfulness": 4, "coverage": 4, "coherence": 4, "fluency": 4})
    ]
    cross = [
        _summary("r", "p01_e01", {"faithfulness": 4, "coverage": 4, "coherence": 5, "fluency": 4})
    ]
    agg = aggregate_finalist(
        finalist=_runc(),
        primary_judge_model="claude-sonnet-4-6",
        primary_scores=primary,
        cross_judge_model="gemini-2.5-pro",
        cross_scores=cross,
    )
    # primary overall = 4.0, cross overall = 4.25 → diff 0.25 ≤ 0.5
    assert agg.contested is False


def test_pairwise_agreement_rate_handles_missing_episodes() -> None:
    """Cross judge missing an episode → that episode's pairs excluded."""
    primary = [
        _summary("r", "p01_e01", {"faithfulness": 4, "coverage": 4}),
        _summary("r", "p02_e01", {"faithfulness": 5}),
    ]
    cross = [
        _summary("r", "p01_e01", {"faithfulness": 4, "coverage": 5}),
    ]
    rate = _pairwise_agreement_rate(primary, cross, tolerance=1)
    # Only p01_e01 has both → 2 pairs, both within tolerance → 1.0
    assert rate == pytest.approx(1.0)


def test_pairwise_agreement_rate_no_overlap_returns_none() -> None:
    rate = _pairwise_agreement_rate(
        [_summary("r", "p01_e01", {"faithfulness": 4})],
        [_summary("r", "p99_e99", {"faithfulness": 4})],
    )
    assert rate is None


# ---------------------------------------------------------------------------
# Markdown report rendering


def test_render_report_markdown_shows_top_3_per_stratum_and_cost() -> None:
    """Markdown contains the run_ids, dimension scores, cost line, and stratum sections."""
    aggregates = [
        FinalistAggregate(
            run_id="cloud_A",
            stratum="cloud",
            primary_judge="claude-sonnet-4-6",
            cross_judge=None,
            n_episodes=5,
            primary_mean_per_dim={
                "faithfulness": 4.8,
                "coverage": 4.6,
                "coherence": 4.5,
                "fluency": 4.7,
            },
            primary_overall_mean=4.65,
            total_cost_usd=1.2,
        ),
        FinalistAggregate(
            run_id="dgx_X",
            stratum="dgx_le_40b",
            primary_judge="claude-sonnet-4-6",
            cross_judge=None,
            n_episodes=5,
            primary_mean_per_dim={
                "faithfulness": 4.0,
                "coverage": 3.8,
                "coherence": 3.7,
                "fluency": 4.1,
            },
            primary_overall_mean=3.9,
            total_cost_usd=1.0,
        ),
    ]
    promotion = [
        StratumPromotion(name="cloud", leader_rouge_l=0.85, floor=0.68, promoted=["cloud_A"]),
        StratumPromotion(name="dgx_le_40b", leader_rouge_l=0.55, floor=0.44, promoted=["dgx_X"]),
    ]
    md = render_report_markdown(
        tag="finale_smoke",
        aggregates=aggregates,
        promotion=promotion,
        total_cost_usd=2.2,
        cost_cap_usd=50.0,
    )
    assert "Finale sweep — `finale_smoke`" in md
    assert "Total spend: **$2.20**" in md
    assert "### cloud" in md
    assert "### dgx_le_40b" in md
    assert "`cloud_A`" in md
    assert "`dgx_X`" in md
    assert "Promotion details" in md
