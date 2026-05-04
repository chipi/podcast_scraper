"""Smoke tests for the GIL evidence bundling autoresearch scorer (#698 Phase 4).

Validates ``score.py`` reads metrics.json + predictions.jsonl correctly and
emits a sane scalar. Quality-gate exits are exercised separately so a
regression doesn't slip through.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORE_PY = REPO_ROOT / "autoresearch/gil_evidence_bundling/eval/score.py"


def _write_run_dir(
    base: Path,
    *,
    metrics: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (base / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(p) + "\n")
    return base


def _gil_payload(
    insights: int,
    grounded_ids: List[str],
    nli_scores: List[float],
) -> Dict[str, Any]:
    nodes = [{"id": f"i{k}", "type": "Insight"} for k in range(insights)] + [
        {"id": f"q{k}", "type": "Quote"} for k in range(len(grounded_ids))
    ]
    edges = []
    for k, iid in enumerate(grounded_ids):
        edges.append(
            {
                "type": "SUPPORTED_BY",
                "source": iid,
                "target": f"q{k}",
                "nli_score": nli_scores[k] if k < len(nli_scores) else None,
            }
        )
    return {"nodes": nodes, "edges": edges}


def _run_score(baseline_dir: Path, variant_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCORE_PY),
            "--baseline",
            str(baseline_dir),
            "--variant",
            str(variant_dir),
            "--log-level",
            "ERROR",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def baseline_run(tmp_path: Path) -> Path:
    return _write_run_dir(
        tmp_path / "baseline",
        metrics={
            "run_duration_seconds": 600.0,
            "episodes_with_gil": 10,
            "llm_gi_extract_quotes_cost_usd": 0.50,
            "llm_gi_score_entailment_cost_usd": 0.80,
            "llm_gi_extract_quotes_calls": 14,
            "llm_gi_score_entailment_calls": 62,
            "llm_gi_extract_quotes_input_tokens": 50_000,
            "llm_gi_score_entailment_input_tokens": 60_000,
        },
        predictions=[
            {
                "episode_id": "ep1",
                "output": {
                    "gil": _gil_payload(
                        insights=4,
                        grounded_ids=["i0", "i1", "i2", "i3"],
                        nli_scores=[0.9, 0.85, 0.8, 0.75],
                    )
                },
            }
        ],
    )


def test_score_emits_scalar_for_clear_win(baseline_run: Path, tmp_path: Path) -> None:
    """Variant with cheaper cost + same grounding → positive score, exit 0."""
    variant = _write_run_dir(
        tmp_path / "variant",
        metrics={
            "run_duration_seconds": 300.0,  # half the wall-clock
            "episodes_with_gil": 10,
            "llm_gi_extract_quotes_cost_usd": 0.10,
            "llm_gi_score_entailment_cost_usd": 0.10,
            "llm_gi_extract_quotes_calls": 1,
            "llm_gi_score_entailment_calls": 5,
            "llm_gi_extract_quotes_input_tokens": 10_000,
            "llm_gi_score_entailment_input_tokens": 15_000,
        },
        predictions=[
            {
                "episode_id": "ep1",
                "output": {
                    "gil": _gil_payload(
                        insights=4,
                        grounded_ids=["i0", "i1", "i2", "i3"],
                        nli_scores=[0.88, 0.84, 0.82, 0.80],
                    )
                },
            }
        ],
    )
    result = _run_score(baseline_run, variant)
    assert result.returncode == 0, result.stderr
    score = float(result.stdout.strip())
    # 0.5 * (1 - 0.20/1.30) + 0.3 * 1.0 + 0.2 * (1 - 300/600) = 0.5 * 0.846 + 0.3 + 0.1 = 0.823
    assert score > 0.5
    assert score <= 1.0


def test_score_aborts_on_grounding_regression(baseline_run: Path, tmp_path: Path) -> None:
    """Grounding drop > 5pp absolute → exit non-zero with GATE FAIL message."""
    # Baseline grounding is 4/4 = 1.0. Variant: 1/4 = 0.25 (drop 0.75).
    variant = _write_run_dir(
        tmp_path / "variant_bad",
        metrics={
            "run_duration_seconds": 300.0,
            "episodes_with_gil": 10,
            "llm_gi_extract_quotes_cost_usd": 0.10,
            "llm_gi_score_entailment_cost_usd": 0.10,
            "llm_gi_extract_quotes_calls": 1,
            "llm_gi_score_entailment_calls": 5,
        },
        predictions=[
            {
                "episode_id": "ep1",
                "output": {
                    "gil": _gil_payload(
                        insights=4,
                        grounded_ids=["i0"],
                        nli_scores=[0.9],
                    )
                },
            }
        ],
    )
    result = _run_score(baseline_run, variant)
    assert result.returncode != 0
    assert "GATE FAIL" in result.stderr or "GATE FAIL" in result.stdout


def test_score_aborts_on_high_fallback_rate(baseline_run: Path, tmp_path: Path) -> None:
    """Fallback rate > 20% → exit non-zero."""
    variant = _write_run_dir(
        tmp_path / "variant_fallback",
        metrics={
            "run_duration_seconds": 300.0,
            "episodes_with_gil": 10,
            "llm_gi_extract_quotes_cost_usd": 0.10,
            "llm_gi_score_entailment_cost_usd": 0.10,
            "llm_gi_extract_quotes_calls": 1,
            "llm_gi_score_entailment_calls": 5,
            # 5 bundled calls + 5 fallbacks = 50% fallback rate.
            "gi_evidence_extract_quotes_bundled_calls": 5,
            "gi_evidence_extract_quotes_bundled_fallbacks": 5,
        },
        predictions=[
            {
                "episode_id": "ep1",
                "output": {
                    "gil": _gil_payload(
                        insights=4,
                        grounded_ids=["i0", "i1", "i2", "i3"],
                        nli_scores=[0.9, 0.85, 0.8, 0.75],
                    )
                },
            }
        ],
    )
    result = _run_score(baseline_run, variant)
    assert result.returncode != 0
    assert "fallback" in (result.stderr + result.stdout).lower()


def test_score_aborts_on_token_explosion(baseline_run: Path, tmp_path: Path) -> None:
    """Input tokens/ep > 50k → exit non-zero."""
    variant = _write_run_dir(
        tmp_path / "variant_tokens",
        metrics={
            "run_duration_seconds": 300.0,
            "episodes_with_gil": 10,
            "llm_gi_extract_quotes_cost_usd": 0.10,
            "llm_gi_score_entailment_cost_usd": 0.10,
            "llm_gi_extract_quotes_calls": 1,
            "llm_gi_score_entailment_calls": 5,
            "llm_gi_extract_quotes_input_tokens": 600_000,  # 60k/ep
            "llm_gi_score_entailment_input_tokens": 0,
        },
        predictions=[
            {
                "episode_id": "ep1",
                "output": {
                    "gil": _gil_payload(
                        insights=4,
                        grounded_ids=["i0", "i1", "i2", "i3"],
                        nli_scores=[0.9, 0.85, 0.8, 0.75],
                    )
                },
            }
        ],
    )
    result = _run_score(baseline_run, variant)
    assert result.returncode != 0
    assert "tokens" in (result.stderr + result.stdout).lower()
