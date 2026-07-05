"""Regression guard for the #382 transformers v5 parity gate.

Re-runs the parity comparison committed at ``data/eval/runs/v5_parity_2026-07-05.json``
against a fresh capture (summarizer baseline + QA span reference JSONL) and
asserts the current code still meets the shipped thresholds:

- Summarizer: min per-episode ROUGE-L(pre, post) >= 0.95 (identical
  checkpoints + deterministic seeds ⇒ byte-identical output on the
  shipped baseline).
- Extractive QA: answer-text match rate >= 0.85 across the fixture set.

Marked as ``nightly + ml_models + slow`` — it loads real BART + roberta
checkpoints from the local HF cache with ``local_files_only=True`` and
runs a full summarize + QA sweep (~4 min wall on a warm cache). Won't
run in the fast unit CI; run explicitly with:

    pytest tests/integration/test_v5_parity_regression.py -m nightly

Under CI, this should be scheduled on the nightly matrix so any future
change to :class:`SummaryModel` / :class:`QAEvidenceBackend` /
:class:`HFEvidenceBackend` / :class:`HFSeq2SeqBackend` that drifts
output is caught before it merges.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.ml_models,
    pytest.mark.slow,
    pytest.mark.e2e,
]

REPO = Path(__file__).resolve().parents[2]
PRE_BASELINE = REPO / "data/eval/baselines/baseline_ml_bart_authority_smoke_v5_pre"
POST_BASELINE = REPO / "data/eval/baselines/baseline_ml_bart_authority_smoke_v5_post"
PRE_QA = REPO / "data/eval/references/qa_baseline_v5_pre.jsonl"
QA_CAPTURE = REPO / "scripts/dev/capture_qa_baseline.py"
COMPARE = REPO / "scripts/eval/compare_v5_parity.py"


@pytest.fixture(scope="module")
def fresh_qa_baseline(tmp_path_factory) -> Path:
    """Capture a fresh QA baseline against the current code — proves the
    QAEvidenceBackend path is still emitting the same spans it did at
    the #382 parity gate."""
    out = tmp_path_factory.mktemp("qa_parity") / "qa_baseline_current.jsonl"
    result = subprocess.run(
        [sys.executable, str(QA_CAPTURE), "--out", str(out)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"QA capture failed:\n{result.stderr}"
    assert out.exists() and out.stat().st_size > 0
    return Path(out)


def test_qa_parity_regression_text_match_stable(fresh_qa_baseline: Path) -> None:
    """Fresh QA capture matches the shipped v5_post baseline on answer text.

    100% match expected — same code, same fixtures, deterministic device=cpu.
    Any drop here means the QA post-processing regressed.
    """
    post_qa = REPO / "data/eval/references/qa_baseline_v5_post.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--pre-baseline",
            str(POST_BASELINE),  # v5_post is our "reference" for regression
            "--post-baseline",
            str(POST_BASELINE),  # summarizer parity is trivially 1.0 here
            "--pre-qa",
            str(post_qa),
            "--post-qa",
            str(fresh_qa_baseline),
            "--out",
            str(fresh_qa_baseline.parent / "regression_report.json"),
            "--qa-match-threshold",
            "1.0",  # same code → 100% expected
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"QA regression detected — fresh capture differs from v5_post baseline.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    report = json.loads((fresh_qa_baseline.parent / "regression_report.json").read_text())
    assert report["qa"]["text_match_rate"] == 1.0, (
        f"QA text match dropped: {report['qa']['text_match_rate']} < 1.0 "
        f"(mismatches: {[p['id'] for p in report['qa']['per_pair'] if not p.get('text_match')]})"
    )


def test_shipped_parity_report_still_passes() -> None:
    """The frozen parity report committed at Phase 7 must still pass its own
    thresholds when re-scored — protects against a change to the compare
    script that would loosen the gate silently."""
    frozen = REPO / "data/eval/runs/v5_parity_2026-07-05.json"
    assert frozen.exists(), "shipped parity report missing"

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--pre-baseline",
            str(PRE_BASELINE),
            "--post-baseline",
            str(POST_BASELINE),
            "--pre-qa",
            str(PRE_QA),
            "--post-qa",
            str(REPO / "data/eval/references/qa_baseline_v5_post.jsonl"),
            "--out",
            str(REPO / "/tmp/parity_recheck.json"),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Re-scored parity report FAILS its own thresholds — thresholds have "
        f"drifted or a comparator bug was introduced.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
