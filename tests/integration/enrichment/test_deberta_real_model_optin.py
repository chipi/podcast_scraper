"""Optional real-model integration test for ``DeBERTaNliScorer``.

Marked ``ml_models`` so it only runs when the operator opts in:

    .venv/bin/pytest -m ml_models tests/integration/enrichment/test_deberta_real_model_optin.py

NOT run in CI per ``[[feedback_no_llm_in_ci]]``. The default test suite
deselects the ``ml_models`` marker; only the explicit operator-side
invocation downloads the ~80 MB model and runs predict.

Why an opt-in test exists: the DeBERTaNliScorer is the only chunk-4
scorer that ever loads a real model — its load + predict paths were
only exercised via mypy / coverage-deferred mocks until this test
landed. The marker keeps CI fast while letting operators verify the
scorer end-to-end before shipping a corpus.

To run:
    pip install -e '.[ml]'
    pytest -m ml_models -k deberta_real_model_optin -s
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = [pytest.mark.ml_models, pytest.mark.integration]


def test_deberta_loads_and_scores_a_contradiction_pair() -> None:
    """End-to-end: load model, score a known-contradiction pair, assert
    contradiction probability > neutral + entailment.
    """
    pytest.importorskip(
        "sentence_transformers",
        reason="install [ml] extra (sentence-transformers) to run this test",
    )
    from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer

    scorer = DeBERTaNliScorer()
    score = asyncio.run(
        scorer.score(
            "AI safety regulation will slow capability research without preventing risks.",
            "AI safety regulation is essential for preventing catastrophic risks.",
        )
    )
    # The two statements directly disagree on AI-safety regulation's utility.
    # On the DeBERTa-v3-small NLI head, that should fall in the contradiction
    # bucket. Don't assert a tight probability bound — the model can drift
    # across versions — just assert the ordering: contradiction > neutral
    # and contradiction > entailment.
    assert score.contradiction > score.neutral, score
    assert score.contradiction > score.entailment, score


def test_deberta_loads_and_scores_an_entailment_pair() -> None:
    """End-to-end: paraphrase pair should score entailment-dominant."""
    pytest.importorskip(
        "sentence_transformers",
        reason="install [ml] extra (sentence-transformers) to run this test",
    )
    from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer

    scorer = DeBERTaNliScorer()
    score = asyncio.run(
        scorer.score(
            "Demographic shifts will pressure entitlement spending.",
            "Demographic shifts will pressure entitlement spending in the next decade.",
        )
    )
    # The second statement narrows the first to a time window — entailment.
    assert score.entailment >= score.contradiction, score


def test_deberta_lazy_load_only_fires_on_first_call() -> None:
    """The lazy-load contract: instantiation must NOT load weights;
    only ``score()`` triggers the model load."""
    pytest.importorskip(
        "sentence_transformers",
        reason="install [ml] extra (sentence-transformers) to run this test",
    )
    from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer

    scorer = DeBERTaNliScorer()
    assert scorer._model is None  # not loaded until first score() call
    asyncio.run(scorer.score("a", "b"))
    assert scorer._model is not None


def test_nli_eval_script_with_live_model_emits_brier(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end test for the ``--with-live-model`` Brier path on the
    eval scoring script. Loads the real DeBERTa model + re-scores a tiny
    gold set + asserts Brier comes back in the result payload."""
    pytest.importorskip(
        "sentence_transformers",
        reason="install [ml] extra (sentence-transformers) to run this test",
    )
    import json
    import subprocess
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "score"
    script = scripts_dir / "enrichment_nli_contradiction.py"

    corpus = tmp_path / "corpus"
    out = corpus / "enrichments"
    out.mkdir(parents=True)
    (out / "nli_contradiction.json").write_text(
        json.dumps({"data": {"contradictions": []}}), encoding="utf-8"
    )
    gold = tmp_path / "gold"
    gold.mkdir()
    # Two-row gold so Brier has a denominator.
    rows = [
        {
            "insight_a_id": "i1",
            "insight_b_id": "i2",
            "insight_a_text": "AI safety regulation is essential.",
            "insight_b_text": "AI safety regulation will slow research without preventing risks.",
            "label": "contradiction",
        },
        {
            "insight_a_id": "i3",
            "insight_b_id": "i4",
            "insight_a_text": "Demographic shifts will pressure entitlements.",
            "insight_b_text": "Demographic shifts will pressure entitlements next decade.",
            "label": "entailment",
        },
    ]
    (gold / "g.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--corpus",
            str(corpus),
            "--gold",
            str(gold),
            "--with-live-model",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "scored"
    assert "brier" in payload, payload
    assert isinstance(payload["brier"], (int, float))
    assert 0.0 <= payload["brier"] <= 1.0
    assert payload["brier_n"] == 2
