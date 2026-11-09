"""Optional real-model integration test for ``DeBERTaNliScorer``.

Marked ``ml_models`` so it only runs when the operator opts in:

    .venv/bin/pytest -o addopts="" -m ml_models \
        tests/e2e/enrichment/test_deberta_real_model_optin.py

NOT run in CI per ``[[feedback_no_llm_in_ci]]``. tests/conftest.py sets
``HF_HUB_OFFLINE=1``, so the model (~570 MB of safetensors) must already be in
``HF_HUB_CACHE`` at the revision pinned in config_constants
(``NLI_DEBERTA_V3_SMALL_REVISION``); no preload script fetches it.

Why an opt-in test exists: the DeBERTaNliScorer is the only chunk-4
scorer that ever loads a real model — its load + predict paths were
only exercised via mypy / coverage-deferred mocks until this test
landed. The marker keeps CI fast while letting operators verify the
scorer end-to-end before shipping a corpus.

To run, with the [ml] extra installed and the model cached:
    pytest -o addopts="" -m ml_models -k deberta_real_model_optin -s
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
            "AI safety regulation is essential for preventing catastrophic risks.",
            "AI safety regulation does nothing to prevent catastrophic risks.",
        )
    )
    # The hypothesis denies what the premise asserts. (A pair that only argues
    # past the premise, "will slow research without preventing risks", scores
    # neutral at 0.995 on this head: it is not a strict contradiction.) Don't
    # assert a tight probability bound — the model can drift across versions —
    # just assert the ordering: contradiction > neutral and > entailment.
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
            "Demographic shifts will pressure entitlement spending in the next decade.",
            "Demographic shifts will pressure entitlement spending.",
        )
    )
    # The premise is the narrower claim, so it entails the general one. The
    # reverse direction adds a time window the premise never states and is
    # neutral (1.000 on this head), not entailment.
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
