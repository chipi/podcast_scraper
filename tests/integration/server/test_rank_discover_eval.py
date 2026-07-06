"""#1139 gate — personalized discovery ranking must beat recency on the seeded personas.

The CI-enforced half of ``scripts/eval/score/rank_discover_v1.py``: it runs the same
offline eval over the committed app-validation corpus + seeded users and asserts the
flip-the-flag gate holds — a mean nDCG@K uplift over the recency baseline, plus a nDCG
floor. If ``rank_discover`` / ``derive_interests`` ever regresses so personalization no
longer surfaces each persona's relevant shows above plain recency, this fails *before*
``APP_PERSONALIZED_RANKING`` could responsibly ship on.

Deterministic (no ML, no network): it reads the checked-in corpus and seeds throwaway
per-user state, exactly as the script does.
"""

from __future__ import annotations

import pytest

from scripts.eval.score.rank_discover_v1 import evaluate

pytestmark = [pytest.mark.integration]


def test_rank_discover_gate_passes_on_seeded_personas() -> None:
    result = evaluate()
    metrics = result["metrics"]
    per_user = result["per_user"]

    # All three seeded personas carry gold and were scored.
    assert metrics["n_users"] >= 3, metrics
    assert len(per_user) == metrics["n_users"]

    # The gate itself: personalization measurably beats recency across the personas.
    assert metrics["gate"]["pass"], metrics
    assert metrics["mean_ndcg_uplift"] >= metrics["gate"]["uplift_min"], metrics

    # No persona regresses — personalized never ranks its gold shows below recency, and
    # each clears the nDCG floor (so a single strong persona can't mask a broken one).
    floor = metrics["gate"]["ndcg_floor"]
    for row in per_user:
        assert row["personalized_ndcg"] >= row["recency_ndcg"], row
        assert row["personalized_ndcg"] >= floor, row
