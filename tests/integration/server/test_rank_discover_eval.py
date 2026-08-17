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


class TestTheEvalScoresTheConfigThatSHIPS:
    """The gate above proves personalisation works — for whichever config the eval was handed.

    Until #21 the eval never accepted one: it hardcoded ``DEFAULT_RANKING_CONFIG`` while
    ``/discover`` loads the OPERATOR-STORED config on every request
    (``routes/app_discover.py``). So an admin ``PUT /ranking-config`` zeroing
    ``interest_affinity`` would have shipped in silence — the gate would have gone on reporting a
    healthy uplift for a feed that had quietly stopped personalising, because it was scoring a
    system nobody was running.

    These pin the two halves of that: the eval must be SENSITIVE to the config, and the CLI must be
    able to score a deployment's stored one.
    """

    def test_a_broken_config_actually_fails_the_gate(self) -> None:
        """The sensitivity check. If a zeroed affinity weight still passed, the ``config=``
        argument would be decorative and the gap would be exactly as open as before."""
        from podcast_scraper.server.app_ranking_config import (
            DEFAULT_RANKING_CONFIG,
            RankingConfig,
        )

        zeroed = RankingConfig(
            signals=tuple(
                (
                    s.__class__(**{**s.__dict__, "weight": 0.0})
                    if s.name == "interest_affinity"
                    else s
                )
                for s in DEFAULT_RANKING_CONFIG.signals
            )
        )
        broken = evaluate(config=zeroed)
        assert not broken["metrics"]["gate"]["pass"], (
            "zeroing interest_affinity left the gate passing — the eval is not actually reading "
            "the config it is given, so a tuning change could still ship unnoticed"
        )

    def test_the_default_config_is_what_the_unparameterised_call_scores(self) -> None:
        """Back-compat: the existing gate call keeps meaning what it meant."""
        from podcast_scraper.server.app_ranking_config import DEFAULT_RANKING_CONFIG

        assert (
            evaluate()["metrics"]["mean_ndcg_uplift"]
            == evaluate(config=DEFAULT_RANKING_CONFIG)["metrics"]["mean_ndcg_uplift"]
        )

    def test_the_cli_can_score_a_stored_config(self, tmp_path) -> None:
        """A deployment runs ``--data-dir <APP_DATA_DIR>``; that path must reach the scorer.

        Asserted through the store's own round-trip rather than by shelling out, so the test pins
        the wiring (stored config -> evaluate) and not the argparse spelling.
        """
        from podcast_scraper.server.app_ranking_config import (
            DEFAULT_RANKING_CONFIG,
            RankingConfig,
        )
        from podcast_scraper.server.app_ranking_config_store import (
            load_ranking_config,
            save_ranking_config,
        )

        zeroed = RankingConfig(
            signals=tuple(
                (
                    s.__class__(**{**s.__dict__, "weight": 0.0})
                    if s.name == "interest_affinity"
                    else s
                )
                for s in DEFAULT_RANKING_CONFIG.signals
            )
        )
        save_ranking_config(tmp_path, zeroed)
        assert not evaluate(config=load_ranking_config(tmp_path))["metrics"]["gate"]["pass"], (
            "a stored config that breaks personalisation must fail the gate when scored — that is "
            "the whole point of --data-dir"
        )
