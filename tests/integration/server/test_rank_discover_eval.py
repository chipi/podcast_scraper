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


class TestTheProfileCanForget:
    """#24 — a persona whose taste MOVED, and whose gold is the taste they moved to.

    Without decay the derived profile is a pure accumulator: 12 outdoor episodes from six months
    ago outvote 4 investing episodes from last week, for ever, so the feed keeps recommending the
    taste the user left. ``u_shifted`` is that user, and its gold (p05) is only reachable if the
    profile can forget.

    Load-bearing, measured by replacing ``_decayed`` with a no-op: nDCG 1.000 -> 0.526. Without a
    persona like this one the whole of #24 could be deleted and every gate number would hold,
    because the other personas heard everything at the same instant.
    """

    def test_the_shifted_persona_is_scored(self) -> None:
        rows = [r for r in evaluate()["per_user"] if r["user_id"] == "u_shifted"]
        assert rows, "u_shifted is missing — nothing in the gate exercises interest decay"
        assert rows[0]["personalized_ndcg"] > rows[0]["recency_ndcg"]

    def test_deleting_decay_costs_the_shifted_persona(self, monkeypatch) -> None:
        """The sensitivity check. If a no-op ``_decayed`` scored the same, decay would be
        decorative and this class would be asserting nothing."""
        from podcast_scraper.server import app_user_corpus

        with_decay = next(r for r in evaluate()["per_user"] if r["user_id"] == "u_shifted")

        monkeypatch.setattr(
            app_user_corpus,
            "_decayed",
            lambda engaged, **kw: [(slug, 1.0) for slug, _ts in engaged],
        )
        without = next(r for r in evaluate()["per_user"] if r["user_id"] == "u_shifted")

        assert without["personalized_ndcg"] < with_decay["personalized_ndcg"], (
            "removing time-decay did not change this persona's score, so the gate cannot tell "
            f"whether #24 is still there: {without} vs {with_decay}"
        )


class TestTheEvalScoresTheINTERESTSPLITThatSHIPS:
    """#27 — the third divergence of the same kind, after the pool (#17) and the config (#21).

    ``/discover`` hands ``rank_discover`` explicit follows and derived interests SEPARATELY, and
    since #19 they carry different weight (an inference counts ``derived_ratio``, currently half).
    The eval merged them into one list and passed it as the EXPLICIT argument, so every inferred
    topic scored as a stated follow and the gate reported on a strictly more personalised system
    than the route runs. Once ranking learned to tell the two apart, the eval had to as well.
    """

    def test_the_scorer_passes_the_two_kinds_to_the_two_arguments(self, monkeypatch) -> None:
        """Pins the WIRING, because the reported counts do not.

        Written after its own sabotage: reverting ``_score_user`` to the merged
        ``rank_discover(corpus, interests, ...)`` call left the split-reporting test below and the
        derived-only test above **both green** (6 passed) — they observe the row dict, not the
        call. And the merged form cannot be caught by score alone: for a derived-only persona,
        weighting every token at 1.0 instead of 0.5 is a monotone transform of the same matched
        count, so the ranking is IDENTICAL and no metric moves. The divergence only shows on mixed
        personas, and only as a re-weighting between kinds.

        So this asserts the contract at the call boundary: whatever reaches the ``interests``
        argument is the user's stated follows, and derived tokens arrive through
        ``derived_interests=`` — the same shape ``routes/app_discover.py`` uses.
        """
        from scripts.eval.score import rank_discover_v1 as module

        calls: list[tuple[tuple, dict]] = []
        real = module.rank_discover

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(module, "rank_discover", spy)
        module.evaluate()

        # The personalized arm of every persona; the recency arm passes [] and is skipped.
        personalized = [(a, k) for a, k in calls if a[1]]
        assert personalized, "no personalized ranking call was made"
        mixed_seen = False
        for args, kwargs in personalized:
            explicit = list(args[1])
            derived = list(kwargs.get("derived_interests", []))
            assert "derived_interests" in kwargs, (
                "derived tokens must arrive through derived_interests=, not folded into the "
                "explicit argument — that is the merge this class exists to prevent"
            )
            assert not (set(explicit) & set(derived)), "a token cannot be both stated and inferred"
            if explicit and derived:
                mixed_seen = True
        assert mixed_seen, (
            "no persona carries BOTH stated and inferred interests, so this run cannot observe "
            "the two being weighted differently at all"
        )

    def test_every_persona_reports_its_interest_split(self) -> None:
        for row in evaluate()["per_user"]:
            assert row["n_explicit_interests"] + row["n_derived_interests"] == row["n_interests"], (
                f"{row['user_id']}: the split must account for every interest, or the report "
                "hides which path the persona exercised"
            )

    def test_a_derived_only_persona_is_actually_scored(self) -> None:
        """The coverage this class exists for: at least one persona reaches ``/discover``'s
        behaviour-only path — no picker, no follows, ranking driven purely by what they heard.

        With only explicit-follow personas the gate could stay green while derivation returned
        nothing at all, because no scored user depended on it. ``u_implicit`` is that user; its
        gold shows (p06, p08) are also the two no other persona covers, so it widens the corpus
        slice the gate touches rather than re-weighting the slice already covered.
        """
        rows = evaluate()["per_user"]
        derived_only = [r for r in rows if r["n_explicit_interests"] == 0]
        assert derived_only, (
            "no persona exercises the derived-interests-only path — #1139's whole premise is that "
            "a user who never opens the picker still gets personalised"
        )
        for row in derived_only:
            assert row["n_derived_interests"] > 0, f"{row['user_id']}: derivation produced nothing"
            assert row["personalized_ndcg"] > row["recency_ndcg"], (
                f"{row['user_id']}: behaviour-derived interests did not beat plain recency — "
                "personalisation for picker-less users is not working"
            )
