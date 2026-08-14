"""Unit tests for budget headroom (#1651).

Every cost signal in the estate is spend-triggered, so an account about to run dry looks
exactly like a healthy one. Job ``8645ecd0`` proved the cost of that on 2026-08-13: it died
mid-corpus with "no budget/credit left on this key", and twelve auto-filed escalations trace
to the same condition.

The property these tests pin is that the check happens BEFORE the batch starts. A cap
evaluated during a run cannot prevent exhaustion — it can only interrupt it halfway and leave
a partially ingested corpus.
"""

from __future__ import annotations

import pytest

from podcast_scraper.workflow.budget_headroom import (
    check_headroom,
    DEFAULT_MAX_CONSUMPTION_RATIO,
    HeadroomVerdict,
    project_batch_cost,
)

pytestmark = [pytest.mark.unit]


class TestCheckHeadroom:
    def test_plenty_of_room_is_ok(self) -> None:
        check = check_headroom(remaining_usd=10.0, projected_cost_usd=1.0)
        assert check.verdict is HeadroomVerdict.OK
        assert check.may_proceed is True
        assert check.consumption_ratio == 0.1

    def test_a_large_share_of_what_is_left_is_flagged_tight(self) -> None:
        check = check_headroom(remaining_usd=10.0, projected_cost_usd=5.0)
        assert check.verdict is HeadroomVerdict.TIGHT
        assert check.may_proceed is True

    def test_exceeding_the_consumption_ratio_is_refused_before_starting(self) -> None:
        """Finishing with nothing left just defers the hard-stop to the next run."""
        check = check_headroom(remaining_usd=10.0, projected_cost_usd=9.0)
        assert check.verdict is HeadroomVerdict.INSUFFICIENT
        assert check.may_proceed is False

    def test_cost_exceeding_remaining_budget_is_refused(self) -> None:
        check = check_headroom(remaining_usd=5.0, projected_cost_usd=9.0)
        assert check.verdict is HeadroomVerdict.INSUFFICIENT
        assert "hard-stop mid-corpus" in check.reason

    def test_zero_remaining_is_refused_with_the_real_consequence_named(self) -> None:
        """This is the 8645ecd0 state — the next provider call hard-stops the run."""
        check = check_headroom(remaining_usd=0.0, projected_cost_usd=1.0)
        assert check.verdict is HeadroomVerdict.INSUFFICIENT
        assert "hard-stop" in check.reason

    def test_negative_remaining_is_refused(self) -> None:
        assert (
            check_headroom(remaining_usd=-2.0, projected_cost_usd=1.0).verdict
            is HeadroomVerdict.INSUFFICIENT
        )

    def test_unreadable_budget_is_unknown_and_still_proceeds(self) -> None:
        """Refusing on unmeasurable headroom trades a possible failure for a certain one.

        A gateway that does not report budgets would otherwise block every run forever, which
        is a worse outcome than the risk being managed.
        """
        check = check_headroom(remaining_usd=None, projected_cost_usd=1.0)
        assert check.verdict is HeadroomVerdict.UNKNOWN
        assert check.may_proceed is True

    def test_missing_projection_is_unknown(self) -> None:
        check = check_headroom(remaining_usd=10.0, projected_cost_usd=None)
        assert check.verdict is HeadroomVerdict.UNKNOWN

    def test_the_ratio_is_configurable(self) -> None:
        assert (
            check_headroom(
                remaining_usd=10.0, projected_cost_usd=5.0, max_consumption_ratio=0.4
            ).verdict
            is HeadroomVerdict.INSUFFICIENT
        )

    def test_explain_shows_the_arithmetic(self) -> None:
        """An operator who cannot check the reasoning will override the gate."""
        text = check_headroom(remaining_usd=10.0, projected_cost_usd=9.0).explain()
        assert "$9.00" in text and "$10.00" in text and "90%" in text

    def test_explain_works_without_numbers(self) -> None:
        assert "unknown" in check_headroom(remaining_usd=None, projected_cost_usd=None).explain()

    def test_the_default_leaves_a_margin(self) -> None:
        assert 0 < DEFAULT_MAX_CONSUMPTION_RATIO < 1.0


class TestProjectBatchCost:
    def test_projects_from_audio_minutes(self) -> None:
        # 15 episodes x 75 min x $0.005/min = $5.62 — close to Latent Space's measured $5.64.
        assert project_batch_cost(15, 75, 0.005) == pytest.approx(5.625)

    def test_a_long_form_feed_costs_more_at_the_same_episode_count(self) -> None:
        """The #1658 thesis: episode count is the wrong unit for a minutes-driven cap."""
        short_form = project_batch_cost(15, 49, 0.005)
        long_form = project_batch_cost(15, 92, 0.005)
        assert long_form > short_form * 1.8

    @pytest.mark.parametrize(
        "episodes,minutes,rate", [(0, 50, 0.005), (10, 0, 0.005), (10, 50, 0), (-1, 50, 0.005)]
    )
    def test_degenerate_inputs_project_zero_rather_than_raising(
        self, episodes: int, minutes: float, rate: float
    ) -> None:
        assert project_batch_cost(episodes, minutes, rate) == 0.0


class TestTheGateActuallyGates:
    def test_the_corpus_repair_against_a_ten_dollar_cap(self) -> None:
        """The concrete decision this was built for: #1655 costs ~$5.85 against a $10 cap.

        Under the default 80% ratio that is TIGHT — it proceeds, and the operator is told,
        which is the honest answer rather than a silent green light.
        """
        check = check_headroom(remaining_usd=10.0, projected_cost_usd=5.85)
        assert check.verdict is HeadroomVerdict.TIGHT
        assert check.may_proceed is True

    def test_the_same_repair_on_a_nearly_spent_key_is_refused(self) -> None:
        check = check_headroom(remaining_usd=2.0, projected_cost_usd=5.85)
        assert check.may_proceed is False
