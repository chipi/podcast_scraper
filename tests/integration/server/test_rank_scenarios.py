"""The ranking-scenario corpus must keep DISCRIMINATING (#71).

The value of `scripts/eval/score/rank_scenarios_v1.py` is that its table shows what each signal
does. A table whose rows are all identical shows nothing while looking exactly as authoritative —
so these tests assert the corpus can still tell the signals apart, and that the specific behaviours
earlier fixes established are still the ones it exhibits.

That guard is not hypothetical. Building this corpus produced three configurations that could not
differ, every one of which printed as a confident row:

* the sparse feed carried no KG, so affinity had nothing to match and the coverage-bias scenario
  was inert;
* no velocity enrichment was written, so every `+ trend` row equalled the row above it;
* significance was "disabled" by zeroing its WEIGHT, which does nothing — it is the base
  multiplier, and `rank_discover` reads its params, never its weight (the #21 trap, one layer on).

Each looked like a finding ("this signal changes nothing here") and was a fixture defect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval" / "score"))

from rank_scenarios_v1 import CONFIGS, observe, SCENARIOS, write_corpus  # noqa: E402

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def observations(tmp_path_factory) -> list[dict]:
    root = write_corpus(tmp_path_factory.mktemp("ranking-scenarios") / "corpus")
    return list(observe(root))


def _order(observations: list[dict], scenario: str, config: str) -> list[str]:
    for row in observations:
        if row["scenario"] == scenario and row["config"] == config:
            return [str(x) for x in row["order"]]
    raise AssertionError(f"no observation for {scenario!r} / {config!r}")


def test_every_scenario_and_config_produced_a_feed(observations: list[dict]) -> None:
    assert len(observations) == len(SCENARIOS) * len(CONFIGS)
    for row in observations:
        assert row["order"], row


def test_each_signal_changes_something_somewhere(observations: list[dict]) -> None:
    """The corpus's whole job. A signal that never moves anything is a fixture that cannot see it.

    Asserted per CONFIG rather than globally: "some row differs somewhere" would pass with three of
    the five configurations completely inert, which is precisely the state this corpus started in.
    """
    for label, _config in CONFIGS:
        if label == "recency only":
            continue  # the baseline every other row is compared against
        differs = [
            s.name
            for s in SCENARIOS
            if _order(observations, s.name, label) != _order(observations, s.name, "recency only")
        ]
        assert differs, f"{label!r} produced the recency ordering in EVERY scenario — it is inert"


def test_a_user_with_no_interests_gets_pure_recency(observations: list[dict]) -> None:
    """`rank_discover` returns early on an empty interest set, so nothing can re-rank a new user.

    Worth pinning as a PROPERTY rather than leaving it as a surprising table row: it means every
    signal is dormant until the first follow, which is a product fact, not a tuning accident.
    """
    baseline = _order(observations, "no interests at all", "recency only")
    for label, _config in CONFIGS:
        assert _order(observations, "no interests at all", label) == baseline, label


def test_one_follow_surfaces_its_show(observations: list[dict]) -> None:
    order = _order(observations, "one niche follow", "+ affinity (shipped)")
    assert order[:2] == ["Rope access rigging", "Anchor theory"], order


def test_a_second_follow_does_not_push_the_first_back_down(observations: list[dict]) -> None:
    """#19, observable: the dilution bug divided by the number of follows, so following a second
    thing demoted the first. Both scenarios must lead with the niche show."""
    one = _order(observations, "one niche follow", "+ affinity (shipped)")
    two = _order(observations, "two follows", "+ affinity (shipped)")
    assert one[:2] == two[:2] == ["Rope access rigging", "Anchor theory"], (one, two)


def test_a_sparse_show_can_win_on_relevance(observations: list[dict]) -> None:
    """#23, observable: significance measured pipeline COVERAGE, so an unenriched show could not
    win however relevant it was. Per-feed normalisation is what lets Field Notes lead here."""
    order = _order(observations, "follows the sparse show", "+ affinity (shipped)")
    assert order[0].startswith("Notes from the"), order
    # And it is genuinely the least-enriched feed — otherwise this proves nothing about coverage.
    assert _order(observations, "follows the sparse show", "+ significance")[0] != order[0]


def test_trend_moves_a_hot_topic_up(observations: list[dict]) -> None:
    with_trend = _order(observations, "one niche follow", "+ trend")
    without = _order(observations, "one niche follow", "+ affinity (shipped)")
    assert with_trend != without
    assert with_trend[0].startswith("Hot topic"), with_trend


def test_twenty_follows_stop_discriminating(observations: list[dict]) -> None:
    """Not a defect — a limit worth knowing. Follow everything and the boost lands on everything;
    saturation bounds the damage but cannot invent a preference the follows do not express."""
    broad = _order(observations, "twenty follows", "+ affinity (shipped)")
    assert broad == _order(observations, "twenty follows", "+ significance"), broad
