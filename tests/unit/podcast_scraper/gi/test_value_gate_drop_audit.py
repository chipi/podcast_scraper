"""What the value gate REMOVED must be recoverable (#1895 F2).

THE BLOCKER THIS CLEARS. ``gi.json`` persists only the insights that PASSED the gate. Once a
run finishes, the pre-gate set exists nowhere — so re-grading a past episode just re-grades the
winners, and the rater comparison #1895 exists for (self vs distinct vs stronger) cannot be run
on any historical episode at all. The issue calls this out as blocking the eval entirely.

WHY IT IS WORTH A WRITE. A well-filtered set and an over-pruned set are indistinguishable from
outside: both look like "fewer insights". No error, no warning, and on a self-hosted box no cost
signal either — the run is $0 whether the gate removed filler or removed the best material in
the episode. Production measured 13-58% dropped against a code comment predicting ~10%, while
SELF-grading, which is supposed to be the lenient case.

Counts alone are not enough, which is why the TEXT is stored: two raters can drop the same
number of insights while disagreeing completely about which ones, and that disagreement is the
entire question the eval asks.
"""

from __future__ import annotations

from types import SimpleNamespace


def _metrics():
    return SimpleNamespace(gi_value_gate_dropped_insights=[])


class TestItReachesMetricsJson:

    def test_default_is_an_empty_list_not_a_missing_key(self):
        from podcast_scraper.workflow.metrics import Metrics

        assert Metrics().finish()["gi_value_gate_dropped_insights"] == []
