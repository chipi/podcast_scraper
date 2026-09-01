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

import pytest

from podcast_scraper import config_constants
from podcast_scraper.gi.pipeline import _record_value_gate_drops


def _metrics():
    return SimpleNamespace(gi_value_gate_dropped_insights=[])


class TestTheDroppedTextIsKept:
    def test_only_the_dropped_ones_are_recorded(self):
        m = _metrics()
        specs = [("kept one", "Claim"), ("dropped one", "Claim"), ("kept two", "Claim")]
        _record_value_gate_drops(specs, [True, False, True], [3, 1, 3], m)

        assert [d["text"] for d in m.gi_value_gate_dropped_insights] == ["dropped one"]

    def test_the_tier_is_recorded_with_it(self):
        """The tier is the rater's actual verdict — the thing a comparison compares."""
        m = _metrics()
        _record_value_gate_drops([("filler", "Claim")], [False], [0], m)
        assert m.gi_value_gate_dropped_insights[0]["tier"] == 0

    def test_nothing_recorded_when_nothing_was_dropped(self):
        m = _metrics()
        _record_value_gate_drops([("a", "Claim"), ("b", "Claim")], [True, True], [3, 3], m)
        assert m.gi_value_gate_dropped_insights == []

    def test_it_accumulates_across_episodes_in_a_run(self):
        """A batch's audit must cover the whole run, not just the last episode."""
        m = _metrics()
        _record_value_gate_drops([("ep1 drop", "Claim")], [False], [1], m)
        _record_value_gate_drops([("ep2 drop", "Claim")], [False], [1], m)
        assert [d["text"] for d in m.gi_value_gate_dropped_insights] == ["ep1 drop", "ep2 drop"]

    def test_plain_string_specs_are_handled(self):
        """Specs are (text, kind) tuples today; a bare string must not become 'tuple(...)'."""
        m = _metrics()
        _record_value_gate_drops(["just text"], [False], [1], m)
        assert m.gi_value_gate_dropped_insights[0]["text"] == "just text"


class TestItIsBounded:
    def test_the_cap_holds(self, monkeypatch):
        """A runaway extraction (#1891: the model ignores the count) must not balloon metrics."""
        monkeypatch.setattr(config_constants, "GI_VALUE_GATE_DROP_AUDIT_MAX", 5)
        m = _metrics()
        n = 20
        _record_value_gate_drops(
            [(f"drop {i}", "Claim") for i in range(n)], [False] * n, [1] * n, m
        )
        assert len(m.gi_value_gate_dropped_insights) == 5

    def test_hitting_the_cap_is_LOUD(self, monkeypatch, caplog):
        """A silently truncated audit trail is as misleading as no audit trail.

        Someone reading the metrics would otherwise conclude the gate dropped exactly N, when N
        is really "N, plus however many we declined to write down".
        """
        monkeypatch.setattr(config_constants, "GI_VALUE_GATE_DROP_AUDIT_MAX", 2)
        m = _metrics()
        caplog.set_level("WARNING")
        _record_value_gate_drops(
            [(f"drop {i}", "Claim") for i in range(6)], [False] * 6, [1] * 6, m
        )
        assert any("NOT recorded" in r.getMessage() for r in caplog.records)

    def test_once_full_it_stops_cleanly(self, monkeypatch):
        monkeypatch.setattr(config_constants, "GI_VALUE_GATE_DROP_AUDIT_MAX", 2)
        m = _metrics()
        _record_value_gate_drops([("a", "C"), ("b", "C")], [False, False], [1, 1], m)
        _record_value_gate_drops([("c", "C")], [False], [1], m)
        assert [d["text"] for d in m.gi_value_gate_dropped_insights] == ["a", "b"]


class TestTelemetryNeverBreaksExtraction:
    def test_no_metrics_object_is_fine(self):
        _record_value_gate_drops([("x", "C")], [False], [1], None)  # must not raise

    def test_a_hostile_metrics_object_is_swallowed(self):
        """Telemetry is best-effort; losing an audit entry must never lose the episode."""

        class Hostile:
            @property
            def gi_value_gate_dropped_insights(self):
                raise RuntimeError("boom")

        _record_value_gate_drops([("x", "C")], [False], [1], Hostile())  # must not raise


class TestItReachesMetricsJson:
    def test_the_field_is_declared_and_exported(self):
        """Both bars — a stray attribute would be dropped before metrics.json like the other 20."""
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        _record_value_gate_drops([("dropped", "Claim")], [False], [1], m)
        exported = m.finish()

        assert "gi_value_gate_dropped_insights" in exported
        assert [d["text"] for d in exported["gi_value_gate_dropped_insights"]] == ["dropped"]

    def test_default_is_an_empty_list_not_a_missing_key(self):
        from podcast_scraper.workflow.metrics import Metrics

        assert Metrics().finish()["gi_value_gate_dropped_insights"] == []

    def test_the_export_is_a_copy_not_the_live_list(self):
        """A shared mutable would let a later episode retroactively edit a written artifact."""
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        _record_value_gate_drops([("first", "C")], [False], [1], m)
        snapshot = m.finish()["gi_value_gate_dropped_insights"]
        _record_value_gate_drops([("second", "C")], [False], [1], m)

        assert [d["text"] for d in snapshot] == ["first"]


class TestItIsActuallyWiredIntoTheGate:
    """Drive ``_gate_on_evidence``, not just the recorder.

    Mutation-testing caught this: deleting the ``_record_value_gate_drops(...)`` call from the
    gate path left all 15 tests above GREEN, because every one of them called the recorder
    directly. A recorder that is never invoked is indistinguishable from no recorder — and the
    whole point of F2 is that the absence is invisible.

    This is the third time in this change set that the WIRING was the defect while the function
    tested fine, so it now gets its own class rather than a comment.
    """

    def test_the_gate_path_records_what_it_dropped(self, monkeypatch):
        from podcast_scraper.gi import pipeline as gi_pipeline
        from podcast_scraper.workflow.metrics import Metrics

        # Gate verdict: keep the first, drop the second.
        monkeypatch.setattr(
            "podcast_scraper.gi.value_gate.value_gate_evaluate",
            lambda specs, **kw: ([True, False], [3, 0]),
        )
        m = Metrics()
        specs = [("survivor", "Claim"), ("casualty", "Claim")]

        kept, _quotes, _tiers = gi_pipeline._gate_on_evidence(
            specs,
            [[], []],
            cfg=SimpleNamespace(),
            provider=None,
            transcript_text=None,
            transcript_segments=None,
            pipeline_metrics=m,
        )

        assert [s[0] for s in kept] == ["survivor"], "the gate must still filter"
        assert [d["text"] for d in m.gi_value_gate_dropped_insights] == ["casualty"], (
            "the dropped insight was not recorded — gi.json keeps only survivors, so this is "
            "the ONLY record of the pre-gate set and #1895's eval cannot run without it"
        )

    def test_an_all_keep_verdict_records_nothing(self, monkeypatch):
        from podcast_scraper.gi import pipeline as gi_pipeline
        from podcast_scraper.workflow.metrics import Metrics

        monkeypatch.setattr(
            "podcast_scraper.gi.value_gate.value_gate_evaluate",
            lambda specs, **kw: ([True, True], [3, 3]),
        )
        m = Metrics()
        gi_pipeline._gate_on_evidence(
            [("a", "Claim"), ("b", "Claim")],
            [[], []],
            cfg=SimpleNamespace(),
            provider=None,
            transcript_text=None,
            transcript_segments=None,
            pipeline_metrics=m,
        )
        assert m.gi_value_gate_dropped_insights == []


@pytest.mark.parametrize("cap", [0, -1])
def test_a_nonpositive_cap_records_nothing_rather_than_crashing(monkeypatch, cap):
    monkeypatch.setattr(config_constants, "GI_VALUE_GATE_DROP_AUDIT_MAX", cap)
    m = _metrics()
    _record_value_gate_drops([("x", "C")], [False], [1], m)
    assert m.gi_value_gate_dropped_insights == []
