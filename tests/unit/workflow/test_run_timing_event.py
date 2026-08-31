"""Run-level stage-time accounting reaches the event stream (#1888).

The timings were always collected but only ever landed in ``run.json`` on the box: a
VictoriaLogs query for ``run_duration_seconds`` / ``avg_gi_seconds`` across three days
of production returned zero lines. Without them, "how much of wall-clock is the
accelerator actually busy" can only be eyeballed from a single run's logs.
"""

from __future__ import annotations

import json

from podcast_scraper.workflow import metrics as metrics_mod
from podcast_scraper.workflow.orchestration import _emit_run_timing_event


def _metrics_with_a_run():
    m = metrics_mod.Metrics()
    m.episodes_scraped_total = 1
    m.transcribe_time_by_episode = {1: 80.0}
    m.summarize_time_by_episode = {1: 120.0}
    m.gi_times = [60.0]
    m.kg_times = [40.0]
    m.download_media_time_by_episode = {1: 30.0}
    m.vector_index_seconds = 20.0
    m._start_time -= 600.0
    return m


def _emitted(caplog):
    for rec in caplog.records:
        msg = rec.getMessage()
        if '"run_timing"' in msg:
            return json.loads(msg)
    return None


def test_emits_the_accounting_with_a_derived_share(caplog):
    caplog.set_level("INFO")
    _emit_run_timing_event(_metrics_with_a_run())

    ev = _emitted(caplog)
    assert ev is not None, "run_timing event was not emitted"
    assert ev["transcribe_sec_total"] == 80.0
    assert ev["gi_sec_total"] == 60.0
    assert ev["kg_sec_total"] == 40.0
    # 80 + 120 + 60 + 40 = 300 of a 600s run.
    assert ev["model_stage_sec_total"] == 300.0
    assert ev["model_stage_share_pct"] == 50.0
    # download 30 + index 20; the rest of wall-clock is claimed by nothing.
    assert ev["local_stage_sec_total"] == 50.0
    assert ev["unaccounted_sec"] == 250.0


def test_carries_no_machine_label(caplog):
    """Machine is a property of the profile, not the run.

    The run context (#1874) already stamps ``profile`` and the per-stage providers, so
    the DGX-vs-cloud split is derivable at query time. A hardcoded ``machine`` field
    here would be wrong for every hybrid profile.
    """
    caplog.set_level("INFO")
    _emit_run_timing_event(_metrics_with_a_run())
    assert "machine" not in (_emitted(caplog) or {})


def test_never_raises_on_a_broken_metrics_object(caplog):
    """Telemetry must not fail a run that otherwise succeeded."""
    caplog.set_level("INFO")

    class Exploding:
        def finish(self):
            raise RuntimeError("metrics blew up")

    _emit_run_timing_event(Exploding())  # must not raise
    assert _emitted(caplog) is None


def test_zero_length_run_does_not_divide_by_zero(caplog):
    caplog.set_level("INFO")
    m = metrics_mod.Metrics()
    _emit_run_timing_event(m)
    ev = _emitted(caplog)
    assert ev is not None
    assert ev.get("model_stage_share_pct") is None
