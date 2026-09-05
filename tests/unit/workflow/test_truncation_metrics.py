"""A truncated run must be distinguishable from a finished one (#1981).

Both supervision loops stop on a wall-clock bound and abandon queued or in-flight episodes, but
the job still reports ``succeeded``. Observed 2026-09-05: the transcription loop hit its 6h budget
after 26 of 30 requested episodes and returned success, so the Dwarkesh feed silently landed at 36
of 40 — and the shortfall was initially misread as the RSS simply having fewer unprocessed
episodes. Nothing in the result distinguished the two.

These counters are the distinguishing signal. ``finish()`` is an explicit dict literal, not
``asdict``, so a counter bumped but not named there is silently dropped — hence the export
assertions.
"""

from __future__ import annotations

import pytest

from podcast_scraper.workflow import metrics

pytestmark = pytest.mark.unit


def test_a_clean_run_reports_no_truncation() -> None:
    m = metrics.Metrics()
    out = m.finish()
    assert out["pipeline_truncation_events"] == 0
    assert out["pipeline_abandoned_in_flight"] == 0
    assert out["pipeline_truncation_reasons"] == []


def test_truncation_is_counted_and_exported() -> None:
    m = metrics.Metrics()
    m.record_truncation("transcription", "wall-clock budget exceeded (21701s > 21600s)")
    out = m.finish()
    assert out["pipeline_truncation_events"] == 1
    # The sequential loop abandons nothing mid-flight; the EVENT is the signal.
    assert out["pipeline_abandoned_in_flight"] == 0
    assert out["pipeline_truncation_reasons"] == [
        "transcription: wall-clock budget exceeded (21701s > 21600s)"
    ]


def test_abandoned_in_flight_accumulates_across_loops() -> None:
    m = metrics.Metrics()
    m.record_truncation("transcription", "wall-clock budget exceeded", abandoned_in_flight=4)
    m.record_truncation("processing", "wall-clock budget exceeded", abandoned_in_flight=1)
    out = m.finish()
    assert out["pipeline_truncation_events"] == 2
    assert out["pipeline_abandoned_in_flight"] == 5
    assert len(out["pipeline_truncation_reasons"]) == 2


def test_a_repeated_reason_is_not_duplicated_in_the_list() -> None:
    """The count is the volume signal; the reasons list is for reading, so keep it distinct."""
    m = metrics.Metrics()
    for _ in range(3):
        m.record_truncation("processing", "main thread exited", abandoned_in_flight=2)
    out = m.finish()
    assert out["pipeline_truncation_events"] == 3
    assert out["pipeline_abandoned_in_flight"] == 6
    assert out["pipeline_truncation_reasons"] == ["processing: main thread exited"]


def test_negative_or_missing_counts_do_not_corrupt_the_total() -> None:
    m = metrics.Metrics()
    m.record_truncation("processing", "r", abandoned_in_flight=-5)
    m.record_truncation("processing", "r2", abandoned_in_flight=None)  # type: ignore[arg-type]
    assert m.finish()["pipeline_abandoned_in_flight"] == 0
