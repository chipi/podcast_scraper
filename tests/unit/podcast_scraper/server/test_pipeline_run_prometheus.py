"""Pure helpers for :mod:`podcast_scraper.server.pipeline_run_prometheus`."""

from __future__ import annotations

import json
import os
import time

from podcast_scraper.server.pipeline_run_prometheus import (
    _command_type_label,
    _KNOWN_COMMAND_TYPES,
    discover_run_json_paths_in_mtime_window,
    last_success_age_seconds,
    parse_iso_utc_z,
)


def _registry(corpus_root, rows) -> None:
    """Write a job registry the way ``pipeline_job_registry`` does."""
    vdir = corpus_root / ".viewer"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "jobs.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )


def test_parse_iso_utc_z_z_suffix() -> None:
    dt = parse_iso_utc_z("2026-05-05T12:00:00Z")
    assert dt is not None
    assert dt.year == 2026


def test_parse_iso_utc_z_none_empty() -> None:
    assert parse_iso_utc_z(None) is None
    assert parse_iso_utc_z("") is None
    assert parse_iso_utc_z("not-a-date") is None


def test_discover_run_json_paths_in_mtime_window(tmp_path) -> None:
    sub = tmp_path / "feeds" / "f1"
    sub.mkdir(parents=True)
    run_json = sub / "run.json"
    run_json.write_text(json.dumps({"metrics": {"avg_transcribe_seconds": 1.2}}), encoding="utf-8")

    anchor = time.time()
    os.utime(run_json, (anchor, anchor))

    assert discover_run_json_paths_in_mtime_window(tmp_path, anchor - 5.0, anchor + 5.0) == [
        run_json.resolve()
    ]

    assert discover_run_json_paths_in_mtime_window(tmp_path, anchor + 100.0, anchor + 200.0) == []


# --- command_type label on podcast_pipeline_jobs_finished_total ---------------------------
# This label is what lets the stalled-ingestion alert distinguish "the NIGHTLY has not
# completed" from "no job of ANY kind has completed". Without it an operator-triggered
# enrichment succeeding masks a dead nightly.


def test_command_type_label_passes_known_values_through() -> None:
    for known in _KNOWN_COMMAND_TYPES:
        assert _command_type_label({"command_type": known}) == known


def test_command_type_label_collapses_unknown_to_other() -> None:
    """An unbounded label on a counter degrades the whole TSDB, not just this metric."""
    assert _command_type_label({"command_type": "run-7f3a9c-user-42"}) == "other"
    assert _command_type_label({"command_type": ""}) == "other"
    assert _command_type_label({}) == "other"
    assert _command_type_label({"command_type": None}) == "other"


def test_command_type_label_allows_every_declared_command() -> None:
    """Drift guard.

    ``_KNOWN_COMMAND_TYPES`` duplicates the ``jobs.COMMAND_*`` constants (importing the job
    runner at metrics-init time is not worth it). A new command type added there and missed
    here would silently record as ``other`` — and an alert filtering on its real name would
    match nothing while looking perfectly healthy. That is the #2119 failure shape exactly:
    two lists that must agree, with nothing asserting that they do.
    """
    from podcast_scraper.server import jobs

    declared = {
        value
        for name, value in vars(jobs).items()
        if name.startswith("COMMAND_") and isinstance(value, str)
    }
    assert declared, "no COMMAND_* constants found — did they move or get renamed?"
    missing = declared - set(_KNOWN_COMMAND_TYPES)
    assert not missing, f"command types absent from _KNOWN_COMMAND_TYPES: {sorted(missing)}"


# --- age of last success, read from the registry ------------------------------------------
# The stalled-ingestion alert runs on this. It reads the durable registry rather than an
# in-memory counter so it survives an API restart or a deploy — a counter reset is
# indistinguishable from a genuine stall.

# Derived from an ISO string rather than a magic epoch int, so the anchor cannot silently
# drift from the dates it is compared against. 01:43:20 puts 01:00:00Z exactly 2600s back
# and the prior day's 22:00:00Z exactly 13400s back.
_NOW = parse_iso_utc_z("2026-11-15T01:43:20Z").timestamp()  # type: ignore[union-attr]


def test_age_is_measured_per_command_type(tmp_path) -> None:
    _registry(
        tmp_path,
        [
            {
                "job_id": "a",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "2026-11-14T22:00:00Z",  # _NOW - 13400s
            },
            {
                "job_id": "b",
                "command_type": "corpus_enrichment",
                "status": "succeeded",
                "ended_at": "2026-11-15T01:00:00Z",  # _NOW - 2600s
            },
        ],
    )
    ages = last_success_age_seconds(tmp_path, now=_NOW)
    assert ages["full_incremental_pipeline"] == 13400.0
    assert ages["corpus_enrichment"] == 2600.0


def test_only_the_most_recent_success_counts(tmp_path) -> None:
    _registry(
        tmp_path,
        [
            {
                "job_id": "old",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "2026-11-01T00:00:00Z",
            },
            {
                "job_id": "new",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "2026-11-15T01:00:00Z",
            },
        ],
    )
    assert last_success_age_seconds(tmp_path, now=_NOW)["full_incremental_pipeline"] == 2600.0


def test_failed_and_running_jobs_do_not_count_as_success(tmp_path) -> None:
    """A nightly that starts and dies every night must still read as stalled."""
    _registry(
        tmp_path,
        [
            {
                "job_id": "f",
                "command_type": "full_incremental_pipeline",
                "status": "failed",
                "ended_at": "2026-11-15T01:00:00Z",
            },
            {
                "job_id": "r",
                "command_type": "full_incremental_pipeline",
                "status": "running",
                "ended_at": "2026-11-15T01:00:00Z",
            },
        ],
    )
    assert "full_incremental_pipeline" not in last_success_age_seconds(tmp_path, now=_NOW)


def test_never_succeeded_is_absent_not_zero(tmp_path) -> None:
    """Zero would read as 'succeeded just now' and invert the alert."""
    _registry(tmp_path, [])
    assert last_success_age_seconds(tmp_path, now=_NOW) == {}
    # A corpus with no registry file at all must behave the same way, not raise.
    assert last_success_age_seconds(tmp_path / "nonexistent", now=_NOW) == {}


def test_future_timestamp_clamps_to_zero_not_negative(tmp_path) -> None:
    """Clock skew between the pipeline container and the API must not read as healthy."""
    _registry(
        tmp_path,
        [
            {
                "job_id": "skew",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "2026-12-01T00:00:00Z",  # well after _NOW
            }
        ],
    )
    assert last_success_age_seconds(tmp_path, now=_NOW)["full_incremental_pipeline"] == 0.0


def test_unparsable_and_missing_ended_at_are_skipped(tmp_path) -> None:
    _registry(
        tmp_path,
        [
            {"job_id": "x", "command_type": "full_incremental_pipeline", "status": "succeeded"},
            {
                "job_id": "y",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "not-a-date",
            },
        ],
    )
    assert last_success_age_seconds(tmp_path, now=_NOW) == {}
