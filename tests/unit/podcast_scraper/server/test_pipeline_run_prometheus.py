"""Pure helpers for :mod:`podcast_scraper.server.pipeline_run_prometheus`."""

from __future__ import annotations

import json
import os
import time

from podcast_scraper.server.pipeline_run_prometheus import (
    discover_run_json_paths_in_mtime_window,
    parse_iso_utc_z,
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
