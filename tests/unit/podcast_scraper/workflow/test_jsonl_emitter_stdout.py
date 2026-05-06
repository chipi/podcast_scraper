"""JSONL stdout echo for Loki (GitHub #746)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import config as config_module
from podcast_scraper.workflow import jsonl_emitter, metrics as metrics_module


def test_jsonl_emitter_echo_stdout_writes_json_lines(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "run.jsonl"
    cfg = config_module.Config(
        rss_urls=[config_module.RssFeedEntry(url="https://example.com/feed.xml")],
    )
    collector = metrics_module.Metrics()
    with jsonl_emitter.JSONLEmitter(collector, str(path), echo_stdout=True) as emitter:
        emitter.emit_run_started(cfg, run_id="stdout-run")
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    row = json.loads(captured[0])
    assert row["event_type"] == "run_started"
    assert row["run_id"] == "stdout-run"
    disk = path.read_text(encoding="utf-8").strip().splitlines()
    assert disk == captured


def test_jsonl_emitter_echo_stdout_off_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "run.jsonl"
    cfg = config_module.Config(
        rss_urls=[config_module.RssFeedEntry(url="https://example.com/feed.xml")],
    )
    collector = metrics_module.Metrics()
    with jsonl_emitter.JSONLEmitter(collector, str(path), echo_stdout=False) as emitter:
        emitter.emit_run_started(cfg, run_id="quiet")
    assert capsys.readouterr().out == ""
    assert path.read_text(encoding="utf-8").strip()
