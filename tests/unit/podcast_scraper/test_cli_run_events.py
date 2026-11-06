"""``pipeline_run.started`` / ``.completed`` from ``cli.main``, and the ``--trigger`` flag."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import cli, config

pytestmark = [pytest.mark.unit]

_EMIT = "podcast_scraper.obs.events.emit_event"
_A = "https://a.example/feed.xml"
_B = "https://b.example/feed.xml"


@pytest.fixture(autouse=True)
def _no_host_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_validate_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "_validate_python_version", lambda: None)
    monkeypatch.delenv("PODCAST_RUN_TRIGGER", raising=False)


def _run_events(emit: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(c.args[0]), dict(c.kwargs))
        for c in emit.call_args_list
        if c.args[0].startswith("pipeline_run.")
    ]


def _multi(tmp_path: Path, fake_run: Any, *extra: str) -> tuple[int, MagicMock]:
    with patch(_EMIT) as emit:
        code = cli.main(
            [_A, "--rss", _B, "--output-dir", str(tmp_path), "--max-episodes", "1", *extra],
            run_pipeline_fn=fake_run,
        )
    return code, emit


def _single(tmp_path: Path, fake_run: Any, *extra: str) -> tuple[int, MagicMock]:
    with patch(_EMIT) as emit:
        code = cli.main(
            [_A, "--output-dir", str(tmp_path), "--max-episodes", "1", *extra],
            run_pipeline_fn=fake_run,
        )
    return code, emit


def test_multi_feed_success_emits_started_then_completed(tmp_path: Path) -> None:
    code, emit = _multi(
        tmp_path, lambda cfg: (3, "ok"), "--run-id", "job-1", "--trigger", "nightly"
    )
    assert code == 0
    (started_type, started), (completed_type, completed) = _run_events(emit)
    assert (started_type, completed_type) == ("pipeline_run.started", "pipeline_run.completed")
    assert started["run_id"] == "job-1"
    assert started["trigger"] == "nightly"
    assert started["feeds"] == 2
    assert started["started_at"].endswith("+00:00")
    assert completed["status"] == "succeeded"
    assert (completed["feeds"], completed["ok"], completed["failed"]) == (2, 2, 0)
    assert completed["not_started"] == 0
    assert completed["episodes_processed"] == 6
    assert completed["duration_s"] >= 0
    assert completed["error_class"] is None


def test_multi_feed_with_one_failed_feed_is_partial(tmp_path: Path) -> None:
    def fake_run(cfg: config.Config) -> tuple[int, str]:
        if "b.example" in (cfg.rss_url or ""):
            raise RuntimeError("simulated feed failure")
        return (2, "ok")

    code, emit = _multi(tmp_path, fake_run)
    assert code == 1
    completed = _run_events(emit)[-1][1]
    assert completed["status"] == "partial"
    assert (completed["ok"], completed["failed"], completed["episodes_processed"]) == (1, 1, 2)


def test_multi_feed_all_feeds_failing_is_failed(tmp_path: Path) -> None:
    def fake_run(cfg: config.Config) -> tuple[int, str]:
        raise RuntimeError("down")

    code, emit = _multi(tmp_path, fake_run)
    assert code == 1
    completed = _run_events(emit)[-1][1]
    assert completed["status"] == "failed"
    assert completed["ok"] == 0
    assert completed["error_class"] == "NonZeroExit"


def test_crash_still_reports_failed_with_error_class_and_reraises(tmp_path: Path) -> None:
    def fake_run(cfg: config.Config) -> tuple[int, str]:
        raise KeyboardInterrupt

    with patch(_EMIT) as emit, pytest.raises(KeyboardInterrupt):
        cli.main(
            [_A, "--rss", _B, "--output-dir", str(tmp_path)],
            run_pipeline_fn=fake_run,
        )
    events = _run_events(emit)
    assert [t for t, _ in events] == ["pipeline_run.started", "pipeline_run.completed"]
    completed = events[-1][1]
    assert completed["status"] == "failed"
    assert completed["error_class"] == "KeyboardInterrupt"
    assert completed["not_started"] == 2


def test_single_feed_success(tmp_path: Path) -> None:
    code, emit = _single(tmp_path, lambda cfg: (4, "ok"), "--run-id", "job-2")
    assert code == 0
    (_, started), (_, completed) = _run_events(emit)
    assert started["feeds"] == 1
    assert started["run_id"] == "job-2"
    assert completed["status"] == "succeeded"
    assert (completed["ok"], completed["failed"], completed["not_started"]) == (1, 0, 0)
    assert completed["episodes_processed"] == 4


def test_single_feed_exception_reports_failed_with_the_exception_class(tmp_path: Path) -> None:
    def fake_run(cfg: config.Config) -> tuple[int, str]:
        raise ValueError("boom")

    code, emit = _single(tmp_path, fake_run)
    assert code == 1
    completed = _run_events(emit)[-1][1]
    assert completed["status"] == "failed"
    assert completed["error_class"] == "ValueError"
    assert (completed["ok"], completed["failed"]) == (0, 1)


def test_auto_run_id_is_not_reported(tmp_path: Path) -> None:
    _, emit = _single(tmp_path, lambda cfg: (1, "ok"), "--run-id", "auto")
    assert _run_events(emit)[0][1].get("run_id") is None


def test_subcommand_emits_no_run_events() -> None:
    with patch(_EMIT) as emit:
        cli.main(["corpus-status", "--help"])
    assert _run_events(emit) == []


def test_emit_failure_does_not_change_the_exit_code(tmp_path: Path) -> None:
    with patch(_EMIT, side_effect=RuntimeError("sink down")):
        code = cli.main(
            [_A, "--output-dir", str(tmp_path), "--max-episodes", "1"],
            run_pipeline_fn=lambda cfg: (1, "ok"),
        )
    assert code == 0


class TestTriggerFlag:
    def test_defaults_to_unknown(self) -> None:
        assert cli.parse_args([_A, "--output-dir", "/tmp/x"]).trigger == "unknown"

    def test_env_supplies_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_RUN_TRIGGER", "backfill")
        assert cli.parse_args([_A, "--output-dir", "/tmp/x"]).trigger == "backfill"

    def test_invalid_env_falls_back_to_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_RUN_TRIGGER", "cronjob")
        assert cli.parse_args([_A, "--output-dir", "/tmp/x"]).trigger == "unknown"

    def test_flag_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_RUN_TRIGGER", "backfill")
        args = cli.parse_args([_A, "--output-dir", "/tmp/x", "--trigger", "manual"])
        assert args.trigger == "manual"

    def test_rejects_unknown_choice(self) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args([_A, "--output-dir", "/tmp/x", "--trigger", "cronjob"])
