"""Unit tests for scheduler error paths and edge cases (#708 coverage).

Pure mocks — no APScheduler dependency. Covers the import-fallback,
prometheus counter no-op, and ``next_run_at`` edge branches that
``test_scheduler_lifecycle.py`` (integration) cannot reach.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.server import scheduler as sched_mod
from podcast_scraper.server.scheduler import (
    _read_operator_yaml,
    SchedulerService,
)


def test_read_operator_yaml_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert _read_operator_yaml(tmp_path / "missing.yaml") == ""


def test_read_operator_yaml_returns_empty_on_oserror(tmp_path: Path) -> None:
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("max_episodes: 1\n", encoding="utf-8")
    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        assert _read_operator_yaml(op) == ""


def test_record_triggered_is_noop_when_counter_unavailable() -> None:
    """``_record_triggered`` swallows when prometheus_client is missing."""
    with patch.object(sched_mod, "_TRIGGERED_COUNTER", None):
        sched_mod._record_triggered("any-name")  # no error raised


def test_record_failed_is_noop_when_counter_unavailable() -> None:
    with patch.object(sched_mod, "_FAILED_COUNTER", None):
        sched_mod._record_failed("any-name", "any-reason")


def test_record_triggered_swallows_counter_exception() -> None:
    """Even if the prometheus call raises, the scheduler keeps running."""
    fake_counter = MagicMock()
    fake_counter.labels.side_effect = RuntimeError("registry corrupted")
    with patch.object(sched_mod, "_TRIGGERED_COUNTER", fake_counter):
        sched_mod._record_triggered("any-name")  # no error raised


def test_record_failed_swallows_counter_exception() -> None:
    fake_counter = MagicMock()
    fake_counter.labels.side_effect = RuntimeError("registry corrupted")
    with patch.object(sched_mod, "_FAILED_COUNTER", fake_counter):
        sched_mod._record_failed("any-name", "any-reason")


def test_ensure_counters_idempotent() -> None:
    """Second call is a no-op when counters already initialised."""
    fake_existing = MagicMock()
    with patch.object(sched_mod, "_TRIGGERED_COUNTER", fake_existing):
        sched_mod._ensure_counters()
        # No new init happened: still the same fake object.
        assert sched_mod._TRIGGERED_COUNTER is fake_existing


def test_ensure_counters_handles_prometheus_import_error() -> None:
    """When prometheus_client is missing, _ensure_counters silently no-ops."""
    with (
        patch.object(sched_mod, "_TRIGGERED_COUNTER", None),
        patch.object(sched_mod, "_FAILED_COUNTER", None),
        patch.dict("sys.modules", {"prometheus_client": None}),
    ):
        sched_mod._ensure_counters()
        # Counters stay None when the import fails.
        assert sched_mod._TRIGGERED_COUNTER is None
        assert sched_mod._FAILED_COUNTER is None


class TestSchedulerServiceEdges:
    @pytest.fixture()
    def operator_yaml(self, tmp_path: Path) -> Path:
        op = tmp_path / "viewer_operator.yaml"
        op.write_text("max_episodes: 1\n", encoding="utf-8")
        return op

    def test_next_run_at_returns_none_when_scheduler_not_started(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        # Scheduler hasn't been started — _scheduler is None.
        assert svc.next_run_at("any") is None

    def test_next_run_at_returns_none_when_get_job_raises(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        fake_sch = MagicMock()
        fake_sch.get_job.side_effect = RuntimeError("scheduler shutting down")
        svc._scheduler = fake_sch
        assert svc.next_run_at("morning") is None

    def test_next_run_at_returns_none_when_job_missing(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        fake_sch = MagicMock()
        fake_sch.get_job.return_value = None
        svc._scheduler = fake_sch
        assert svc.next_run_at("morning") is None

    def test_next_run_at_returns_none_when_job_has_no_run_time(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        fake_job = MagicMock()
        fake_job.next_run_time = None
        fake_sch = MagicMock()
        fake_sch.get_job.return_value = fake_job
        svc._scheduler = fake_sch
        assert svc.next_run_at("morning") is None

    def test_next_run_at_swallows_astimezone_error(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        broken_dt = MagicMock()
        broken_dt.astimezone.side_effect = ValueError("naive datetime")
        fake_job = MagicMock()
        fake_job.next_run_time = broken_dt
        fake_sch = MagicMock()
        fake_sch.get_job.return_value = fake_job
        svc._scheduler = fake_sch
        assert svc.next_run_at("morning") is None

    def test_running_property_false_when_scheduler_none(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        assert svc.running is False

    def test_running_property_uses_scheduler_running_attr(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        fake_sch = MagicMock()
        fake_sch.running = True
        svc._scheduler = fake_sch
        assert svc.running is True

    def test_start_is_idempotent_when_scheduler_already_set(
        self, tmp_path: Path, operator_yaml: Path
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        existing = MagicMock()
        svc._scheduler = existing
        svc.start()  # short-circuits without re-creating
        assert svc._scheduler is existing

    def test_start_warns_and_noops_when_apscheduler_missing(
        self, tmp_path: Path, operator_yaml: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        operator_yaml.write_text(
            "scheduled_jobs:\n  - name: x\n    cron: '0 4 * * *'\n", encoding="utf-8"
        )
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        with patch.dict("sys.modules", {"apscheduler.schedulers.background": None}):
            with caplog.at_level("WARNING"):
                svc.start()
        assert svc._scheduler is None
        assert any("apscheduler" in rec.message for rec in caplog.records)

    def test_shutdown_swallows_scheduler_error(
        self, tmp_path: Path, operator_yaml: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        fake_sch = MagicMock()
        fake_sch.shutdown.side_effect = RuntimeError("already shut down")
        svc._scheduler = fake_sch
        with caplog.at_level("WARNING"):
            svc.shutdown()  # no exception propagated
        assert svc._scheduler is None
        assert any("shutdown error" in rec.message for rec in caplog.records)

    def test_timezone_property_returns_string(self, tmp_path: Path, operator_yaml: Path) -> None:
        svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
        assert isinstance(svc.timezone, str)
        assert len(svc.timezone) > 0


def test_scheduler_timezone_picks_podcast_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_SCHEDULER_TZ", "Europe/Berlin")
    monkeypatch.setenv("TZ", "America/New_York")
    assert sched_mod._scheduler_timezone() == "Europe/Berlin"


def test_scheduler_timezone_falls_back_to_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_SCHEDULER_TZ", raising=False)
    monkeypatch.setenv("TZ", "America/New_York")
    assert sched_mod._scheduler_timezone() == "America/New_York"


def test_scheduler_timezone_defaults_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_SCHEDULER_TZ", raising=False)
    monkeypatch.delenv("TZ", raising=False)
    assert sched_mod._scheduler_timezone() == "UTC"
