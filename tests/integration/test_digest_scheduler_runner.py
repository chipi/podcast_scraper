"""Unit-ish coverage for the delivery digest-scheduler sidecar runner (#1412).

The runner (``infra/deploy/digest_scheduler.py``) is a standalone script mounted into the compose
sidecar, not part of the package — loaded here by path. These lock the loop behaviour the sidecar
depends on: interval alignment (never skip a slot hour), heartbeat liveness, and — most importantly
— that one bad enqueue cycle never kills the loop.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # _run_once imports the server package

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "infra" / "deploy" / "digest_scheduler.py"


def _load(tmp_path: Path, monkeypatch, *, interval: int = 3600, offset: int = 120):
    monkeypatch.setenv("DIGEST_HEARTBEAT_FILE", str(tmp_path / "hb" / "tick"))
    monkeypatch.setenv("DIGEST_INTERVAL_SECONDS", str(interval))
    monkeypatch.setenv("DIGEST_INTERVAL_OFFSET_SECONDS", str(offset))
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("DIGEST_CORPUS_ROOT", str(tmp_path / "corpus"))
    spec = importlib.util.spec_from_file_location("digest_scheduler_under_test", _RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sleep_aligns_to_top_of_interval(tmp_path: Path, monkeypatch) -> None:
    """A fixed sleep drifts and eventually skips a slot hour; the runner must fire at a stable
    per-interval offset so every clock interval gets exactly one fire."""
    mod = _load(tmp_path, monkeypatch, interval=3600, offset=120)
    captured: dict[str, float] = {}
    for now in (1_000.0, 1_700_000_000.5, 1_700_003_599.9):  # arbitrary points across an interval
        monkeypatch.setattr(mod.time, "time", lambda now=now: now)
        mod._sleep_to_next_interval(sleep=lambda d: captured.__setitem__("d", d))
        fire_at = now + captured["d"]
        assert fire_at % 3600 == 120  # always lands at HH:02:00
        assert 0 < captured["d"] <= 3600 + 120


def test_beat_writes_heartbeat(tmp_path: Path, monkeypatch) -> None:
    mod = _load(tmp_path, monkeypatch)
    mod._beat()
    beat = (tmp_path / "hb" / "tick").read_text(encoding="utf-8")
    assert beat.isdigit()  # a unix timestamp


def test_run_once_reports_enqueued_count(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load(tmp_path, monkeypatch)
    from podcast_scraper.server import app_digest_personal

    monkeypatch.setattr(app_digest_personal, "enqueue_due_digests", lambda root, data: ["a", "b"])
    mod._run_once()
    assert "enqueued 2 envelope(s)" in capsys.readouterr().out


def test_cycle_survives_enqueue_error_and_still_beats(tmp_path: Path, monkeypatch, capsys) -> None:
    """The crux: a failing enqueue must be caught (loop survives) AND the heartbeat still fires
    (the container stays healthy; a persistently-empty loop is caught by the homelab dead-man)."""
    mod = _load(tmp_path, monkeypatch)
    from podcast_scraper.server import app_digest_personal

    def boom(root, data):
        raise RuntimeError("outbox unreachable")

    monkeypatch.setattr(app_digest_personal, "enqueue_due_digests", boom)
    mod._cycle()  # must NOT raise
    assert "cycle error" in capsys.readouterr().out
    assert (tmp_path / "hb" / "tick").exists()  # beat happened despite the error
