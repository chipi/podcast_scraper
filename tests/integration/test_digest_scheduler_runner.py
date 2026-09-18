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


def _stub_all(mod, monkeypatch, **returns):
    """Stub every enqueuer in ``_ENQUEUERS``; unnamed ones return []."""
    import importlib

    for label, module_name, func_name in mod._ENQUEUERS:
        module = importlib.import_module(f"podcast_scraper.server.{module_name}")
        value = returns.get(label, [])
        if isinstance(value, Exception):

            def _boom(root, data, _exc=value):
                raise _exc

            monkeypatch.setattr(module, func_name, _boom)
        else:
            monkeypatch.setattr(module, func_name, lambda root, data, _v=value: _v)


def test_run_once_reports_per_enqueuer_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    """A bare total is ambiguous across several enqueuers, and that ambiguity is what hid #2119:
    the sidecar logged a healthy 'enqueued 0' for months while daily_recap was never called."""
    mod = _load(tmp_path, monkeypatch)
    _stub_all(mod, monkeypatch, weekly=["a", "b"], daily_recap=["c"])
    mod._run_once()
    out = capsys.readouterr().out
    assert "enqueued 3 envelope(s)" in out
    assert "weekly=2" in out
    assert "daily_recap=1" in out
    assert "recommendations=0" in out


def test_one_failing_enqueuer_does_not_mute_the_others(tmp_path: Path, monkeypatch, capsys) -> None:
    """The cadences are independent. A broken assembler in one must not silence every
    notification the product has — which is what coupling them would do."""
    mod = _load(tmp_path, monkeypatch)
    _stub_all(
        mod,
        monkeypatch,
        weekly=RuntimeError("outbox unreachable"),
        daily_recap=["still-delivered"],
    )
    mod._run_once()
    out = capsys.readouterr().out
    assert "enqueuer weekly failed" in out
    assert "weekly=ERR" in out
    assert "daily_recap=1" in out  # the healthy one still ran
    assert "still-delivered" in out


def test_cycle_survives_enqueue_error_and_still_beats(tmp_path: Path, monkeypatch, capsys) -> None:
    """The crux: a failing enqueue must be caught (loop survives) AND the heartbeat still fires
    (the container stays healthy; a persistently-empty loop is caught by the homelab dead-man)."""
    mod = _load(tmp_path, monkeypatch)
    _stub_all(mod, monkeypatch, weekly=RuntimeError("outbox unreachable"))
    mod._cycle()  # must NOT raise
    assert "enqueuer weekly failed" in capsys.readouterr().out
    assert (tmp_path / "hb" / "tick").exists()  # beat happened despite the error


def test_cycle_survives_a_non_enqueuer_error(tmp_path: Path, monkeypatch, capsys) -> None:
    """Per-enqueuer isolation must not remove the outer backstop: anything else that throws in a
    cycle still has to be caught and still has to beat."""
    mod = _load(tmp_path, monkeypatch)

    def boom() -> None:
        raise RuntimeError("something else entirely")

    monkeypatch.setattr(mod, "_run_once", boom)
    mod._cycle()  # must NOT raise
    assert "cycle error" in capsys.readouterr().out
    assert (tmp_path / "hb" / "tick").exists()


def test_enqueuers_match_the_in_process_scheduler(tmp_path: Path, monkeypatch) -> None:
    """THE REGRESSION GUARD for #2119.

    ``infra/deploy/digest_scheduler.py`` is a separate process from
    ``server/scheduler.py``, and prod runs ONLY the sidecar. When ``daily_recap`` (#2039) and the
    monthly ``recommendations`` digest were added to the in-process scheduler and not to the
    sidecar, both shipped dead to production and stayed dead — silently, because the sidecar kept
    logging a healthy 'enqueued 0 envelope(s)'.

    This fails the moment ``scheduler.py`` calls an ``enqueue_due_*`` the sidecar does not.
    """
    import re

    mod = _load(tmp_path, monkeypatch)
    scheduler_src = (_REPO_ROOT / "src" / "podcast_scraper" / "server" / "scheduler.py").read_text(
        encoding="utf-8"
    )

    called = set(re.findall(r"(app_digest_\w+)\.(enqueue_due_\w+)\(", scheduler_src))
    assert called, "found no enqueue_due_* calls in scheduler.py — did the call shape change?"

    declared = {(module_name, func_name) for _, module_name, func_name in mod._ENQUEUERS}
    missing = called - declared
    assert not missing, (
        "scheduler.py drives enqueuers the production sidecar does not call, so they are DEAD in "
        f"prod: {sorted(missing)}. Add them to _ENQUEUERS in infra/deploy/digest_scheduler.py."
    )
