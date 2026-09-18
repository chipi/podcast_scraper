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
    """Stub every enqueuer in the SHARED dispatcher; unnamed ones return []."""
    import importlib

    from podcast_scraper.server import app_digest_dispatch

    for label, module_name, func_name in app_digest_dispatch.ENQUEUERS:
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


def test_neither_scheduler_owns_a_private_enqueuer_list(tmp_path: Path, monkeypatch) -> None:
    """THE REGRESSION GUARD for #2119, at cause.

    The bug was two hand-maintained dispatch lists. ``daily_recap`` (#2039) and the monthly
    ``recommendations`` digest were added to ``server/scheduler.py`` and not to the sidecar; since
    the player runs ONLY the sidecar (``PODCAST_SERVE_APP_ONLY=1``, ADR-116), both shipped dead to
    production and stayed dead — silently, because the sidecar logged a healthy 'enqueued 0'.

    The fix is that there is now exactly ONE list, in ``app_digest_dispatch.ENQUEUERS``. This test
    fails if either scheduler starts calling ``enqueue_due_*`` directly again, which is the only
    way the two paths can diverge.
    """
    import re

    sources = {
        "server/scheduler.py": _REPO_ROOT / "src" / "podcast_scraper" / "server" / "scheduler.py",
        "infra/deploy/digest_scheduler.py": _RUNNER,
    }
    for label, path in sources.items():
        direct = re.findall(
            r"app_digest_\w+\.(enqueue_due_\w+)\(", path.read_text(encoding="utf-8")
        )
        assert not direct, (
            f"{label} calls {sorted(set(direct))} directly instead of going through "
            "app_digest_dispatch.enqueue_all_due(). That reintroduces a second dispatch list, "
            "which is exactly how #2119 shipped two digests dead to prod."
        )


def _json_lines(out: str) -> list[dict]:
    import json

    return [
        json.loads(line.split("] ", 1)[1])
        for line in out.splitlines()
        if line.startswith("[digest-scheduler-json] ")
    ]


def test_structured_log_emits_correlation_id_per_envelope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """THE trace join (#2119).

    The delivery worker already stamps ``correlation_id`` (= the envelope id) on every span,
    log, metric and the outbound Resend header, so a delivered email is traceable end to end —
    but only FORWARDS. Without this, "when was this envelope enqueued, and by which cadence?"
    required regex over a free-text tick line. The sidecar cannot emit OTEL spans
    (``network_mode: none``), so a structured line carrying the same key is the correlation its
    isolation permits.
    """
    mod = _load(tmp_path, monkeypatch)
    _stub_all(mod, monkeypatch, weekly=["dgst_2026W38_u_abc"], daily_recap=["drcp_20260918_u_abc"])
    mod._run_once()

    events = _json_lines(capsys.readouterr().out)
    enqueued = [e for e in events if e["event"] == "digest.envelope.enqueued"]
    assert len(enqueued) == 2
    by_id = {e["correlation_id"]: e for e in enqueued}
    assert by_id["dgst_2026W38_u_abc"]["cadence"] == "weekly"
    assert by_id["drcp_20260918_u_abc"]["cadence"] == "daily_recap"
    # Every event carries the key the delivery side joins on.
    assert all("correlation_id" in e for e in enqueued)


def test_structured_tick_carries_per_cadence_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    """Machine-readable counts per cadence, so a dashboard or query never has to parse prose."""
    from podcast_scraper.server import app_digest_dispatch

    mod = _load(tmp_path, monkeypatch)
    _stub_all(mod, monkeypatch, weekly=["a", "b"])
    mod._run_once()

    ticks = [e for e in _json_lines(capsys.readouterr().out) if e["event"] == "digest.tick"]
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick["total"] == 2
    assert tick["cadences"]["weekly"] == 2
    # EVERY cadence is present with an explicit count — a missing key would reintroduce the
    # "is it zero or is it never called?" ambiguity that hid #2119.
    assert set(tick["cadences"]) == {label for label, _, _ in app_digest_dispatch.ENQUEUERS}
    assert tick["errored"] == []


def test_structured_log_reports_a_failing_cadence(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load(tmp_path, monkeypatch)
    _stub_all(mod, monkeypatch, weekly=RuntimeError("outbox unreachable"), daily_recap=["x"])
    mod._run_once()

    events = _json_lines(capsys.readouterr().out)
    failed = [e for e in events if e["event"] == "digest.enqueuer.failed"]
    assert [e["cadence"] for e in failed] == ["weekly"]
    assert "outbox unreachable" in failed[0]["error"]
    tick = next(e for e in events if e["event"] == "digest.tick")
    assert tick["errored"] == ["weekly"]
    assert tick["cadences"]["daily_recap"] == 1  # healthy cadence still reported


def test_structured_lines_are_valid_json(tmp_path: Path, monkeypatch, capsys) -> None:
    """A line that does not parse is worse than no line — the log pipeline would drop it
    silently, which is the same class of failure as the bug this all came from."""
    import json

    mod = _load(tmp_path, monkeypatch)
    _stub_all(mod, monkeypatch, weekly=["a"])
    mod._run_once()
    raw = [
        line for line in capsys.readouterr().out.splitlines() if "[digest-scheduler-json] " in line
    ]
    assert raw
    for line in raw:
        json.loads(line.split("] ", 1)[1])  # raises if malformed


def test_sidecar_drives_every_shared_enqueuer(tmp_path: Path, monkeypatch, capsys) -> None:
    """End to end through the real sidecar entrypoint: every enqueuer in the shared list is
    driven, and each one's count is reported separately."""
    from podcast_scraper.server import app_digest_dispatch

    mod = _load(tmp_path, monkeypatch)
    _stub_all(
        mod, monkeypatch, **{label: [f"{label}-1"] for label, _, _ in app_digest_dispatch.ENQUEUERS}
    )
    mod._run_once()
    out = capsys.readouterr().out
    for label, _, _ in app_digest_dispatch.ENQUEUERS:
        assert f"{label}=1" in out, f"sidecar did not report {label}"
        assert f"{label}-1" in out, f"sidecar dropped {label}'s envelope id"
