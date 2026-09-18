"""Per-cadence delivery health (#2119) — the signal that catches "output never arrived".

The bug these guard against: every pre-existing alert on the digest chain watched liveness or
needed successful traffic to already exist, so two cadences ran dead for months behind green
dashboards. The operator's requirement is explicit — detect the case where the scheduler is
healthy and exactly ONE cadence is broken.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import app_digest_dispatch, app_digest_health

pytestmark = pytest.mark.unit


def _result(ids=None, errors=None):
    return app_digest_dispatch.DispatchResult(ids=ids or {}, errors=errors or {})


def _state(tmp_path: Path) -> dict[str, Any]:
    raw = (tmp_path / app_digest_health.STATE_FILENAME).read_text(encoding="utf-8")
    loaded: dict[str, Any] = json.loads(raw)
    return loaded


def test_records_last_success_per_cadence(tmp_path: Path) -> None:
    app_digest_health.record_dispatch(
        tmp_path, _result(ids={"weekly": ["a"], "daily_recap": []}), now=1000
    )
    cad = _state(tmp_path)["cadences"]
    assert cad["weekly"]["last_success_ts"] == 1000
    assert cad["weekly"]["last_success_count"] == 1
    # Ran, produced nothing — NOT a success, and not an error either.
    assert cad["daily_recap"]["last_success_ts"] is None
    assert cad["daily_recap"]["last_run_ts"] == 1000


def test_idle_pass_carries_last_success_forward(tmp_path: Path) -> None:
    """THE crux of an age metric: 'nobody was due this hour' must neither look like a fresh
    success nor reset the age. Get this wrong and the alert can never fire."""
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["a"]}), now=1000)
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": []}), now=5000)
    cad = _state(tmp_path)["cadences"]
    assert cad["weekly"]["last_success_ts"] == 1000  # carried forward, not bumped to 5000
    assert cad["weekly"]["last_run_ts"] == 5000  # but we know it DID run


def test_one_broken_cadence_is_visible_while_others_are_healthy(tmp_path: Path) -> None:
    """The operator's stated requirement: the scheduler can be fine while exactly one cadence
    fails, and that must be distinguishable."""
    app_digest_health.record_dispatch(
        tmp_path,
        _result(ids={"weekly": ["a"], "recommendations": ["b"]}, errors={"daily_recap": "boom"}),
        now=1000,
    )
    cad = _state(tmp_path)["cadences"]
    assert cad["weekly"]["last_success_ts"] == 1000
    assert cad["recommendations"]["last_success_ts"] == 1000
    assert cad["daily_recap"]["last_error"] == "boom"
    assert cad["daily_recap"]["last_error_ts"] == 1000
    assert cad["daily_recap"]["last_success_ts"] is None


def test_totals_accumulate_across_passes(tmp_path: Path) -> None:
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["a", "b"]}), now=1)
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["c"]}), now=2)
    assert _state(tmp_path)["cadences"]["weekly"]["total_enqueued"] == 3


def test_record_dispatch_never_raises_on_unwritable_dir(tmp_path: Path) -> None:
    """Health bookkeeping must not be able to break the delivery loop it measures."""
    target = tmp_path / "file-not-a-dir"
    target.write_text("x", encoding="utf-8")
    app_digest_health.record_dispatch(target, _result(ids={"weekly": ["a"]}), now=1)  # no raise


def test_corrupt_state_is_a_fresh_start_not_a_crash(tmp_path: Path) -> None:
    (tmp_path / app_digest_health.STATE_FILENAME).write_text("{not json", encoding="utf-8")
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["a"]}), now=7)
    assert _state(tmp_path)["cadences"]["weekly"]["last_success_ts"] == 7


def test_read_state_returns_empty_when_absent(tmp_path: Path) -> None:
    assert app_digest_health.read_state(tmp_path) == {}


def test_dispatch_writes_health_state_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Through the real dispatcher, not the recorder directly."""
    import importlib

    for label, module_name, func_name in app_digest_dispatch.ENQUEUERS:
        module = importlib.import_module(f"podcast_scraper.server.{module_name}")
        monkeypatch.setattr(
            module, func_name, lambda root, data, _l=label: ([f"{_l}_x"] if _l == "weekly" else [])
        )

    app_digest_dispatch.enqueue_all_due(tmp_path / "corpus", tmp_path)
    cad = _state(tmp_path)["cadences"]
    assert set(cad) == {label for label, _, _ in app_digest_dispatch.ENQUEUERS}
    assert cad["weekly"]["last_success_ts"] is not None
    assert cad["daily_recap"]["last_success_ts"] is None  # ran, nobody due


def test_health_failure_does_not_break_dispatch(tmp_path: Path, monkeypatch) -> None:
    """If the recorder throws, the dispatch result must still be returned intact."""
    import importlib

    for _, module_name, func_name in app_digest_dispatch.ENQUEUERS:
        module = importlib.import_module(f"podcast_scraper.server.{module_name}")
        monkeypatch.setattr(module, func_name, lambda root, data: ["x"])

    def _boom(*a, **k):
        raise RuntimeError("health store on fire")

    monkeypatch.setattr(app_digest_health, "record_dispatch", _boom)
    res = app_digest_dispatch.enqueue_all_due(tmp_path / "corpus", tmp_path)
    assert res.total == len(app_digest_dispatch.ENQUEUERS)  # dispatch unaffected


def test_consenting_users_counts_email_channel_only(tmp_path: Path, monkeypatch) -> None:
    """The denominator. Zero envelopes is CORRECT with an empty roster; without this the age
    alerts cannot tell that from a fault."""
    from podcast_scraper.server import app_comms_store, app_user_store

    class _U:
        def __init__(self, uid):
            self.user_id = uid

    monkeypatch.setattr(app_user_store, "list_users", lambda d: [_U("u1"), _U("u2")])

    def _comms(data_dir, uid):
        if uid == "u1":  # email on for digest, off for daily_recap
            return {"types": {"digest": {"email": True}, "daily_recap": {"email": False}}}
        return {"types": {"digest": {"email": True}, "daily_recap": {"email": True}}}

    monkeypatch.setattr(app_comms_store, "get_comms", _comms)
    counts = app_digest_health.count_consenting_users(tmp_path)
    assert counts["digest"] == 2
    assert counts["daily_recap"] == 1


def test_consenting_users_survives_one_unreadable_user(tmp_path: Path, monkeypatch) -> None:
    from podcast_scraper.server import app_comms_store, app_user_store

    class _U:
        def __init__(self, uid):
            self.user_id = uid

    monkeypatch.setattr(app_user_store, "list_users", lambda d: [_U("bad"), _U("good")])

    def _comms(data_dir, uid):
        if uid == "bad":
            raise OSError("permission denied")
        return {"types": {"digest": {"email": True}}}

    monkeypatch.setattr(app_comms_store, "get_comms", _comms)
    assert app_digest_health.count_consenting_users(tmp_path)["digest"] == 1


# The gauge-EXPORT test needs the real ``prometheus_client`` (an optional extra), so it lives in
# tests/integration/server/test_app_digest_health_metrics.py — the 3-tier policy forbids
# importorskip in tests/unit, and mocking the registry would test the mock rather than the
# collector. Everything above is pure filesystem work and stays here.
