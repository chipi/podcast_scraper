"""Gauge export for per-cadence delivery health (#2119).

Integration tier because it needs the real ``prometheus_client`` (an optional extra) — the
3-tier policy forbids ``importorskip`` in tests/unit, and mocking the registry would test the
mock rather than the collector. The filesystem side of this module is covered by
tests/unit/podcast_scraper/server/test_app_digest_health.py.

What these lock: absence is meaningful. A cadence that has NEVER succeeded must be ABSENT from
the success-age family rather than reported as zero — the alert distinguishes "never happened"
from "just happened" purely by that, and reporting 0 would invert the alert.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("prometheus_client")

from podcast_scraper.server import app_digest_dispatch, app_digest_health  # noqa: E402

pytestmark = pytest.mark.integration


def _result(ids=None, errors=None):
    return app_digest_dispatch.DispatchResult(ids=ids or {}, errors=errors or {})


def _collect(tmp_path: Path) -> dict[str, dict[str, float]]:
    """Build the collector without mutating the global REGISTRY.

    Registering twice in one session raises, and a swallowed duplicate-registration error would
    silently make these assertions test nothing.
    """
    import prometheus_client

    captured: list = []
    orig = prometheus_client.REGISTRY.register
    prometheus_client.REGISTRY.register = lambda c: captured.append(c)  # type: ignore[assignment]
    try:
        assert app_digest_health.install_metrics(object(), tmp_path) is True
    finally:
        prometheus_client.REGISTRY.register = orig  # type: ignore[assignment]

    out: dict[str, dict[str, float]] = {}
    for family in captured[0].collect():
        # Sample.labels is a dict keyed by label NAME; these families carry exactly one label.
        out[family.name] = {
            next(iter(s.labels.values())): s.value for s in family.samples if s.labels
        }
    return out


def test_success_age_is_exported_per_cadence(tmp_path: Path) -> None:
    now = int(time.time())
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["a"]}), now=now - 100)
    ages = _collect(tmp_path)["podcast_digest_last_success_age_seconds"]
    assert 90 <= ages["weekly"] <= 130  # ~100s, allowing clock granularity


def test_never_succeeded_cadence_is_absent_not_zero(tmp_path: Path) -> None:
    """THE load-bearing property. Exporting 0 would read as 'succeeded just now' and the alert
    would never fire — which is the #2119 failure re-created in the telemetry."""
    app_digest_health.record_dispatch(
        tmp_path, _result(ids={"weekly": ["a"], "daily_recap": []}), now=int(time.time())
    )
    families = _collect(tmp_path)
    assert "daily_recap" not in families["podcast_digest_last_success_age_seconds"]
    # But it IS in run-age: that pair separates "never called" from "called, nobody due".
    assert "daily_recap" in families["podcast_digest_last_run_age_seconds"]


def test_error_age_exported_for_a_failing_cadence(tmp_path: Path) -> None:
    app_digest_health.record_dispatch(
        tmp_path,
        _result(ids={"weekly": ["a"]}, errors={"daily_recap": "boom"}),
        now=int(time.time()),
    )
    families = _collect(tmp_path)
    assert "daily_recap" in families["podcast_digest_last_error_age_seconds"]
    assert "weekly" not in families["podcast_digest_last_error_age_seconds"]


def test_consenting_users_exported_as_the_denominator(tmp_path: Path) -> None:
    """Without this the age alerts cannot tell 'nobody consented' from 'broken', and an alert
    that cries wolf on an empty roster gets muted."""
    app_digest_health.record_dispatch(
        tmp_path,
        _result(ids={"weekly": ["a"]}),
        consenting={"digest": 2, "daily_recap": 0},
        now=int(time.time()),
    )
    consenting = _collect(tmp_path)["podcast_digest_consenting_users"]
    assert consenting["digest"] == 2
    assert consenting["daily_recap"] == 0


def test_totals_exported(tmp_path: Path) -> None:
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["a", "b"]}), now=1)
    app_digest_health.record_dispatch(tmp_path, _result(ids={"weekly": ["c"]}), now=2)
    assert _collect(tmp_path)["podcast_digest_enqueued_total"]["weekly"] == 3


def test_exporter_is_quiet_when_no_state_exists(tmp_path: Path) -> None:
    """Pre-deploy, or before the first tick: export nothing rather than zeros. The alerts use
    noDataState: OK, so silence here is correct silence."""
    families = _collect(tmp_path)
    assert families["podcast_digest_last_success_age_seconds"] == {}
    assert families["podcast_digest_consenting_users"] == {}
