"""The gauges must actually register when the REAL app is built (#2119).

This exists because the component tests passed while production exported nothing.

``install_metrics()`` was unit- and integration-tested by calling it directly with a real
path, and all of it passed. But it was WIRED inside ``_install_metrics()``, which
``create_app`` calls ~27 lines BEFORE ``_configure_platform_auth()`` — the function that sets
``app.state.app_data_dir``. So the guard read ``None``, skipped, logged nothing (the except
branch only fires on a raised exception, and "guard was falsy" is not one), and prod ran with
the state file being written correctly and no gauges at all.

The lesson is narrow and worth keeping: testing a component is not testing its installation.
These go through ``create_app`` so the ORDER is what is under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("fastapi")

pytestmark = pytest.mark.integration


def _collected_names(registry) -> set[str]:
    names = set()
    for family in registry.collect():
        names.add(family.name)
    return names


def _build(tmp_path: Path, monkeypatch, *, metrics: bool = True, data_dir: bool = True):
    """Build a real app via create_app with an isolated Prometheus registry."""
    import prometheus_client

    from podcast_scraper.server.app import create_app

    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1" if metrics else "0")
    out = tmp_path / "output"
    (out / ".app").mkdir(parents=True, exist_ok=True)
    if data_dir:
        monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
        (tmp_path / "appdata").mkdir(parents=True, exist_ok=True)
    else:
        monkeypatch.delenv("APP_DATA_DIR", raising=False)

    # Isolated registry: the global one persists across tests in a session and a duplicate
    # registration would either raise or silently no-op, hiding the very thing under test.
    reg = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(prometheus_client, "REGISTRY", reg)
    import podcast_scraper.server.app_digest_health as h

    monkeypatch.setattr(h, "REGISTRY", reg, raising=False)

    app = create_app(output_dir=out)
    return app, reg


def test_gauges_register_when_the_real_app_is_built(tmp_path: Path, monkeypatch) -> None:
    """THE regression guard. Fails if the install is ever moved back above the line that sets
    app_data_dir, which is exactly how this shipped broken."""
    app, reg = _build(tmp_path, monkeypatch)
    assert app.state.app_data_dir is not None, "app_data_dir must be set by create_app"
    names = _collected_names(reg)
    assert "podcast_digest_last_success_age_seconds" in names, (
        "the digest health gauges did not register on a real create_app — check that the "
        "install still runs AFTER _configure_platform_auth sets app.state.app_data_dir"
    )
    assert "podcast_digest_last_run_age_seconds" in names
    assert "podcast_digest_consenting_users" in names


def test_app_data_dir_is_set_before_the_install_runs(tmp_path: Path, monkeypatch) -> None:
    """Pin the ordering directly, so a failure says WHICH invariant broke rather than just
    'a gauge is missing'."""
    import podcast_scraper.server.app_digest_health as h

    seen: list[object] = []
    orig = h.install_metrics

    def _spy(app, data_dir):
        seen.append(data_dir)
        return orig(app, data_dir)

    monkeypatch.setattr(h, "install_metrics", _spy)
    _build(tmp_path, monkeypatch)
    assert seen, "install_metrics was never called by create_app"
    assert seen[0] is not None


def test_no_gauges_when_metrics_disabled(tmp_path: Path, monkeypatch) -> None:
    """Default posture stays a no-op — nothing registers unless metrics are switched on."""
    _, reg = _build(tmp_path, monkeypatch, metrics=False)
    assert "podcast_digest_last_success_age_seconds" not in _collected_names(reg)


def test_app_builds_even_if_the_install_raises(tmp_path: Path, monkeypatch) -> None:
    """Telemetry never breaks the app (ADR-120) — a failing install must not stop create_app."""
    import podcast_scraper.server.app_digest_health as h

    def _boom(app, data_dir):
        raise RuntimeError("registry on fire")

    monkeypatch.setattr(h, "install_metrics", _boom)
    app, _ = _build(tmp_path, monkeypatch)  # must not raise
    assert app is not None
