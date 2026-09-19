"""``podcast_pipeline_last_success_age_seconds`` must register on the REAL app (#2119).

The sibling of ``test_app_digest_health_wiring.py``, and for the same reason: a gauge that is
component-tested but mis-wired exports nothing in production, silently, because "the guard was
falsy" never raises.

This gauge carries the ``podcast-ingest-stalled`` alert, which previously queried the
VictoriaLogs tail of the job registry. That could not work — the registry is a mutable file
rewritten whole on every status change, while Alloy tails it assuming append-only — and over 30
days the alert's query matched on exactly one day, the day the file happened to be re-read from
offset zero. Replacing a signal that never worked with one that is mis-wired would be no
improvement, so the installation is what these tests exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("fastapi")

pytestmark = pytest.mark.integration


def _collected(registry) -> dict[str, list]:
    return {f.name: list(f.samples) for f in registry.collect()}


def _build(tmp_path: Path, monkeypatch, *, metrics: bool = True, jobs: list | None = None):
    import prometheus_client

    from podcast_scraper.server.app import create_app

    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1" if metrics else "0")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
    (tmp_path / "appdata").mkdir(parents=True, exist_ok=True)

    out = tmp_path / "output"
    (out / ".app").mkdir(parents=True, exist_ok=True)
    if jobs is not None:
        vdir = out / ".viewer"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "jobs.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in jobs), encoding="utf-8"
        )

    # Isolated registry — the global one persists across the session, and a duplicate
    # registration would silently no-op and hide the thing under test.
    reg = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(prometheus_client, "REGISTRY", reg)
    import podcast_scraper.server.pipeline_run_prometheus as prm

    monkeypatch.setattr(prm, "REGISTRY", reg, raising=False)

    return create_app(output_dir=out), reg


def test_gauge_registers_when_the_real_app_is_built(tmp_path: Path, monkeypatch) -> None:
    """THE regression guard: fails if the install moves above the line setting output_dir."""
    app, reg = _build(
        tmp_path,
        monkeypatch,
        jobs=[
            {
                "job_id": "a",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    assert app.state.output_dir is not None, "output_dir must be set by create_app"
    assert "podcast_pipeline_last_success_age_seconds" in _collected(reg), (
        "the job age gauge did not register on a real create_app — check the install still "
        "runs AFTER app.state.output_dir is set"
    )


def test_gauge_reads_the_registry_at_scrape_time(tmp_path: Path, monkeypatch) -> None:
    """Scrape-time collection is the point: a job finishing after boot must show up without
    the app being restarted."""
    app, reg = _build(tmp_path, monkeypatch, jobs=[])
    assert _collected(reg)["podcast_pipeline_last_success_age_seconds"] == []

    # Write a success AFTER the app was built; the next scrape must see it.
    vdir = Path(app.state.output_dir) / ".viewer"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "jobs.jsonl").write_text(
        json.dumps(
            {
                "job_id": "late",
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "ended_at": "2026-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = _collected(reg)["podcast_pipeline_last_success_age_seconds"]
    assert [s.labels["command_type"] for s in samples] == ["full_incremental_pipeline"]
    assert samples[0].value > 0


def test_never_succeeded_exports_no_sample(tmp_path: Path, monkeypatch) -> None:
    """Absent, not zero. A zero reads as 'succeeded just now' and inverts the alert."""
    _, reg = _build(
        tmp_path,
        monkeypatch,
        jobs=[
            {
                "job_id": "f",
                "command_type": "full_incremental_pipeline",
                "status": "failed",
                "ended_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    assert _collected(reg)["podcast_pipeline_last_success_age_seconds"] == []


def test_no_gauge_when_metrics_disabled(tmp_path: Path, monkeypatch) -> None:
    _, reg = _build(tmp_path, monkeypatch, metrics=False, jobs=[])
    assert "podcast_pipeline_last_success_age_seconds" not in _collected(reg)


def test_app_builds_even_if_the_install_raises(tmp_path: Path, monkeypatch) -> None:
    """Telemetry never breaks the app (ADR-120)."""
    import podcast_scraper.server.pipeline_run_prometheus as prm

    monkeypatch.setattr(
        prm, "install_job_age_metrics", lambda _root: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    app, _ = _build(tmp_path, monkeypatch, jobs=[])  # must not raise
    assert app is not None


def test_an_unreadable_registry_does_not_break_the_scrape(tmp_path: Path, monkeypatch) -> None:
    """A corrupt registry must degrade to 'no sample', not raise inside /metrics — a scrape
    that 500s takes down every other metric on the endpoint too."""
    app, reg = _build(tmp_path, monkeypatch, jobs=[])
    vdir = Path(app.state.output_dir) / ".viewer"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "jobs.jsonl").write_bytes(b"\xff\xfe not json at all\n")
    assert _collected(reg)["podcast_pipeline_last_success_age_seconds"] == []
