"""enrichment source probes: not-configured, success, error, filtering, eval scan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_obs.config import TargetConfig
from podcast_obs.sources import enrichment


def _target(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


# --- run_status --------------------------------------------------------------------


def test_run_status_not_configured() -> None:
    result = enrichment.run_status(_target())
    assert result["ok"] is False
    assert result["configured"] is False


def test_run_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        enrichment, "get_json", lambda url, **_: {"available": True, "status": "ok"}
    )
    result = enrichment.run_status(_target(api_base="http://x"))
    assert result["ok"] is True
    assert result["data"]["status"] == "ok"


def test_run_status_transport_error_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, **_):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(enrichment, "get_json", boom)
    result = enrichment.run_status(_target(api_base="http://x"))
    assert result["ok"] is False
    assert result["configured"] is True
    assert "connection refused" in result["error"]


# --- recent_runs (filtering on command_type) ---------------------------------------


def test_recent_runs_filters_to_corpus_enrichment(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "path": "/corpus",
        "jobs": [
            {
                "job_id": "p1",
                "command_type": "full_incremental_pipeline",
                "created_at": "2026-06-01T00:00:00Z",
            },
            {
                "job_id": "e1",
                "command_type": "corpus_enrichment",
                "created_at": "2026-06-02T00:00:00Z",
            },
            {
                "job_id": "e2",
                "command_type": "corpus_enrichment",
                "created_at": "2026-06-03T00:00:00Z",
            },
        ],
    }
    monkeypatch.setattr(enrichment, "get_json", lambda url, **_: payload)
    result = enrichment.recent_runs(_target(api_base="http://x"), limit=10)
    assert result["ok"] is True
    ids = [r["job_id"] for r in result["data"]["runs"]]
    assert ids == ["e2", "e1"]  # newest first; pipeline job filtered out
    assert result["data"]["count"] == 2


def test_recent_runs_not_configured() -> None:
    result = enrichment.recent_runs(_target())
    assert result["ok"] is False
    assert result["configured"] is False


# --- health (with optional enricher_id) -------------------------------------------


def test_health_passes_enricher_id_param(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get_json(url, *, params=None, **_):
        captured["url"] = url
        captured["params"] = params
        return {"available": False}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    enrichment.health(_target(api_base="http://x"), enricher_id="topic_similarity")
    assert captured["params"] == {"enricher_id": "topic_similarity"}


def test_health_no_enricher_id_passes_none_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get_json(url, *, params=None, **_):
        captured["params"] = params
        return {"enrichers": {}}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    enrichment.health(_target(api_base="http://x"))
    assert captured["params"] is None


# --- metrics ------------------------------------------------------------------------


def test_metrics_default_window(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get_json(url, *, params=None, **_):
        captured["params"] = params
        return {"window": "24h", "per_enricher": {}}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    enrichment.metrics(_target(api_base="http://x"))
    assert captured["params"] == {"window": "24h"}


def test_metrics_custom_window(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get_json(url, *, params=None, **_):
        captured["params"] = params
        return {"window": "1h"}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    enrichment.metrics(_target(api_base="http://x"), window="1h")
    assert captured["params"] == {"window": "1h"}


# --- run_summary -------------------------------------------------------------------


def test_run_summary_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        enrichment, "get_json", lambda url, **_: {"status": "ok", "per_enricher": {}}
    )
    result = enrichment.run_summary(_target(api_base="http://x"))
    assert result["ok"] is True
    assert result["data"]["status"] == "ok"


# --- recent_events -----------------------------------------------------------------


def test_recent_events_filters_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get_json(url, *, params=None, **_):
        captured["params"] = params
        return {"events": [], "count": 0}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    enrichment.recent_events(
        _target(api_base="http://x"),
        enricher_id="topic_cooccurrence",
        event_type="enrichment.enricher.completed",
        limit=25,
    )
    assert captured["params"] == {
        "limit": 25,
        "enricher_id": "topic_cooccurrence",
        "event_type": "enrichment.enricher.completed",
    }


def test_recent_events_omits_falsy_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get_json(url, *, params=None, **_):
        captured["params"] = params
        return {"events": []}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    enrichment.recent_events(_target(api_base="http://x"))
    assert captured["params"] == {"limit": 50}


# --- eval_history (on-disk scan) ---------------------------------------------------


def test_eval_history_no_root_is_unconfigured(tmp_path: Path) -> None:
    result = enrichment.eval_history(_target(api_base="http://x"), eval_root=str(tmp_path / "nope"))
    assert result["ok"] is False
    assert result["configured"] is False


def test_eval_history_picks_up_enrichment_prefixed_runs(tmp_path: Path) -> None:
    eval_root = tmp_path / "runs"
    eval_root.mkdir()
    (eval_root / "enrichment-2026-06-01").mkdir()
    (eval_root / "enrichment-2026-06-02").mkdir()
    (eval_root / "autoresearch-2026-06-03").mkdir()  # not enrichment-tagged

    result = enrichment.eval_history(_target(api_base="http://x"), eval_root=str(eval_root))
    assert result["ok"] is True
    names = [r["run_id"] for r in result["data"]["runs"]]
    # sorted reverse alphabetically — newest first by id convention
    assert names == ["enrichment-2026-06-02", "enrichment-2026-06-01"]


def test_eval_history_uses_metadata_kind_marker(tmp_path: Path) -> None:
    eval_root = tmp_path / "runs"
    eval_root.mkdir()
    run_dir = eval_root / "untagged-2026-06-04"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"kind": "enrichment", "scorer": "topic_similarity"}),
        encoding="utf-8",
    )
    result = enrichment.eval_history(_target(api_base="http://x"), eval_root=str(eval_root))
    assert result["ok"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["runs"][0]["metadata"]["kind"] == "enrichment"


def test_eval_history_respects_limit(tmp_path: Path) -> None:
    eval_root = tmp_path / "runs"
    eval_root.mkdir()
    for n in range(5):
        (eval_root / f"enrichment-{n:02d}").mkdir()
    result = enrichment.eval_history(
        _target(api_base="http://x"), eval_root=str(eval_root), limit=2
    )
    assert result["data"]["count"] == 2


def test_eval_history_tolerates_malformed_metadata(tmp_path: Path) -> None:
    eval_root = tmp_path / "runs"
    eval_root.mkdir()
    run_dir = eval_root / "enrichment-bad"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("not json{", encoding="utf-8")
    result = enrichment.eval_history(_target(api_base="http://x"), eval_root=str(eval_root))
    assert result["ok"] is True
    assert result["data"]["runs"][0]["metadata"] == {}


# --- re_enable ---------------------------------------------------------------------


def test_re_enable_posts_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_post(url, *, json=None, **_):
        captured["url"] = url
        captured["json"] = json
        return {"enricher_id": "x", "auto_disabled": False, "consecutive_failures": 0}

    monkeypatch.setattr(enrichment, "post_json", fake_post)
    result = enrichment.re_enable(_target(api_base="http://x"), "x", reason="operator override")
    assert result["ok"] is True
    assert captured["json"] == {"reason": "operator override"}
    assert captured["url"].endswith("/api/enrichment/health/x/re-enable")


def test_re_enable_no_reason_sends_empty_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_post(url, *, json=None, **_):
        captured["json"] = json
        return {"enricher_id": "x", "auto_disabled": False}

    monkeypatch.setattr(enrichment, "post_json", fake_post)
    enrichment.re_enable(_target(api_base="http://x"), "x")
    assert captured["json"] == {}


def test_re_enable_not_configured() -> None:
    result = enrichment.re_enable(_target(), "x")
    assert result["ok"] is False
    assert result["configured"] is False


# --- cancel ------------------------------------------------------------------------


def test_cancel_targets_jobs_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_post(url, **_):
        captured["url"] = url
        return {"job_id": "j1", "status": "cancelled"}

    monkeypatch.setattr(enrichment, "post_json", fake_post)
    result = enrichment.cancel(_target(api_base="http://x"), "j1")
    assert result["ok"] is True
    assert captured["url"].endswith("/api/jobs/j1/cancel")


def test_cancel_transport_error_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, **_):
        raise RuntimeError("timeout")

    monkeypatch.setattr(enrichment, "post_json", boom)
    result = enrichment.cancel(_target(api_base="http://x"), "j1")
    assert result["ok"] is False
    assert result["configured"] is True
