"""Unit tests for scripts/obs/health_drill.py (#1819) — probes with mocked backends.

Loaded by path like test_obs_sync.py (scripts/ is not a package). Every probe is
exercised on its PASS and FAIL branches with injected Grafana/VictoriaLogs
responses — no network, no tokens.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "health_drill", REPO / "scripts" / "obs" / "health_drill.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclasses + `from __future__ import annotations` resolve field types via
    # sys.modules[cls.__module__] — a path-loaded module must register itself first.
    sys.modules["health_drill"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def drill(monkeypatch):
    mod = _load()
    monkeypatch.setenv("GRAFANA_URL", "http://grafana.test")
    monkeypatch.setenv("GRAFANA_OBS_TOKEN", "t")
    return mod


class TestDatasourceHealth:
    def test_all_healthy_passes(self, drill, monkeypatch):
        def fake(path, **_kw):
            if path == "/api/datasources":
                return [{"uid": "a"}, {"uid": "b"}]
            return {"status": "OK"}

        monkeypatch.setattr(drill, "_grafana", fake)
        v = drill.probe_datasource_health()
        assert v.status == "PASS" and "2 datasources" in v.evidence

    def test_unhealthy_datasource_fails_with_uid_named(self, drill, monkeypatch):
        def fake(path, **_kw):
            if path == "/api/datasources":
                return [{"uid": "good"}, {"uid": "dead"}]
            if "dead" in path:
                raise RuntimeError("boom")
            return {"status": "OK"}

        monkeypatch.setattr(drill, "_grafana", fake)
        v = drill.probe_datasource_health()
        assert v.status == "FAIL" and "dead=" in v.evidence

    def test_plugin_without_health_resource_falls_back_to_a_real_probe(self, drill, monkeypatch):
        """The tempo plugin has NO health resource (404 plugin.notImplemented) while its
        backend is fine — the drill must probe THROUGH the datasource instead of calling
        the missing endpoint unhealthy (#1819 false alarm, 2026-08-25)."""
        calls = []

        def fake(path, **_kw):
            calls.append(path)
            if path == "/api/datasources":
                return [{"uid": "victoriatraces-tempo", "type": "tempo"}]
            if path.endswith("/health"):
                raise drill.urllib.error.HTTPError(path, 404, "notImplemented", {}, None)
            if "/proxy/uid/victoriatraces-tempo/api/echo" in path:
                return {}
            raise RuntimeError(f"unexpected {path}")

        monkeypatch.setattr(drill, "_grafana", fake)
        v = drill.probe_datasource_health()
        assert v.status == "PASS", v.evidence
        assert any("proxy" in c for c in calls), "no fallback probe was attempted"

    def test_no_health_resource_and_dead_backend_still_fails(self, drill, monkeypatch):
        def fake(path, **_kw):
            if path == "/api/datasources":
                return [{"uid": "victoriatraces-tempo", "type": "tempo"}]
            if path.endswith("/health"):
                raise drill.urllib.error.HTTPError(path, 404, "notImplemented", {}, None)
            raise RuntimeError("backend down")

        monkeypatch.setattr(drill, "_grafana", fake)
        v = drill.probe_datasource_health()
        assert v.status == "FAIL" and "victoriatraces-tempo" in v.evidence


class TestMetricFreshness:
    @staticmethod
    def _rows(*ages):
        return {
            "data": {
                "result": [
                    {"metric": {"job": f"j{i}", "instance": "x"}, "value": [0, str(a)]}
                    for i, a in enumerate(ages)
                ]
            }
        }

    def test_fresh_targets_pass(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_grafana", lambda *_a, **_k: self._rows(5, 60))
        v = drill.probe_metric_freshness()
        assert v.status == "PASS"

    def test_stale_target_fails(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_grafana", lambda *_a, **_k: self._rows(5, 999))
        v = drill.probe_metric_freshness()
        assert v.status == "FAIL" and "j1@x" in v.evidence

    def test_zero_series_fails(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_grafana", lambda *_a, **_k: self._rows())
        v = drill.probe_metric_freshness()
        assert v.status == "FAIL"


class TestAlertRules:
    @staticmethod
    def _rules(*healths):
        return {
            "data": {
                "groups": [
                    {"rules": [{"name": f"r{i}", "health": h} for i, h in enumerate(healths)]}
                ]
            }
        }

    def test_all_ok_passes(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_grafana", lambda *_a, **_k: self._rules("ok", "ok"))
        assert drill.probe_alert_rules().status == "PASS"

    def test_error_rule_fails_named(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_grafana", lambda *_a, **_k: self._rules("ok", "error"))
        v = drill.probe_alert_rules()
        assert v.status == "FAIL" and "r1" in v.evidence

    def test_zero_rules_fails(self, drill, monkeypatch):
        # The 2026-08-24 lesson baked in: an alert stack with no rules is not healthy.
        monkeypatch.setattr(drill, "_grafana", lambda *_a, **_k: self._rules())
        assert drill.probe_alert_rules().status == "FAIL"


class TestLogProbes:
    def test_logs_flowing_pass_and_fail(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_vlogs_count", lambda _q: 42)
        assert drill.probe_logs_flowing().status == "PASS"
        monkeypatch.setattr(drill, "_vlogs_count", lambda _q: 0)
        assert drill.probe_logs_flowing().status == "FAIL"

    def test_dedup_skip_when_quiet(self, drill, monkeypatch):
        monkeypatch.setattr(drill, "_vlogs_count", lambda _q: 0)
        assert drill.probe_log_dedup().status == "SKIP"

    def test_dedup_fail_on_duplication(self, drill, monkeypatch):
        counts = iter([804, 402])
        monkeypatch.setattr(drill, "_vlogs_count", lambda _q: next(counts))
        v = drill.probe_log_dedup()
        assert v.status == "FAIL" and "x2.0" in v.evidence

    def test_dedup_pass_when_unique(self, drill, monkeypatch):
        counts = iter([400, 398])
        monkeypatch.setattr(drill, "_vlogs_count", lambda _q: next(counts))
        assert drill.probe_log_dedup().status == "PASS"


class TestErrorPlaneAndMain:
    def test_error_plane_skips_without_dsn(self, drill, monkeypatch):
        monkeypatch.delenv("DRILL_SENTRY_DSN", raising=False)
        assert drill.probe_error_plane().status == "SKIP"

    def test_main_exit_codes(self, drill, monkeypatch, capsys):
        monkeypatch.setattr(drill, "PROBES", [lambda: drill.Verdict("p", "PASS", "e")])
        assert drill.main() == 0
        monkeypatch.setattr(drill, "PROBES", [lambda: drill.Verdict("p", "SKIP", "e")])
        assert drill.main() == 2
        monkeypatch.setattr(
            drill,
            "PROBES",
            [lambda: (_ for _ in ()).throw(RuntimeError("crash"))],
        )
        assert drill.main() == 1
        out = capsys.readouterr().out
        assert "probe crashed" in out

    def test_main_requires_env(self, drill, monkeypatch):
        monkeypatch.delenv("GRAFANA_URL", raising=False)
        assert drill.main() == 1
