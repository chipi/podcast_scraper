"""Unit tests for the observability GitOps sync scripts (ADR-117).

``scripts/obs/{grafana,sentry}_sync.py`` push dashboards/alert-rules to per-tenant
accounts. The risky logic is the account env-name mapping, the dry-run/apply gate
(a bug that POSTs on a dry run — or silently no-ops on apply — is the danger), and
the disabled-tenant skip. These are loaded by path (scripts/ is not a package).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "obs" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


grafana = _load("grafana_sync")
sentry = _load("sentry_sync")


class _Resp:
    status = 201

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def read(self) -> bytes:
        return b"{}"


def _urlopen_recorder(calls: list[str]):
    def _fake(req, timeout: int = 30):  # noqa: ANN001
        calls.append(req.full_url)
        return _Resp()

    return _fake


# --- account env-name mapping (account upper-cased, '-' -> '_') --------------- #


def test_grafana_acct_env_mapping() -> None:
    assert grafana._acct_env("common", "URL") == "GRAFANA_COMMON_URL"
    assert grafana._acct_env("podcast-player", "TOKEN") == "GRAFANA_PODCAST_PLAYER_TOKEN"


def test_sentry_acct_mapping() -> None:
    assert sentry._acct("common", "ORG") == "SENTRY_COMMON_ORG"
    assert sentry._acct("podcast-operator", "TOKEN") == "SENTRY_PODCAST_OPERATOR_TOKEN"


# --- the dry-run / apply gate (both scripts) --------------------------------- #


def test_grafana_post_gate(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(grafana.urllib.request, "urlopen", _urlopen_recorder(calls))
    grafana._post("https://g", "tok", "/api/folders", {"title": "x"}, apply=False)
    assert calls == [], "dry run must make no request"
    grafana._post("https://g", "tok", "/api/folders", {"title": "x"}, apply=True)
    assert calls == ["https://g/api/folders"], "apply must POST once"


def test_sentry_post_gate(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sentry.urllib.request, "urlopen", _urlopen_recorder(calls))
    sentry._post("https://sentry.io", "tok", "org", "proj", {"name": "r"}, apply=False)
    assert calls == [], "dry run must make no request"
    sentry._post("https://sentry.io", "tok", "org", "proj", {"name": "r"}, apply=True)
    assert calls == ["https://sentry.io/api/0/projects/org/proj/rules/"], "apply must POST once"


# --- disabled tenants never reach the network -------------------------------- #


def test_grafana_disabled_tenant_skipped(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(grafana.urllib.request, "urlopen", _urlopen_recorder(calls))
    grafana._sync_tenant("player", {"enabled": False}, apply=True)
    assert calls == []


def test_sentry_disabled_tenant_skipped(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sentry.urllib.request, "urlopen", _urlopen_recorder(calls))
    sentry._sync_tenant("player", {"enabled": False}, apply=True)
    assert calls == []


# --- folders, the vps tenant and the dry-run diff (2026-10-15) ---------------- #
#
# The mini Grafana had the 8 common dashboards + beta-usage in General (dashboards were
# POSTed without folderUid), vps/ had no tenant (podcast-app reached Grafana only through a
# separate script with a dead token), and the dry run printed a POST per file whatever the
# live state, so "lists what would change" listed everything.


class _Grafana:
    """A fake Grafana: GETs answer from *live*, every request is recorded as (method, path)."""

    def __init__(self, live: dict) -> None:
        self.live = live
        self.calls: list[tuple[str, str]] = []

    def urlopen(self, req, timeout: int = 30):  # noqa: ANN001
        import email.message
        import io
        import json
        import urllib.error

        path = req.full_url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path
        self.calls.append((req.get_method(), path))
        if req.get_method() == "GET":
            if path not in self.live:
                raise urllib.error.HTTPError(
                    req.full_url, 404, "nf", email.message.Message(), io.BytesIO(b"{}")
                )
            body = json.dumps(self.live[path]).encode()

            class R(_Resp):
                status = 200

                def read(self) -> bytes:
                    return body

            return R()
        return _Resp()


def _tenant_dir(tmp_path, monkeypatch, dashboards: dict) -> None:
    import json

    monkeypatch.setattr(grafana, "REPO", tmp_path)
    d = tmp_path / "config" / "grafana" / "dashboards" / "t"
    d.mkdir(parents=True)
    for name, dash in dashboards.items():
        (d / name).write_text(json.dumps(dash))


def _env(monkeypatch) -> None:
    monkeypatch.setenv("GRAFANA_COMMON_URL", "http://g")
    monkeypatch.setenv("GRAFANA_COMMON_TOKEN", "tok")


def test_dashboards_are_posted_into_the_tenant_folder(tmp_path, monkeypatch) -> None:
    import json

    _tenant_dir(tmp_path, monkeypatch, {"a.json": {"uid": "a", "title": "A"}})
    _env(monkeypatch)
    sent: list[dict] = []
    fake = _Grafana({})

    def urlopen(req, timeout: int = 30):  # noqa: ANN001
        if req.get_method() == "POST" and req.full_url.endswith("/api/dashboards/db"):
            sent.append(json.loads(req.data))
        return fake.urlopen(req, timeout)

    monkeypatch.setattr(grafana.urllib.request, "urlopen", urlopen)
    grafana._sync_tenant("t", {"grafana": {"folder": "VPS — Podcast"}}, apply=True)
    assert sent and sent[0]["folderUid"] == "vps-podcast"


def test_dry_run_with_credentials_diffs_and_only_reads(tmp_path, monkeypatch, capsys) -> None:
    _tenant_dir(
        tmp_path,
        monkeypatch,
        {
            "same.json": {"uid": "same", "title": "S", "panels": []},
            "changed.json": {"uid": "changed", "title": "C", "panels": [{"title": "new"}]},
            "moved.json": {"uid": "moved", "title": "M"},
            "new.json": {"uid": "new", "title": "N"},
        },
    )
    _env(monkeypatch)
    folder = {"meta": {"folderUid": "f"}}
    fake = _Grafana(
        {
            "/api/dashboards/uid/same": {
                **folder,
                "dashboard": {"uid": "same", "title": "S", "panels": [], "id": 7, "version": 3},
            },
            "/api/dashboards/uid/changed": {
                **folder,
                "dashboard": {"uid": "changed", "title": "C", "panels": []},
            },
            "/api/dashboards/uid/moved": {
                "meta": {"folderUid": ""},
                "dashboard": {"uid": "moved", "title": "M"},
            },
        }
    )
    monkeypatch.setattr(grafana.urllib.request, "urlopen", fake.urlopen)
    grafana._sync_tenant("t", {"grafana": {"folder": "F"}}, apply=False)
    out = capsys.readouterr().out
    assert {m for m, _ in fake.calls} == {"GET"}, "a dry run must never write"
    assert "UNCHANGED same.json" in out
    assert "CHANGED   changed.json" in out
    assert "MOVE      moved.json" in out
    assert "NEW       new.json" in out


def test_tenant_without_dashboards_creates_no_folder(tmp_path, monkeypatch) -> None:
    _tenant_dir(tmp_path, monkeypatch, {})
    _env(monkeypatch)
    fake = _Grafana({})
    monkeypatch.setattr(grafana.urllib.request, "urlopen", fake.urlopen)
    grafana._sync_tenant("t", {"grafana": {"folder": "F"}}, apply=True)
    assert fake.calls == []


def test_alert_rules_are_not_synced_from_here() -> None:
    # agentic-ai-homelab's rules.yaml owns them; a second writer here regressed live rules.
    assert not (REPO / "config" / "grafana" / "alerts").exists()
    assert "alert-rules" not in (REPO / "scripts" / "obs" / "grafana_sync.py").read_text()


def test_every_dashboard_dir_has_a_tenant() -> None:
    import yaml

    tenants = yaml.safe_load((REPO / "config" / "obs" / "tenants.yaml").read_text())["tenants"]
    dirs = {p.name for p in (REPO / "config" / "grafana" / "dashboards").iterdir() if p.is_dir()}
    assert dirs <= set(tenants), f"dashboards no tenant syncs: {sorted(dirs - set(tenants))}"


def test_vps_tenant_folder_is_the_live_vps_podcast_folder() -> None:
    import yaml

    tenants = yaml.safe_load((REPO / "config" / "obs" / "tenants.yaml").read_text())["tenants"]
    assert grafana._slug(tenants["vps"]["grafana"]["folder"]) == "vps-podcast"
