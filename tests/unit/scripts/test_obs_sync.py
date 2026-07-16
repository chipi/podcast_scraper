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
