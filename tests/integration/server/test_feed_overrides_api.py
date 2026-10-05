"""Operator overrides (#2283) on the operator API: who may change them, what is stored, audited.

The endpoint lives at ``/api/feeds/overrides`` beside the feed list, on the tailnet control plane
(operator, 2026-10-05). The rules under test:

* the operator guard: an admin session or the operator key; anyone else 403 — reads included;
* it is NOT on the consumer surface (``/api/app/admin/overrides`` is gone) and not mounted without
  the feeds API;
* a value the pipeline could not apply is refused (422): an unknown field, a non-ISO language, a
  non-http image, a blank url or guid;
* a language is stored as its ISO code (``English`` -> ``en``);
* every change is audited with WHO (``via`` + ``by``) and the value before and after;
* a cookie write from a foreign origin is refused (``app_csrf``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper import overrides
from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_user_store import create_user

pytestmark = [pytest.mark.integration]

FEED = "https://example.com/feed.xml"
KEY = "op-key-for-tests"
BASE = "/api/feeds/overrides"


def _app(tmp_path: Path, *, feeds_api: bool = True):
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=feeds_api)
    app.state.operator_api_key = KEY
    app.state.audit_path = tmp_path / "appdata" / "audit.jsonl"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    return app


def _as(app, role: str) -> tuple[TestClient, str]:
    user = create_user(
        app.state.app_data_dir,
        provider="mock",
        subject=role,
        email=f"{role}@x.io",
        name=role,
        role=role,
    )
    client = TestClient(app)
    cookie = app_sessions.sign(
        {"user_id": user.user_id, "iat": int(time.time())}, app.state.session_secret
    )
    client.cookies.set(app_sessions.SESSION_COOKIE, cookie)
    return client, user.user_id


def _keyed(app) -> TestClient:
    return TestClient(app, headers={"X-Operator-Key": KEY})


def _p(tmp_path: Path, **extra: str) -> Dict[str, str]:
    return {"path": str(tmp_path), **extra}


def _audit(tmp_path: Path) -> List[Dict[str, Any]]:
    path = tmp_path / "appdata" / "audit.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if str(r.get("action", "")).startswith("override_")]


# --- who may use it ---------------------------------------------------------------------------

_ROUTES = [
    ("get", BASE, {}),
    ("put", f"{BASE}/feed", {"url": FEED}),
    ("delete", f"{BASE}/feed", {"url": FEED}),
    ("put", f"{BASE}/episode", {"url": FEED, "guid": "g1"}),
    ("delete", f"{BASE}/episode", {"url": FEED, "guid": "g1"}),
]


@pytest.mark.parametrize("method,path,params", _ROUTES)
def test_only_an_admin_or_the_operator_key_may_use_it(
    tmp_path: Path, method: str, path: str, params: Dict[str, str]
) -> None:
    app = _app(tmp_path)
    kwargs: Dict[str, Any] = {"params": _p(tmp_path, **params)}
    if method == "put":
        kwargs["json"] = {"title": "x"}
    assert getattr(TestClient(app), method)(path, **kwargs).status_code == 403
    for role in ("listener", "creator"):
        client, _ = _as(app, role)
        assert getattr(client, method)(path, **kwargs).status_code == 403, role
    assert not overrides.overrides_path(tmp_path).exists(), "a refused write must change nothing"
    admin, _ = _as(app, "admin")
    assert getattr(admin, method)(path, **kwargs).status_code != 403
    assert getattr(_keyed(app), method)(path, **kwargs).status_code != 403


def test_it_is_not_on_the_consumer_surface(tmp_path: Path) -> None:
    admin, _ = _as(_app(tmp_path), "admin")
    assert admin.get("/api/app/admin/overrides").status_code == 404


def test_it_is_not_mounted_without_the_feeds_api(tmp_path: Path) -> None:
    # The public planes mount no feeds API; there, the path does not exist at all.
    app = _app(tmp_path, feeds_api=False)
    assert _keyed(app).get(BASE, params=_p(tmp_path)).status_code == 404


# --- what is stored ---------------------------------------------------------------------------


def test_an_admin_sets_a_feed_language_and_it_is_stored_as_iso(tmp_path: Path) -> None:
    admin, _ = _as(_app(tmp_path), "admin")
    r = admin.put(f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "English"})
    assert r.status_code == 200, r.text
    assert r.json()["fields"] == {"language": "en"}
    stored = overrides.load_overrides(tmp_path)
    assert overrides.feed_fields_for(stored, FEED).language == "en"  # type: ignore[union-attr]
    listed = admin.get(BASE, params=_p(tmp_path)).json()
    assert listed["feeds"][FEED]["fields"] == {"language": "en"}


def test_the_store_is_the_one_the_pipeline_reads(tmp_path: Path) -> None:
    # Written beside feeds.spec.yaml at the corpus root — where apply_overrides looks.
    _keyed(_app(tmp_path)).put(
        f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "en"}
    )
    assert (tmp_path / overrides.OVERRIDES_BASENAME).is_file()


def test_an_episode_override_is_keyed_by_feed_and_guid(tmp_path: Path) -> None:
    body = {
        "title": "Corrected title",
        "published_date": "2026-01-02",
        "hosts": ["Jane Doe"],
        "speaker_renames": {"Jon Smyth": "John Smith"},
    }
    r = _keyed(_app(tmp_path)).put(
        f"{BASE}/episode", params=_p(tmp_path, url=FEED, guid="g1"), json=body
    )
    assert r.status_code == 200, r.text
    ep = overrides.episode_fields_for(overrides.load_overrides(tmp_path), FEED, "g1")
    assert ep is not None and ep.title == "Corrected title" and ep.hosts == ["Jane Doe"]
    assert overrides.episode_fields_for(overrides.load_overrides(tmp_path), FEED, "g2") is None


@pytest.mark.parametrize(
    "body",
    [
        {"language": "klingon"},
        {"language": "und"},
        {"image_url": "ftp://example.com/a.png"},
        {"not_a_field": 1},
    ],
)
def test_a_value_the_pipeline_could_not_apply_is_refused(tmp_path: Path, body: dict) -> None:
    r = _keyed(_app(tmp_path)).put(f"{BASE}/feed", params=_p(tmp_path, url=FEED), json=body)
    assert r.status_code == 422
    assert not overrides.overrides_path(tmp_path).exists()


@pytest.mark.parametrize(
    "path,params",
    [
        (f"{BASE}/feed", {"url": "   "}),
        (f"{BASE}/episode", {"url": FEED, "guid": "   "}),
    ],
)
def test_a_blank_key_is_a_422_not_a_500(tmp_path: Path, path: str, params: dict) -> None:
    r = _keyed(_app(tmp_path)).put(path, params=_p(tmp_path, **params), json={"title": "T"})
    assert r.status_code == 422
    assert not overrides.overrides_path(tmp_path).exists()


def test_a_path_outside_the_anchor_is_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent
    r = _keyed(_app(tmp_path)).put(
        f"{BASE}/feed", params={"path": str(outside), "url": FEED}, json={"language": "en"}
    )
    assert r.status_code in (400, 403)
    assert not (outside / overrides.OVERRIDES_BASENAME).exists()


def test_deleting_a_feed_removes_its_episodes_too(tmp_path: Path) -> None:
    client = _keyed(_app(tmp_path))
    client.put(f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "en"})
    client.put(f"{BASE}/episode", params=_p(tmp_path, url=FEED, guid="g1"), json={"title": "T"})
    assert client.delete(f"{BASE}/feed", params=_p(tmp_path, url=FEED)).status_code == 200
    assert overrides.load_overrides(tmp_path).feeds == {}
    assert client.delete(f"{BASE}/feed", params=_p(tmp_path, url=FEED)).status_code == 404


# --- what is audited --------------------------------------------------------------------------


def test_every_change_records_who_and_before_and_after(tmp_path: Path) -> None:
    app = _app(tmp_path)
    admin, me = _as(app, "admin")
    admin.put(f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "en"})
    admin.put(f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "es"})
    admin.delete(f"{BASE}/feed", params=_p(tmp_path, url=FEED))
    rows = _audit(tmp_path)
    assert [r["action"] for r in rows] == [
        "override_feed_set",
        "override_feed_set",
        "override_feed_deleted",
    ]
    assert all(r["via"] == "admin_session" and r["by"] == me for r in rows)
    assert rows[0]["before"] is None and rows[0]["after"] == {"language": "en"}
    assert rows[1]["before"] == {"language": "en"} and rows[1]["after"] == {"language": "es"}
    assert rows[2]["after"] is None


def test_a_key_change_is_audited_as_the_key(tmp_path: Path) -> None:
    _keyed(_app(tmp_path)).put(
        f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "en"}
    )
    (row,) = _audit(tmp_path)
    assert row["via"] == "operator_key" and "by" not in row


def test_a_refused_write_is_not_audited_as_a_change(tmp_path: Path) -> None:
    client, _ = _as(_app(tmp_path), "creator")
    client.put(f"{BASE}/feed", params=_p(tmp_path, url=FEED), json={"language": "en"})
    assert _audit(tmp_path) == []


# --- cross-site ---------------------------------------------------------------------------------


def test_a_cookie_write_from_a_foreign_origin_is_refused(tmp_path: Path) -> None:
    admin, _ = _as(_app(tmp_path), "admin")
    r = admin.put(
        f"{BASE}/feed",
        params=_p(tmp_path, url=FEED),
        json={"language": "en"},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403
    assert not overrides.overrides_path(tmp_path).exists()
