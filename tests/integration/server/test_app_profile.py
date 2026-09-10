"""Route tests for the profile avatar upload/serve surface (Area E)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # valid signature + filler (not decoded)
_JPEG = b"\xff\xd8\xff" + b"\x00" * 64
_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64


def _authed(tmp_path: Path) -> tuple[TestClient, Path, str]:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, data_dir, user.user_id


def test_upload_then_me_points_at_it_and_it_serves(tmp_path: Path) -> None:
    client, _, uid = _authed(tmp_path)
    resp = client.post("/api/app/profile/avatar", files={"file": ("a.png", _PNG, "image/png")})
    assert resp.status_code == 200, resp.text
    served = resp.json()["image"]
    assert served.startswith(f"/api/app/profile/{uid}/avatar")
    # /me now carries the uploaded avatar (overriding any OAuth one).
    assert client.get("/api/app/me").json()["image"] == served
    # And the served route returns the bytes with the right type.
    got = client.get(f"/api/app/profile/{uid}/avatar")
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"
    assert got.content == _PNG


def test_upload_rejects_unsupported_type(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    resp = client.post("/api/app/profile/avatar", files={"file": ("a.gif", _PNG, "image/gif")})
    assert resp.status_code == 415


def test_upload_rejects_content_type_mismatch(tmp_path: Path) -> None:
    # Declares PNG but sends JPEG bytes → magic-byte sniff rejects it.
    client, _, _ = _authed(tmp_path)
    resp = client.post("/api/app/profile/avatar", files={"file": ("a.png", _JPEG, "image/png")})
    assert resp.status_code == 400


def test_upload_rejects_oversize(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024 + 1)
    resp = client.post("/api/app/profile/avatar", files={"file": ("a.png", big, "image/png")})
    assert resp.status_code == 413


def test_reupload_replaces_prior_format(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    client.post("/api/app/profile/avatar", files={"file": ("a.png", _PNG, "image/png")})
    client.post("/api/app/profile/avatar", files={"file": ("a.webp", _WEBP, "image/webp")})
    # Exactly one avatar file remains (the png was removed).
    avatars = sorted((data_dir / "users" / uid).glob("avatar.*"))
    assert [p.name for p in avatars] == ["avatar.webp"]


def test_serve_404_for_unknown_or_unsafe_id(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    assert client.get("/api/app/profile/u_000000000000000000000000/avatar").status_code == 404
    assert client.get("/api/app/profile/..%2f..%2fsecret/avatar").status_code == 404


def test_upload_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    anon = TestClient(app)
    resp = anon.post("/api/app/profile/avatar", files={"file": ("a.png", _PNG, "image/png")})
    assert resp.status_code == 401
