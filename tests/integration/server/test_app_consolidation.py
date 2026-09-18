"""Integration tests for the P3 Consolidation resurfacing routes (#1123).

Pacing settings need only a signed-in user + the app data dir. The resurfacing
FEED now also needs a corpus root: since #38 it resolves each due highlight's
graph refs so that it withholds exactly what Your Week and the digest email
withhold. Before that it answered from per-user files alone, and could list
captures the other two surfaces silently dropped.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _signed_in_client(root: Path) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(
        root / "appdata", provider="stub", subject="u", email="u@x.com", name="U"
    )
    signed = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, signed)
    return client


def _only_user(root: Path) -> str:
    """The single user id ``_signed_in_client`` created."""
    return get_or_create_user(
        root / "appdata", provider="stub", subject="u", email="u@x.com", name="U"
    ).user_id


def test_resurfacing_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/resurfacing").status_code in (401, 403)


def test_resurfacing_settings_pause_and_empty_feed(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)

    # Default pacing settings.
    got = client.get("/api/app/resurfacing/settings")
    assert got.status_code == 200
    settings = got.json()
    assert settings["paused"] is False

    # Pause via PUT (echo the current settings back with the flag flipped so the
    # body always satisfies the schema).
    settings["paused"] = True
    put = client.put("/api/app/resurfacing/settings", json=settings)
    assert put.status_code == 200
    assert put.json()["paused"] is True

    # The feed honours the pause and, with no captured highlights, is empty.
    feed = client.get("/api/app/resurfacing")
    assert feed.status_code == 200
    body = feed.json()
    assert body["paused"] is True
    assert body["items"] == []

    # Marking an absent highlight is a 404, NOT the "idempotent 204 no-op" this used to assert.
    #
    # That phrasing dressed up a gap as a feature: the route wrote whatever key it was handed, with
    # no existence or ownership check, so resurfacing.json accumulated an entry per call, for ever
    # (#39). Nothing read those keys back — select_due iterates highlights — so it was unbounded
    # growth rather than a wrong answer, and a 204 made it look deliberate. It stopped being merely
    # untidy when #35 wired the mark to a `?revisit=` query parameter, i.e. to any string a user can
    # type into their address bar.
    assert client.post("/api/app/resurfacing/h_absent/surfaced").status_code == 404


# --- retire / un-retire: the round trip Saved depends on (operator 2026-09-18) -------------------
#
# Saved stays a straight list of every capture; `retired` is one more FIELD on the highlight,
# joined from the resurfacing state on read like `anchor_status`. No second list, no second store.


def _capture(client: TestClient, slug: str = "ep-1") -> str:
    """One captured moment, returning its id."""
    made = client.post(
        "/api/app/highlights",
        json={"episode_slug": slug, "kind": "moment", "start_ms": 1000},
    )
    assert made.status_code == 201, made.text
    return str(made.json()["id"])


def _highlight(client: TestClient, hid: str) -> dict:
    got = client.get("/api/app/highlights")
    assert got.status_code == 200
    return next(h for h in got.json()["items"] if h["id"] == hid)


def test_retire_is_reversible_and_visible_in_saved(tmp_path: Path) -> None:
    """The point of the undo: Saved must SHOW the state, or it cannot offer to reverse it.

    Retiring hides a capture from Revisit, so Revisit is exactly where the undo cannot live.
    """
    client = _signed_in_client(tmp_path)
    hid = _capture(client)

    assert _highlight(client, hid)["retired"] is False

    assert client.post(f"/api/app/resurfacing/{hid}/retire").status_code == 204
    assert _highlight(client, hid)["retired"] is True

    assert client.delete(f"/api/app/resurfacing/{hid}/retire").status_code == 204
    assert _highlight(client, hid)["retired"] is False


def test_unretiring_keeps_the_existing_rung(tmp_path: Path) -> None:
    """Resuming must not restart the ladder at 2 days.

    `set_resurfacing_retired` merges into the record rather than replacing it. If it clobbered,
    a capture reviewed four times would return tomorrow instead of in three months, and the user
    would read that as the app having forgotten.
    """
    client = _signed_in_client(tmp_path)
    hid = _capture(client)
    client.post(f"/api/app/resurfacing/{hid}/surfaced")
    client.post(f"/api/app/resurfacing/{hid}/surfaced")

    client.post(f"/api/app/resurfacing/{hid}/retire")
    client.delete(f"/api/app/resurfacing/{hid}/retire")

    state = app_user_state.get_resurfacing_state(tmp_path / "appdata", _only_user(tmp_path))
    assert state[hid]["count"] == 2, "un-retiring reset the ladder"
    assert "retired" not in state[hid], "retired must be REMOVED, not written False"


def test_unretire_404s_on_an_id_the_caller_does_not_own(tmp_path: Path) -> None:
    """Same ownership rule as retire/surfaced — the id arrives off the wire."""
    client = _signed_in_client(tmp_path)
    assert client.delete("/api/app/resurfacing/h_absent/retire").status_code == 404


def test_unretiring_something_never_retired_is_a_no_op(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)
    hid = _capture(client)
    assert client.delete(f"/api/app/resurfacing/{hid}/retire").status_code == 204
    assert _highlight(client, hid)["retired"] is False


def test_a_corrupt_resurfacing_entry_does_not_break_the_saved_list(tmp_path: Path) -> None:
    """The join in GET /highlights reads a hand-editable file, so it tolerates the wrong shape.

    `select_due` already survives this; the new read path had to as well, or one bad key would
    take down the whole Saved tab rather than one card's badge.
    """
    client = _signed_in_client(tmp_path)
    hid = _capture(client)
    app_user_state._write(
        tmp_path / "appdata", _only_user(tmp_path), "resurfacing", {hid: "corrupt"}
    )

    assert _highlight(client, hid)["retired"] is False
