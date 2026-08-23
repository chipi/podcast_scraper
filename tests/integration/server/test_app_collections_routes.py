"""Route tests for the collections surface ``/api/app/collections`` (#1417)."""

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


def test_create_list_add_detail_delete(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    # a highlight to add
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h1", "episode_slug": "ep", "kind": "span", "created_at": 1}
    )

    assert client.get("/api/app/collections").json()["items"] == []
    cid = client.post("/api/app/collections", json={"name": "AI takes"}).json()["id"]

    added = client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": "h1"})
    assert added.status_code == 200 and added.json()["count"] == 1

    detail = client.get(f"/api/app/collections/{cid}").json()
    assert detail["collection"]["name"] == "AI takes"
    assert [h["id"] for h in detail["highlights"]] == ["h1"]

    # delete the collection
    remaining = client.delete(f"/api/app/collections/{cid}")
    assert remaining.status_code == 200 and remaining.json()["items"] == []


def test_add_item_to_unknown_collection_404(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    resp = client.post("/api/app/collections/col_missing/items", json={"highlight_id": "h1"})
    assert resp.status_code == 404


def _highlight(client: TestClient, slug: str = "ep1") -> str:
    resp = client.post(
        "/api/app/highlights", json={"episode_slug": slug, "kind": "moment", "start_ms": 1000}
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def test_the_count_matches_the_cards_after_a_highlight_is_deleted(tmp_path: Path) -> None:
    """The badge said 2 while 1 card rendered — and both numbers were in the SAME response.

    Deleting a highlight touches only highlights.json, so every collection that held it keeps the
    id. The detail view already dropped ids it could not hydrate, but ``count`` was the raw
    membership length, so CollectionDetail contradicted itself. Counts are now resolved against the
    live highlight ids on every response that carries one.

    Replaces test_detail_drops_deleted_highlight, which named a deleted highlight but used an id
    that never existed — so it never exercised the delete path, and it asserted only the dropping,
    never that the count agreed.
    """
    client, _data_dir, _uid = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "c"}).json()["id"]
    keep, doomed = _highlight(client), _highlight(client, "ep2")
    for hid in (keep, doomed):
        client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": hid})
    assert client.get(f"/api/app/collections/{cid}").json()["collection"]["count"] == 2

    assert client.delete(f"/api/app/highlights/{doomed}").status_code == 200

    detail = client.get(f"/api/app/collections/{cid}").json()
    assert [h["id"] for h in detail["highlights"]] == [keep]
    assert detail["collection"]["count"] == len(detail["highlights"]) == 1
    # The list surface has to agree with the detail surface, not merely with itself.
    listed = client.get("/api/app/collections").json()["items"]
    assert next(c for c in listed if c["id"] == cid)["count"] == 1


def test_filing_a_highlight_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """Membership is an opaque id list, so the store accepts any string.

    An unknown id was therefore stored forever: uncountable (it can never hydrate) and
    unrenderable. 404 is the honest answer — the client asked to file something that is not there.
    """
    client, _data_dir, _uid = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "c"}).json()["id"]
    resp = client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": "ghost"})
    assert resp.status_code == 404, resp.text
    assert client.get(f"/api/app/collections/{cid}").json()["collection"]["count"] == 0


def test_an_unknown_collection_reports_the_collection_not_the_highlight(tmp_path: Path) -> None:
    """Both checks 404; the path resource is checked first so the message stays truthful."""
    client, _data_dir, _uid = _authed(tmp_path)
    hid = _highlight(client)
    resp = client.post("/api/app/collections/col_missing/items", json={"highlight_id": hid})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "collection not found"


def test_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/collections").status_code in (401, 403)


def test_remove_item_route(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    for hid in ("h1", "h2"):
        app_user_state.add_highlight(
            data_dir, uid, {"id": hid, "episode_slug": "ep", "kind": "span", "created_at": 1}
        )
    cid = client.post("/api/app/collections", json={"name": "c"}).json()["id"]
    client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": "h1"})
    client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": "h2"})
    resp = client.request("DELETE", f"/api/app/collections/{cid}/items/h1")
    assert resp.status_code == 200 and resp.json()["count"] == 1
    ids = [h["id"] for h in client.get(f"/api/app/collections/{cid}").json()["highlights"]]
    assert ids == ["h2"]


def test_deleting_a_highlight_does_not_leave_its_notes_to_resurrect(tmp_path: Path) -> None:
    """Through the real routes: the note is gone from the API, not just from the client's memory.

    The client prunes deleted highlights' notes locally, so the user is SHOWN the note disappearing.
    Server-side it survived, and came back on the next full load.
    """
    client, _data_dir, _uid = _authed(tmp_path)
    hid = _highlight(client)
    note = client.post(
        "/api/app/notes", json={"target": "highlight", "target_id": hid, "text": "why this matters"}
    )
    assert note.status_code == 201, note.text
    assert len(client.get("/api/app/notes").json()["items"]) == 1

    assert client.delete(f"/api/app/highlights/{hid}").status_code == 200

    assert client.get("/api/app/notes").json()["items"] == []
    assert "why this matters" not in client.get("/api/app/highlights/export.md").text


def test_the_export_carries_every_note_the_user_wrote(tmp_path: Path) -> None:
    """Highlight notes, episode notes and insight notes — the last two never appeared at all."""
    client, _data_dir, _uid = _authed(tmp_path)
    hid = _highlight(client, "ep1")
    for body in (
        {"target": "highlight", "target_id": hid, "text": "note on the highlight"},
        {"target": "episode", "target_id": "ep1", "text": "note on the episode"},
        {"target": "insight", "target_id": "ins-99", "text": "note on an insight"},
    ):
        assert client.post("/api/app/notes", json=body).status_code == 201

    md = client.get("/api/app/highlights/export.md").text
    assert "note on the highlight" in md
    assert "note on the episode" in md
    assert "note on an insight" in md
