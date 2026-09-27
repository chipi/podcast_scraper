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

    added = client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "highlight", "ref": "h1"}
    )
    assert added.status_code == 200 and added.json()["count"] == 1

    detail = client.get(f"/api/app/collections/{cid}").json()
    assert detail["collection"]["name"] == "AI takes"
    assert [i["ref"] for i in detail["items"]] == ["h1"]

    # delete the collection
    remaining = client.delete(f"/api/app/collections/{cid}")
    assert remaining.status_code == 200 and remaining.json()["items"] == []


def test_offline_create_then_pin_replays_without_losing_the_item(tmp_path: Path) -> None:
    # #2004 BLOCKER: an offline "create collection + pin an episode" queues a create with a
    # client-minted id and a pin targeting that id. On replay the create must be idempotent AND keep
    # the client id, or the pin 404s and the item is lost. This is the full replay chain.
    client, data_dir, uid = _authed(tmp_path)
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h1", "episode_slug": "ep", "kind": "span", "created_at": 1}
    )
    body = {"name": "Reading list", "client_id": "col_offline1"}

    first = client.post("/api/app/collections", json=body)
    assert first.status_code == 201 and first.json()["id"] == "col_offline1"

    # The response was "lost" and the outbox replays the create: same id, 200, no duplicate.
    replay = client.post("/api/app/collections", json=body)
    assert replay.status_code == 200 and replay.json()["id"] == "col_offline1"
    assert len(client.get("/api/app/collections").json()["items"]) == 1

    # The pin the client queued against the client id now resolves against a real collection.
    pinned = client.post(
        "/api/app/collections/col_offline1/items", json={"kind": "highlight", "ref": "h1"}
    )
    assert pinned.status_code == 200 and pinned.json()["count"] == 1


def test_cover_is_derived_from_the_first_episode_member_and_recomputed(
    tmp_path: Path, monkeypatch
) -> None:
    # CO.6: the cover is the first episode/highlight member's artwork, cached on the row and
    # recomputed on membership change. Patch the artwork resolver so the test needs no image fixture.
    from podcast_scraper.server.routes import app_collections

    monkeypatch.setattr(
        app_collections, "_episode_artwork", lambda root, slug: f"https://art/{slug}.jpg"
    )
    client, _, _ = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "Watch"}).json()["id"]

    # A topic carries no artwork → still no cover.
    client.post(f"/api/app/collections/{cid}/items", json={"kind": "topic", "ref": "topic:ai"})
    assert client.get("/api/app/collections").json()["items"][0]["cover_url"] is None

    # Adding an episode derives the cover from it.
    after_add = client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "episode", "ref": "ep-1"}
    ).json()
    assert after_add["cover_url"] == "https://art/ep-1.jpg"

    # Removing the episode recomputes back to no cover (the topic has none).
    removed = client.delete(f"/api/app/collections/{cid}/items?kind=episode&ref=ep-1").json()
    assert removed["cover_url"] is None


def test_list_backfills_a_missing_cover_for_a_board_that_has_items(
    tmp_path: Path, monkeypatch
) -> None:
    """A board with members but no stored cover gets one on first LIST, and it sticks.

    `cover_url` is only written by `_recompute_cover` on a membership change, so a board populated
    before covers existed — or one whose recompute lost a race / ran while the corpus root was
    briefly unavailable — kept `None` forever with nothing to heal it. The operator's Boards tab
    showed the empty placeholder on boards that clearly had episodes in them (2026-09-16).
    """
    from podcast_scraper.server import app_collections_store
    from podcast_scraper.server.routes import app_collections

    monkeypatch.setattr(
        app_collections, "_episode_artwork", lambda root, slug: f"https://art/{slug}.jpg"
    )
    client, data_dir, user_id = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "AI"}).json()["id"]
    client.post(f"/api/app/collections/{cid}/items", json={"kind": "episode", "ref": "ep-9"})

    # Simulate the pre-feature state: members present, cover never derived.
    app_collections_store.set_cover(data_dir, user_id, cid, None)
    assert client.get("/api/app/collections").json()["items"][0]["cover_url"] is not None

    # And it was PERSISTED, not just computed for that one response — the list stays a cheap read.
    rows = app_collections_store.list_collections(data_dir, user_id)
    assert rows[0]["cover_url"] == "https://art/ep-9.jpg"


def test_list_does_not_backfill_an_empty_board(tmp_path: Path, monkeypatch) -> None:
    """An empty board must stay coverless — and must not pay for a derivation attempt."""
    from podcast_scraper.server.routes import app_collections

    calls: list[str] = []
    monkeypatch.setattr(
        app_collections,
        "_recompute_cover",
        lambda request, data_dir, user_id, cid: calls.append(cid),
    )
    client, _, _ = _authed(tmp_path)
    client.post("/api/app/collections", json={"name": "Empty"})
    assert client.get("/api/app/collections").json()["items"][0]["cover_url"] is None
    assert calls == []


def test_cover_derives_from_a_show_member_via_its_feed_image(tmp_path: Path, monkeypatch) -> None:
    # A shows-only board fell through to the placeholder even though shows carry a feed image
    # (operator 2026-09-13). The cover now resolves a show member via its feed artwork.
    from podcast_scraper.server.routes import app_collections

    monkeypatch.setattr(
        app_collections, "_show_artwork_map", lambda root: {"feed-9": "https://art/feed-9.jpg"}
    )
    client, _, _ = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "Tech"}).json()["id"]
    after = client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "show", "ref": "feed-9"}
    ).json()
    assert after["cover_url"] == "https://art/feed-9.jpg"


def test_show_artwork_map_resolves_feed_images_and_skips_rows_without_a_feed_id(
    tmp_path: Path, monkeypatch
) -> None:
    # Exercises _show_artwork_map's OWN logic — the layer the cover test above monkeypatches over.
    # Faking one level down (aggregate_feeds) pins the real artwork_url call, the "thumb" size, the
    # local-art -> None fallback, and the feed_id filter.
    from podcast_scraper.server.routes import app_collections

    fake_feeds = [
        {"feed_id": "feed-9", "image_local_relpath": "corpus-art/feed-9/cover.jpg"},
        # No local art but a remote image_url -> falls back to the remote URL (matches the show tile).
        {"feed_id": "feed-remote", "image_url": "https://cdn/remote.jpg"},
        {"feed_id": "feed-noart"},  # neither -> None
        {"image_local_relpath": "corpus-art/orphan.jpg"},  # no feed_id -> dropped entirely
    ]
    monkeypatch.setattr(app_collections, "cached_catalog", lambda root: object())
    monkeypatch.setattr(app_collections, "aggregate_feeds", lambda catalog: fake_feeds)

    result = app_collections._show_artwork_map(tmp_path)

    assert result == {
        "feed-9": "/api/app/artwork?ref=corpus-art%2Ffeed-9%2Fcover.jpg&size=thumb",
        "feed-remote": "https://cdn/remote.jpg",
        "feed-noart": None,
    }


def test_cover_is_none_when_a_show_member_has_no_resolvable_artwork(
    tmp_path: Path, monkeypatch
) -> None:
    # Covers the show-branch FALLTHROUGH in _derive_cover (the codecov/patch gap from #2063):
    # a show whose ref is NOT in the artwork map (art falsy) must `continue` past it, not crash —
    # the cover stays None when nothing else resolves.
    from podcast_scraper.server.routes import app_collections

    monkeypatch.setattr(app_collections, "_show_artwork_map", lambda root: {})
    client, _, _ = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "No art"}).json()["id"]
    after = client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "show", "ref": "feed-missing"}
    ).json()
    assert after["cover_url"] is None


def test_cover_recomputed_when_its_source_highlight_is_deleted(tmp_path: Path, monkeypatch) -> None:
    # advisor M5 (server half): deleting a highlight that a collection's cover derived from must
    # refresh that cover, not leave it pointing at the gone member's episode.
    from podcast_scraper.server.routes import app_collections

    monkeypatch.setattr(
        app_collections, "_episode_artwork", lambda root, slug: f"https://art/{slug}.jpg"
    )
    client, data_dir, uid = _authed(tmp_path)
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h1", "episode_slug": "ep", "kind": "span", "created_at": 1}
    )
    cid = client.post("/api/app/collections", json={"name": "C"}).json()["id"]
    added = client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "highlight", "ref": "h1"}
    ).json()
    assert added["cover_url"] == "https://art/ep.jpg"  # derived from h1's episode

    client.delete("/api/app/highlights/h1")
    # The cover recomputed — h1 is gone, no other member carries artwork → cleared.
    assert client.get("/api/app/collections").json()["items"][0]["cover_url"] is None


def test_add_item_to_unknown_collection_404(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    resp = client.post(
        "/api/app/collections/col_missing/items", json={"kind": "highlight", "ref": "h1"}
    )
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
        client.post(f"/api/app/collections/{cid}/items", json={"kind": "highlight", "ref": hid})
    assert client.get(f"/api/app/collections/{cid}").json()["collection"]["count"] == 2

    assert client.delete(f"/api/app/highlights/{doomed}").status_code == 200

    detail = client.get(f"/api/app/collections/{cid}").json()
    assert [i["ref"] for i in detail["items"]] == [keep]
    assert detail["collection"]["count"] == len(detail["items"]) == 1
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
    resp = client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "highlight", "ref": "ghost"}
    )
    assert resp.status_code == 404, resp.text
    assert client.get(f"/api/app/collections/{cid}").json()["collection"]["count"] == 0


def test_an_unknown_collection_reports_the_collection_not_the_highlight(tmp_path: Path) -> None:
    """Both checks 404; the path resource is checked first so the message stays truthful."""
    client, _data_dir, _uid = _authed(tmp_path)
    hid = _highlight(client)
    resp = client.post(
        "/api/app/collections/col_missing/items", json={"kind": "highlight", "ref": hid}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "collection not found"


def test_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/collections").status_code in (401, 403)


def test_mixed_kinds_resolve_with_deep_links(tmp_path: Path) -> None:
    """RFC-119: a collection holds mixed typed items; the detail resolves deep-links per kind."""
    client, _data_dir, _uid = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "research"}).json()["id"]
    client.post(f"/api/app/collections/{cid}/items", json={"kind": "episode", "ref": "ep-x"})
    client.post(
        f"/api/app/collections/{cid}/items", json={"kind": "topic", "ref": "topic:ai-safety"}
    )
    client.post(
        f"/api/app/collections/{cid}/items",
        json={"kind": "search", "ref": "sleep science", "scope": "mine"},
    )
    client.post(
        f"/api/app/collections/{cid}/items",
        json={"kind": "link", "ref": "https://example.com/post", "title": "A post"},
    )
    items = client.get(f"/api/app/collections/{cid}").json()["items"]
    by_kind = {i["kind"]: i for i in items}
    assert by_kind["episode"]["deep_link"] == "/episode/ep-x"
    assert (
        by_kind["topic"]["title"] == "ai safety"
        and by_kind["topic"]["deep_link"] == "/topic/topic:ai-safety"
    )
    assert (
        by_kind["search"]["scope"] == "mine"
        and "q=sleep%20science" in by_kind["search"]["deep_link"]
    )
    assert (
        by_kind["link"]["title"] == "A post"
        and by_kind["link"]["deep_link"] == "https://example.com/post"
    )
    assert client.get("/api/app/collections").json()["items"][0]["count"] == 4


def test_remove_item_route(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    for hid in ("h1", "h2"):
        app_user_state.add_highlight(
            data_dir, uid, {"id": hid, "episode_slug": "ep", "kind": "span", "created_at": 1}
        )
    cid = client.post("/api/app/collections", json={"name": "c"}).json()["id"]
    client.post(f"/api/app/collections/{cid}/items", json={"kind": "highlight", "ref": "h1"})
    client.post(f"/api/app/collections/{cid}/items", json={"kind": "highlight", "ref": "h2"})
    resp = client.delete(
        f"/api/app/collections/{cid}/items", params={"kind": "highlight", "ref": "h1"}
    )
    assert resp.status_code == 200 and resp.json()["count"] == 1
    ids = [i["ref"] for i in client.get(f"/api/app/collections/{cid}").json()["items"]]
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


class TestWhichBoardsAlreadyHoldThisItem:
    """``GET /collections/containing`` (operator 2026-09-19).

    The add-to-collection picker listed every board identically, so the only way to learn an item
    was already on one was to add it again and watch nothing happen — the add is idempotent, so
    that tap is silent.

    Its OWN resource, not a query pair on ``GET /collections``. As a field, ``contains`` made a
    ``Collection`` mean different things depending on how it was fetched: set on a list read,
    absent on every mutation response. That needed a nullable tri-state and a written doctrine to
    stay coherent — context leaking into an entity schema.
    """

    def test_reports_only_the_boards_that_hold_it(self, tmp_path: Path) -> None:
        client, data_dir, uid = _authed(tmp_path)
        app_user_state.add_highlight(
            data_dir, uid, {"id": "h1", "episode_slug": "ep", "kind": "span", "created_at": 1}
        )
        holding = client.post("/api/app/collections", json={"name": "Holding"}).json()["id"]
        client.post("/api/app/collections", json={"name": "Empty"})
        client.post(
            f"/api/app/collections/{holding}/items", json={"kind": "highlight", "ref": "h1"}
        )

        body = client.get(
            "/api/app/collections/containing", params={"kind": "highlight", "ref": "h1"}
        ).json()
        assert body == {"ids": [holding], "checked": True}

    def test_the_same_ref_under_a_DIFFERENT_kind_is_not_a_match(self, tmp_path: Path) -> None:
        # Membership is keyed on (kind, ref), and ids collide across kinds — an episode slug and a
        # person id can be the same string. Matching on ref alone would mark the wrong board.
        client, _data_dir, _uid = _authed(tmp_path)
        cid = client.post("/api/app/collections", json={"name": "Board"}).json()["id"]
        client.post(f"/api/app/collections/{cid}/items", json={"kind": "episode", "ref": "ep1"})

        body = client.get(
            "/api/app/collections/containing", params={"kind": "person", "ref": "ep1"}
        ).json()
        assert body == {"ids": [], "checked": True}

    def test_an_unreadable_file_is_NOT_CHECKED_rather_than_in_nothing(self, tmp_path: Path) -> None:
        """The distinction the client leans on.

        ``checked: false`` with an empty ``ids`` means "we could not look". Rendering that as "it
        is in none of your boards" gives the user the one answer they act on by saving the item a
        second time — so it must never be inferred from a failed read.
        """
        client, data_dir, uid = _authed(tmp_path)
        client.post("/api/app/collections", json={"name": "Board"})
        (data_dir / "users" / uid / "collections.json").write_text("{not json", encoding="utf-8")

        body = client.get(
            "/api/app/collections/containing", params={"kind": "episode", "ref": "ep1"}
        ).json()
        assert body == {"ids": [], "checked": False}

    def test_a_long_url_ref_is_still_answerable(self, tmp_path: Path) -> None:
        # `kind=link` stores a URL, and the ADD endpoint caps `ref` at nothing at all. A bound here
        # tighter than what can be stored would make such an item permanently unqueryable: 422, and
        # the picker silently shows nothing marked.
        client, _data_dir, _uid = _authed(tmp_path)
        cid = client.post("/api/app/collections", json={"name": "Links"}).json()["id"]
        url = "https://example.test/" + ("a" * 900)
        assert (
            client.post(f"/api/app/collections/{cid}/items", json={"kind": "link", "ref": url})
        ).status_code == 200

        body = client.get(
            "/api/app/collections/containing", params={"kind": "link", "ref": url}
        ).json()
        assert body == {"ids": [cid], "checked": True}

    def test_the_list_endpoint_carries_no_membership_at_all(self, tmp_path: Path) -> None:
        # The point of the split: a Collection means one thing however it was fetched.
        client, _data_dir, _uid = _authed(tmp_path)
        created = client.post("/api/app/collections", json={"name": "Board"}).json()
        listed = client.get("/api/app/collections").json()["items"][0]
        assert "contains" not in created
        assert "contains" not in listed

    def test_containing_is_not_swallowed_by_the_collection_id_route(self, tmp_path: Path) -> None:
        """Route ORDER, pinned.

        ``/collections/{collection_id}`` would happily match ``containing`` as an id and 404. It
        only works because the literal path is declared first, which a reorder silently undoes.
        """
        client, _data_dir, _uid = _authed(tmp_path)
        resp = client.get(
            "/api/app/collections/containing", params={"kind": "episode", "ref": "ep1"}
        )
        assert resp.status_code == 200, "'containing' was matched as a collection id"
        assert "ids" in resp.json()

    def test_one_read_regardless_of_how_many_boards(self, tmp_path: Path, monkeypatch) -> None:
        """The cap is 200 boards; asking per board re-read the same file once per board.

        Pinned as a COUNT rather than a timing, so it cannot regress quietly into an N-read loop
        again — which is what it was when it first shipped.
        """
        from podcast_scraper.server import app_collections_store as store

        client, data_dir, uid = _authed(tmp_path)
        for i in range(8):
            client.post("/api/app/collections", json={"name": f"Board {i}"})

        reads = {"n": 0}
        real_read = store._read

        def counting_read(*a: object, **k: object) -> dict:
            reads["n"] += 1
            return real_read(*a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(store, "_read", counting_read)
        client.get("/api/app/collections/containing", params={"kind": "episode", "ref": "ep1"})
        assert reads["n"] == 1, f"{reads['n']} reads for 8 boards — the per-row loop is back"
