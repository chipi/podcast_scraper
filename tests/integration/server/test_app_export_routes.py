"""Route test for the Obsidian export zip (RFC-113, #1472)."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_slugs import episode_slug
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _corpus(root: Path) -> str:
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    rel = "metadata/0001.metadata.json"
    (meta / "0001.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "fa", "title": "Show", "url": "https://p/f.xml"},
                "episode": {
                    "episode_id": "e1",
                    "title": "NVIDIA",
                    "published_date": "2024-01-01T00:00:00",
                    "duration_seconds": 1000,
                },
                "summary": {"title": "S", "bullets": ["a"]},
                "content": {"transcript_file_path": "transcripts/0001.txt"},
            }
        ),
        encoding="utf-8",
    )
    (root / "transcripts" / "0001.txt").write_text("hi", encoding="utf-8")
    (meta / "0001.kg.json").write_text(
        json.dumps({"episode_id": "e1", "nodes": []}), encoding="utf-8"
    )
    return episode_slug("fa", "e1", rel)


def _authed(tmp_path: Path):
    root = tmp_path / "corpus"
    data_dir = tmp_path / "app"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="google", subject="s", email="u@g.com", name="U")
    client = TestClient(app)
    tok = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, tok)
    return client, root, data_dir, user.user_id


def test_export_zip_contains_notes_and_manifest(tmp_path: Path) -> None:
    client, root, data_dir, uid = _authed(tmp_path)
    slug = _corpus(root)
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h_1",
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 1000,
            "quote_text": "a line",
            "created_at": 1,
            "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}],
        },
    )
    resp = client.get("/api/app/export", params={"format": "obsidian"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["x-export-mode"] == "full"
    rev = int(resp.headers["x-export-revision"])
    # A client echoes the vault identity alongside the revision (#41). A bare number cannot
    # identify a snapshot across a server-side state reset, so an epoch-less request is
    # answered with a full export by design.
    ep = resp.headers["x-export-epoch"]
    assert ep

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "manifest.json" in names
    assert "closelistening/Highlights/h_1.md" in names
    assert "closelistening/Topics/topic_ai.md" in names
    assert f"closelistening/Episodes/{slug}.md" in names
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == "obsidian" and manifest["namespace"] == "closelistening"

    # incremental: nothing changed → empty delta, same revision
    r2 = client.get("/api/app/export", params={"format": "obsidian", "since": rev, "epoch": ep})
    assert r2.headers["x-export-mode"] == "incremental"
    assert int(r2.headers["x-export-written"]) == 0
    assert int(r2.headers["x-export-revision"]) == rev

    # add a highlight → incremental returns only the new note
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h_2",
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 2000,
            "quote_text": "second",
            "created_at": 2,
            "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}],
        },
    )
    r3 = client.get("/api/app/export", params={"format": "obsidian", "since": rev, "epoch": ep})
    assert r3.headers["x-export-mode"] == "incremental"
    names3 = set(zipfile.ZipFile(io.BytesIO(r3.content)).namelist())
    assert "closelistening/Highlights/h_2.md" in names3
    assert "closelistening/Highlights/h_1.md" not in names3  # unchanged, not re-sent


def test_bad_format_400(tmp_path: Path) -> None:
    client, root, _, _ = _authed(tmp_path)
    _corpus(root)
    assert client.get("/api/app/export", params={"format": "notion"}).status_code == 400


def test_requires_auth(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _corpus(root)
    app = create_app(root, static_dir=False)
    app.state.app_data_dir = tmp_path / "app"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/export", params={"format": "obsidian"}).status_code in (
        401,
        403,
    )


def test_a_pre_reset_cursor_gets_a_full_export_over_the_wire(tmp_path: Path) -> None:
    """The collision, end to end through the route (#41).

    The server's cursor restarts at 0 whenever its export state is lost or unreadable, then climbs
    back through values a client may still hold. Matching integers used to mean "incremental", so
    the server computed a delta against ITS OWN snapshot rather than against the client's actual
    vault — and because the cursors then advanced in lockstep, that client never asked for a full
    export again.
    """
    client, root, data_dir, uid = _authed(tmp_path)
    slug = _corpus(root)
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h_1",
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 1000,
            "quote_text": "a line",
            "created_at": 1,
            "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}],
        },
    )
    first = client.get("/api/app/export", params={"format": "obsidian"})
    rev, epoch = int(first.headers["x-export-revision"]), first.headers["x-export-epoch"]

    # The server loses its export state; exports resume and the counter climbs back through `rev`.
    (data_dir / "users" / uid / "export_state.json").unlink()
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h_2",
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 2000,
            "created_at": 2,
            "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}],
        },
    )
    rebuilt = client.get("/api/app/export", params={"format": "obsidian"})
    assert int(rebuilt.headers["x-export-revision"]) == rev  # the integers collide
    assert rebuilt.headers["x-export-epoch"] != epoch  # the identities do not

    stale = client.get(
        "/api/app/export", params={"format": "obsidian", "since": rev, "epoch": epoch}
    )
    assert stale.headers["x-export-mode"] == "full"
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(stale.content)).read("manifest.json"))
    assert manifest["replace_namespace"] is True
    assert manifest["epoch"] == rebuilt.headers["x-export-epoch"]


def test_the_same_vault_zips_to_identical_bytes(tmp_path: Path) -> None:
    """`zipfile.writestr` stamps WALL-CLOCK time per entry (#44).

    So two exports of an unchanged vault produced different bytes: no ETag, no content-addressed
    caching, and any test asserting on zip bytes would flake by construction. The CONTENT was
    already deterministic — this makes the container match it.
    """
    client, root, data_dir, uid = _authed(tmp_path)
    slug = _corpus(root)
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h_1",
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 1000,
            "quote_text": "a line",
            "created_at": 1,
            "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}],
        },
    )
    first = client.get("/api/app/export", params={"format": "obsidian"})
    second = client.get("/api/app/export", params={"format": "obsidian"})

    # Assert the MECHANISM, not just equal bytes. Two exports in one test run land in the same
    # second, and zip's DOS timestamps have 2-second granularity — so byte-equality passes with the
    # bug in place almost always, and fails only if the run straddles a boundary. That is a flaky
    # test that proves nothing; sabotage confirmed it (reverting to `writestr(path, ...)` left it
    # green). Every entry must carry the FIXED stamp.
    entries = zipfile.ZipFile(io.BytesIO(first.content)).infolist()
    assert entries, "empty zip"
    stamps = {e.date_time for e in entries}
    assert stamps == {(1980, 1, 1, 0, 0, 0)}, stamps

    assert first.content == second.content, "identical vaults must produce identical zip bytes"
