"""Unit tests for the personal digest assembler (#1415, app_digest_personal).

Isolates the assembler logic (due selection, graph-carrying item shaping, zero-content, envelope
building) from a real corpus by stubbing the KG resolution. Also validates that a built envelope
satisfies the committed contract schema — proving the assembler produces seam-valid output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from podcast_scraper.server import (
    app_comms_store,
    app_digest_personal,
    app_graph_refs,
    app_outbox_store,
)
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = pytest.mark.unit

_ROOT = Path("/unused")  # KG resolution is stubbed; the corpus root is never read.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCHEMA = json.loads(
    (_REPO_ROOT / "docs" / "api" / "delivery-envelope.schema.json").read_text(encoding="utf-8")
)

_REFS = [{"id": "person:jane-doe", "kind": "person", "label": "Jane Doe"}]


@pytest.fixture(autouse=True)
def _stub_kg(monkeypatch: pytest.MonkeyPatch) -> None:
    # Episodes carry graph refs; a slug ending in "-bare" has none (to test the flat-clip drop).
    def fake_refs(_root: Path, slug: str, *, limit: int = 3) -> list[dict[str, str]]:
        return [] if slug.endswith("-bare") else list(_REFS)

    monkeypatch.setattr(app_graph_refs, "refs_for_slug", fake_refs)


def _user(data_dir: Path, *, provider: str = "google") -> str:
    u = get_or_create_user(data_dir, provider=provider, subject="s1", email="u@gmail.com", name="U")
    return u.user_id


def _add_highlight(data_dir: Path, uid: str, slug: str, *, hid: str, created_at: int) -> None:
    from podcast_scraper.server import app_user_state

    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": hid,
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 60_000,
            "quote_text": "a memorable line",
            "created_at": created_at,
        },
    )


def test_zero_content_returns_none(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    assert app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9) is None


def test_payload_carries_the_graph(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    _add_highlight(
        tmp_path, uid, "ep-one", hid="h1", created_at=1000
    )  # nonzero + long overdue → due
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    item = payload["sections"][0]["items"][0]
    assert item["graph_refs"] == _REFS
    assert item["deep_link"] == "/player/ep-one?t=60"
    assert item["source"] == "user"


def test_flat_clip_is_dropped(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    _add_highlight(tmp_path, uid, "ep-bare", hid="h1", created_at=1000)  # no KG refs → dropped
    assert app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9) is None


def test_passive_user_gets_auto_seeded_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #1416: a user who captured nothing still gets a non-empty digest from editor's-picks.
    uid = _user(tmp_path)
    auto_item = {
        "episode_slug": "ep-heard",
        "graph_refs": _REFS,
        "deep_link": "/player/ep-heard?t=60",
        "t_ms": 60_000,
        "quote": "auto pick",
        "source": "auto",
    }
    monkeypatch.setattr(
        app_digest_personal.app_auto_picks,
        "auto_pick_items",
        lambda root, dd, uid, *, exclude_slugs, limit: [auto_item],
    )
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    items = payload["sections"][0]["items"]
    assert [i["source"] for i in items] == ["auto"]


def test_built_envelope_matches_contract_schema(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    comms = app_comms_store.set_comms(tmp_path, uid, digest={"enabled": True})
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    from podcast_scraper.server.app_user_store import get_user

    user = get_user(tmp_path, uid)
    assert user is not None
    envelope = app_digest_personal.build_email_envelope(user, comms, payload, now=10**9)
    errors = sorted(Draft202012Validator(_SCHEMA).iter_errors(envelope), key=str)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    assert envelope["expires_at"] > envelope["created_at"]


def test_enqueue_for_user_gated_on_consent(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    # digest off → nothing enqueued
    assert app_digest_personal.enqueue_for_user(_ROOT, tmp_path, uid, now=10**9) is None
    app_comms_store.set_comms(tmp_path, uid, digest={"enabled": True})
    eid = app_digest_personal.enqueue_for_user(_ROOT, tmp_path, uid, now=10**9)
    assert eid is not None and eid.startswith("dgst_")
    # the enqueued envelope is now visible to the worker view
    pending = app_outbox_store.list_pending(tmp_path, channel="email", now=10**9)
    assert [e["id"] for e in pending] == [eid]


def test_enqueue_skips_unverified_email(tmp_path: Path) -> None:
    uid = _user(tmp_path, provider="stub")  # not google → email not verified
    app_comms_store.set_comms(tmp_path, uid, digest={"enabled": True})
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    assert app_digest_personal.enqueue_for_user(_ROOT, tmp_path, uid, now=10**9) is None


def test_enqueue_is_period_idempotent(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    app_comms_store.set_comms(tmp_path, uid, digest={"enabled": True})
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    a = app_digest_personal.enqueue_for_user(_ROOT, tmp_path, uid, now=10**9)
    b = app_digest_personal.enqueue_for_user(_ROOT, tmp_path, uid, now=10**9)
    assert a == b  # same period → same id
    assert len(app_outbox_store.list_pending(tmp_path, channel="email", now=10**9)) == 1


def _add_push_sub(tmp_path: Path, uid: str, endpoint: str = "https://push.invalid/x") -> None:
    from podcast_scraper.server import app_push_store

    app_push_store.add_subscription(tmp_path, uid, {"endpoint": endpoint, "keys": {"auth": "a"}})


def test_push_envelope_matches_contract_schema(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    comms = app_comms_store.set_comms(tmp_path, uid, push={"enabled": True})
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    from podcast_scraper.server.app_user_store import get_user

    user = get_user(tmp_path, uid)
    assert user is not None
    nudge = app_digest_personal._nudge_payload(payload["sections"][0]["items"])
    env = app_digest_personal.build_push_envelope(
        user, comms, {"endpoint": "https://push.invalid/x"}, nudge, now=10**9
    )
    errors = sorted(Draft202012Validator(_SCHEMA).iter_errors(env), key=str)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


def test_enqueue_push_gated_on_subscription(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    app_comms_store.set_comms(tmp_path, uid, push={"enabled": True})
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    # push enabled but no subscription → nothing
    assert app_digest_personal.enqueue_push_for_user(_ROOT, tmp_path, uid, now=10**9) == []
    _add_push_sub(tmp_path, uid)
    ids = app_digest_personal.enqueue_push_for_user(_ROOT, tmp_path, uid, now=10**9)
    assert len(ids) == 1 and ids[0].startswith("ndg_")
    pending = app_outbox_store.list_pending(tmp_path, channel="push", now=10**9)
    assert [e["id"] for e in pending] == ids


def test_due_batch_enqueues_both_channels(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    app_comms_store.set_comms(tmp_path, uid, digest={"enabled": True}, push={"enabled": True})
    _add_push_sub(tmp_path, uid)
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    ids = app_digest_personal.enqueue_due_digests(_ROOT, tmp_path, now=10**9)
    assert any(i.startswith("dgst_") for i in ids)
    assert any(i.startswith("ndg_") for i in ids)
