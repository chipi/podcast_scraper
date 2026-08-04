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

from podcast_scraper.server import app_comms_store, app_digest_personal, app_outbox_store
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
    def fake_refs(_root: Path, slug: str) -> list[dict[str, str]]:
        return [] if slug.endswith("-bare") else list(_REFS)

    monkeypatch.setattr(app_digest_personal, "_graph_refs_for_slug", fake_refs)


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
