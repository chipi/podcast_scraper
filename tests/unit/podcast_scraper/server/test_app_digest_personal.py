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
    #
    # The empty-slug case mirrors the real `refs_for_slug`, which returns [] before touching the
    # corpus. Without that line the stub answered a blank slug with refs — more generous than
    # production — and the gate-equivalence test below read that as the two gates disagreeing.
    # A stub that is kinder than the real thing manufactures failures as readily as it hides them.
    def fake_refs(_root: Path, slug: str, *, limit: int = 3) -> list[dict[str, str]]:
        if not slug:
            return []
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
    assert item["deep_link"] == "/player/ep-one?t=60&revisit=h1"
    assert item["source"] == "user"


# --- the digest must be able to ADVANCE the ladder it reads from (#35) --------------------------
#
# `revisit` on the link is the whole mechanism: arriving at the player carrying it marks the
# highlight surfaced. Without it the digest was write-only — it selected due items every week and
# nothing it produced could ever mark one reviewed, so an email-only reader received the same five
# items indefinitely. The one advance path in the product was a dismiss button in a tab they may
# never open.


def test_a_revisit_item_carries_what_advances_it(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    item = payload["sections"][0]["items"][0]
    # As DATA for the in-app card (it builds its own route rather than parsing the link)...
    assert item["highlight_id"] == "h1"
    # ...and in the LINK for the email, which has only a URL to work with.
    assert "revisit=h1" in item["deep_link"]


def test_a_highlight_id_needing_escaping_survives_the_link(tmp_path: Path) -> None:
    """Ids are server-generated today; this pins the encoding before that stops being true.

    An unescaped ``&`` or ``=`` in an id would silently truncate the query and mark the WRONG
    highlight — or none — with no error raised anywhere.
    """
    uid = _user(tmp_path)
    _add_highlight(tmp_path, uid, "ep-one", hid="h&1=x", created_at=1000)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    item = payload["sections"][0]["items"][0]
    assert item["highlight_id"] == "h&1=x"
    assert "revisit=h%261%3Dx" in item["deep_link"]
    assert item["deep_link"].count("&") == 1  # the separator only, not the id's own


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
    # now=10**9 is 01:00 UTC; a daily cadence at hour 1 matches the slot.
    app_comms_store.set_comms(
        tmp_path,
        uid,
        digest={"enabled": True, "cadence": "daily", "hour": 1},
        push={"enabled": True},
    )
    _add_push_sub(tmp_path, uid)
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    ids = app_digest_personal.enqueue_due_digests(_ROOT, tmp_path, now=10**9)
    assert any(i.startswith("dgst_") for i in ids)
    assert any(i.startswith("ndg_") for i in ids)


def test_due_batch_skips_users_outside_their_slot(tmp_path: Path) -> None:
    # #1 cadence gate: a user whose chosen hour != now's hour is not enqueued.
    uid = _user(tmp_path)
    app_comms_store.set_comms(
        tmp_path, uid, digest={"enabled": True, "cadence": "daily", "hour": 9}
    )
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    assert (
        app_digest_personal.enqueue_due_digests(_ROOT, tmp_path, now=10**9) == []
    )  # now is hour 1


def test_is_due_slot_weekly(tmp_path: Path) -> None:
    import datetime as _dt

    when = _dt.datetime.fromtimestamp(10**9, _dt.timezone.utc)  # a specific UTC weekday/hour
    comms = {"digest": {"cadence": "weekly", "day_of_week": when.weekday(), "hour": when.hour}}
    assert app_digest_personal._is_due_slot(comms, 10**9) is True
    comms["digest"]["hour"] = (when.hour + 1) % 24
    assert app_digest_personal._is_due_slot(comms, 10**9) is False


# --- pacing pause applies to resurfacing, not just to one tab (found in review) ----------------
#
# RFC-101 §5 calls frequency/pause/dismiss "per-user settings" on spaced resurfacing. Only
# GET /resurfacing passed `paused` to select_due, so a user who paused pacing still had their
# captures resurfaced through Your Week, the digest email and the push nudge — three surfaces
# ignoring the setting the UI presents as switching it off. The separate comms.digest.paused
# consent gate governs whether the EMAIL is sent; this governs whether resurfacing CONTENT exists.


def _pause(data_dir: Path, uid: str, paused: bool) -> None:
    from podcast_scraper.server import app_user_state

    app_user_state.set_resurfacing_settings(data_dir, uid, paused=paused)


def test_paused_pacing_suppresses_the_revisit_section(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)

    assert app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9) is not None

    _pause(tmp_path, uid, True)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    kinds = [s["kind"] for s in (payload or {}).get("sections", [])]
    assert (
        "revisit" not in kinds
    ), f"pacing is paused but the digest still carries a revisit section: {kinds}"


def test_unpausing_restores_the_revisit_section(tmp_path: Path) -> None:
    uid = _user(tmp_path)
    _add_highlight(tmp_path, uid, "ep-one", hid="h1", created_at=1000)
    _pause(tmp_path, uid, True)
    assert app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9) is None

    _pause(tmp_path, uid, False)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    assert [s["kind"] for s in payload["sections"]] == ["revisit"]


def test_paused_pacing_also_suppresses_auto_picks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-picks ARE resurfacing — GI editor's-picks standing in for captures the user lacks."""
    from podcast_scraper.server import app_auto_picks

    monkeypatch.setattr(
        app_auto_picks,
        "auto_pick_items",
        lambda *a, **k: [{"source": "auto", "graph_refs": _REFS, "deep_link": "/player/x?t=0"}],
    )
    uid = _user(tmp_path)
    assert app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9) is not None

    _pause(tmp_path, uid, True)
    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    kinds = [s["kind"] for s in (payload or {}).get("sections", [])]
    assert "revisit" not in kinds, "paused pacing still topped the digest up with auto-picks"


def test_push_nudge_selects_the_revisit_section_by_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sections[0] is not necessarily revisit — it is only appended when non-empty.

    A push-enabled user with no due captures but a non-empty new_in_follows used to get a
    "resurface-nudge.v1" whose highlight_count counted NEW EPISODES and whose lead was not a
    highlight.
    """
    from podcast_scraper.server import app_digest_sections

    monkeypatch.setattr(
        app_digest_sections,
        "new_in_follows_items",
        lambda *a, **k: [{"source": "follow", "graph_refs": _REFS, "deep_link": "/player/y?t=0"}],
    )
    monkeypatch.setattr(app_digest_sections, "trending_items", lambda *a, **k: [])
    uid = _user(tmp_path)  # no highlights at all → no revisit section
    # Push must be fully enabled, or this asserts nothing: enqueue_push_for_user returns [] on
    # consent long before it reaches the section selection.
    app_comms_store.set_comms(tmp_path, uid, push={"enabled": True})
    _add_push_sub(tmp_path, uid)

    payload = app_digest_personal.assemble_digest_payload(_ROOT, tmp_path, uid, now=10**9)
    assert payload is not None
    assert payload["sections"][0]["kind"] == "new_in_follows"

    # With no revisit content there is nothing to nudge about, so no envelope is enqueued —
    # rather than a "resurface-nudge" built from follow items.
    assert app_digest_personal.enqueue_push_for_user(_ROOT, tmp_path, uid, now=10**9) == []


def test_the_two_gates_agree_for_every_shape_of_highlight(tmp_path: Path) -> None:
    """`carries_the_graph` and `_digest_item`'s drop must be the SAME condition (#38).

    The assembler cannot call the predicate — it needs the refs themselves — so the two express one
    rule in two places. That is precisely the arrangement that let the three revisit surfaces
    disagree in the first place, so it gets pinned rather than trusted.
    """
    from podcast_scraper.server import app_graph_refs

    shapes = [
        {"id": "h1", "episode_slug": "ep-one", "created_at": 1},  # resolves via episode KG
        {"id": "h2", "episode_slug": "ep-bare", "created_at": 1},  # stubbed to no refs
        {"id": "h3", "episode_slug": "", "created_at": 1},  # no slug at all
        {"id": "h4", "episode_slug": "ep-one", "graph_refs": _REFS, "created_at": 1},  # stored refs
        {"id": "h5", "episode_slug": "ep-bare", "graph_refs": _REFS, "created_at": 1},  # stored win
        {
            "id": "h6",
            "episode_slug": "ep-one",
            "graph_refs": [],
            "created_at": 1,
        },  # empty → resolve
        {"id": "h7", "episode_slug": "ep-bare", "graph_refs": [{"no": "id"}], "created_at": 1},
    ]
    for h in shapes:
        predicate = app_graph_refs.carries_the_graph(_ROOT, h)
        assembled = app_digest_personal._digest_item(_ROOT, h) is not None
        assert predicate == assembled, (
            f"{h['id']}: carries_the_graph={predicate} but the assembler "
            f"{'kept' if assembled else 'dropped'} it — the two gates have drifted"
        )
