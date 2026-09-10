"""Unit tests for the in-app notification inbox store (wave-I)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_comms_store, app_notifications_store

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"


def test_empty_inbox_defaults(tmp_path: Path) -> None:
    assert app_notifications_store.list_notifications(tmp_path, _UID) == []
    assert app_notifications_store.unread_count(tmp_path, _UID) == 0


def test_add_and_list_newest_first(tmp_path: Path) -> None:
    app_notifications_store.add_notification(tmp_path, _UID, ntype="product", title="Old", now=1000)
    app_notifications_store.add_notification(
        tmp_path, _UID, ntype="new_episodes", title="New", deep_link="/player/x", now=2000
    )
    items = app_notifications_store.list_notifications(tmp_path, _UID)
    assert [i["title"] for i in items] == ["New", "Old"]  # newest-first
    assert items[0]["deep_link"] == "/player/x"
    assert app_notifications_store.unread_count(tmp_path, _UID) == 2


def test_mark_read_one(tmp_path: Path) -> None:
    rec = app_notifications_store.add_notification(tmp_path, _UID, ntype="product", title="A")
    assert rec is not None
    assert app_notifications_store.mark_read(tmp_path, _UID, rec["id"]) is True
    assert app_notifications_store.unread_count(tmp_path, _UID) == 0
    # idempotent + unknown id is False
    assert app_notifications_store.mark_read(tmp_path, _UID, rec["id"]) is True
    assert app_notifications_store.mark_read(tmp_path, _UID, "nope") is False


def test_mark_all_read(tmp_path: Path) -> None:
    app_notifications_store.add_notification(tmp_path, _UID, ntype="product", title="A")
    app_notifications_store.add_notification(tmp_path, _UID, ntype="product", title="B")
    assert app_notifications_store.mark_all_read(tmp_path, _UID) == 2
    assert app_notifications_store.unread_count(tmp_path, _UID) == 0
    assert app_notifications_store.mark_all_read(tmp_path, _UID) == 0  # nothing left


def test_bounded_to_max(tmp_path: Path) -> None:
    for i in range(app_notifications_store._MAX_RECORDS + 25):
        app_notifications_store.add_notification(
            tmp_path, _UID, ntype="product", title=f"n{i}", now=1000 + i
        )
    items = app_notifications_store.list_notifications(tmp_path, _UID, limit=10_000)
    assert len(items) == app_notifications_store._MAX_RECORDS
    # the newest survived, the oldest dropped
    assert items[0]["title"] == f"n{app_notifications_store._MAX_RECORDS + 24}"


def test_dedupe_key_prevents_double_post(tmp_path: Path) -> None:
    first = app_notifications_store.add_notification(
        tmp_path, _UID, ntype="new_episodes", title="Ep 1", dedupe_key="feed:p1:seq:5"
    )
    second = app_notifications_store.add_notification(
        tmp_path, _UID, ntype="new_episodes", title="Ep 1 again", dedupe_key="feed:p1:seq:5"
    )
    assert first is not None and second is None
    assert len(app_notifications_store.list_notifications(tmp_path, _UID)) == 1


def test_emit_gated_on_in_app_consent(tmp_path: Path) -> None:
    # in_app defaults ON → emit writes.
    rec = app_notifications_store.emit(tmp_path, _UID, ntype="new_episodes", title="Fresh")
    assert rec is not None
    assert app_notifications_store.unread_count(tmp_path, _UID) == 1
    # turn in_app OFF for the type → emit is a no-op.
    app_comms_store.set_comms(tmp_path, _UID, types={"new_episodes": {"in_app": False}})
    assert app_notifications_store.emit(tmp_path, _UID, ntype="new_episodes", title="Muted") is None
    assert app_notifications_store.unread_count(tmp_path, _UID) == 1  # unchanged


def test_dedupe_key_suppresses_re_emit_even_after_read(tmp_path: Path) -> None:
    # A read record still blocks re-emit of the same key — a user who read and moved on must not be
    # re-alerted for the same episode (the dedupe scans ALL records, not just unread).
    first = app_notifications_store.add_notification(
        tmp_path, _UID, ntype="new_episodes", title="Ep 1", dedupe_key="newep:s1"
    )
    assert first is not None
    app_notifications_store.mark_read(tmp_path, _UID, first["id"])
    second = app_notifications_store.add_notification(
        tmp_path, _UID, ntype="new_episodes", title="Ep 1 again", dedupe_key="newep:s1"
    )
    assert second is None
    assert len(app_notifications_store.list_notifications(tmp_path, _UID)) == 1


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert app_notifications_store.list_notifications(tmp_path, "../evil") == []
    assert app_notifications_store.unread_count(tmp_path, "../evil") == 0
    with pytest.raises(ValueError):
        app_notifications_store.add_notification(tmp_path, "../evil", ntype="product", title="x")
