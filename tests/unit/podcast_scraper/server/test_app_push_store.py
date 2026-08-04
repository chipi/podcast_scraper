"""Unit tests for the per-user push-subscription store (#1415, app_push_store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_push_store

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"


def _sub(endpoint: str = "https://push.example.invalid/a") -> dict:
    return {"endpoint": endpoint, "keys": {"p256dh": "pub", "auth": "authv"}}


def test_empty_when_unset(tmp_path: Path) -> None:
    assert app_push_store.list_subscriptions(tmp_path, _UID) == []


def test_add_and_list(tmp_path: Path) -> None:
    app_push_store.add_subscription(tmp_path, _UID, _sub())
    subs = app_push_store.list_subscriptions(tmp_path, _UID)
    assert [s["endpoint"] for s in subs] == ["https://push.example.invalid/a"]


def test_add_dedupes_on_endpoint(tmp_path: Path) -> None:
    app_push_store.add_subscription(tmp_path, _UID, _sub())
    app_push_store.add_subscription(tmp_path, _UID, _sub())  # same endpoint → replace
    assert len(app_push_store.list_subscriptions(tmp_path, _UID)) == 1
    app_push_store.add_subscription(tmp_path, _UID, _sub("https://push.example.invalid/b"))
    assert len(app_push_store.list_subscriptions(tmp_path, _UID)) == 2


def test_add_requires_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        app_push_store.add_subscription(tmp_path, _UID, {"keys": {}})


def test_remove(tmp_path: Path) -> None:
    app_push_store.add_subscription(tmp_path, _UID, _sub("https://push.example.invalid/a"))
    app_push_store.add_subscription(tmp_path, _UID, _sub("https://push.example.invalid/b"))
    remaining = app_push_store.remove_subscription(tmp_path, _UID, "https://push.example.invalid/a")
    assert [s["endpoint"] for s in remaining] == ["https://push.example.invalid/b"]


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert app_push_store.list_subscriptions(tmp_path, "../evil") == []
    with pytest.raises(ValueError):
        app_push_store.add_subscription(tmp_path, "../evil", _sub())
