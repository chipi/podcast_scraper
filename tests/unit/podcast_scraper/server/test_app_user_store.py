"""Unit tests for the file-based per-user identity store (#1063)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server.app_user_store import (
    delete_user,
    get_or_create_user,
    get_user,
    list_users,
    set_disabled,
    user_id_for,
)


def test_user_id_is_deterministic_and_prefixed() -> None:
    assert user_id_for("google", "sub1") == user_id_for("google", "sub1")
    assert user_id_for("google", "sub1") != user_id_for("google", "sub2")
    assert user_id_for("google", "sub1") != user_id_for("other", "sub1")
    assert user_id_for("google", "sub1").startswith("u_")


def test_get_or_create_is_idempotent(tmp_path: Path) -> None:
    u1 = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    # second call with different email returns the SAME user, unchanged
    u2 = get_or_create_user(tmp_path, provider="google", subject="s1", email="b@x.com", name="B")
    assert u1.user_id == u2.user_id
    assert u2.email == "a@x.com"
    loaded = get_user(tmp_path, u1.user_id)
    assert loaded is not None and loaded.email == "a@x.com" and loaded.name == "A"


def test_get_user_missing(tmp_path: Path) -> None:
    assert get_user(tmp_path, "u_does_not_exist") is None


def test_set_disabled_and_list_users(tmp_path: Path) -> None:
    u1 = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    u2 = get_or_create_user(tmp_path, provider="google", subject="s2", email="b@x.com", name="B")
    assert set_disabled(tmp_path, u1.user_id, True) is True
    reloaded = get_user(tmp_path, u1.user_id)
    assert reloaded is not None and reloaded.disabled is True
    assert set_disabled(tmp_path, "u_does_not_exist", True) is False
    assert {u.user_id for u in list_users(tmp_path)} == {u1.user_id, u2.user_id}


def test_delete_user(tmp_path: Path) -> None:
    user = get_or_create_user(tmp_path, provider="google", subject="s1", email="a@x.com", name="A")
    assert delete_user(tmp_path, user.user_id) is True
    assert get_user(tmp_path, user.user_id) is None
    assert delete_user(tmp_path, user.user_id) is False
