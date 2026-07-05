"""Unit tests for the file-based per-user identity store (#1063)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server.app_user_store import (
    delete_user,
    get_or_create_user,
    get_user,
    list_users,
    set_disabled,
    set_role,
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


def test_path_traversal_user_ids_are_rejected_not_found(tmp_path: Path) -> None:
    """A non-conforming user_id (``..`` / ``/`` / non-hex) must never touch the
    filesystem: lookups/mutations return the graceful not-found result and a file
    outside the users/ tree is never read or removed. Real ids are always
    ``user_id_for`` = ``u_`` + 24 hex; anything else is treated as unknown."""
    victim = tmp_path / "secret.txt"
    victim.write_text("keep me", encoding="utf-8")

    for bad in ("../secret", "u_../../secret", "u_/etc/passwd", "u_NOTHEX", "", "u_short"):
        assert get_user(tmp_path, bad) is None
        assert set_disabled(tmp_path, bad, True) is False
        assert set_role(tmp_path, bad, "admin") is False
        assert delete_user(tmp_path, bad) is False

    assert victim.read_text(encoding="utf-8") == "keep me"


def test_valid_shape_but_absent_user_id_is_not_found(tmp_path: Path) -> None:
    """A well-formed but non-existent id is a clean not-found (distinct from rejected)."""
    assert get_user(tmp_path, "u_" + "0" * 24) is None
    assert delete_user(tmp_path, "u_" + "1" * 24) is False
    assert set_disabled(tmp_path, "u_" + "2" * 24, True) is False
    assert set_role(tmp_path, "u_" + "3" * 24, "admin") is False
