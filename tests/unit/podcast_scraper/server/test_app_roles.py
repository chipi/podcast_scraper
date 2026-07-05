"""Unit tests for the platform role vocabulary + login-role resolution (#1128)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_roles
from podcast_scraper.server.app_user_store import (
    create_user,
    get_or_create_user,
    get_user,
    set_role,
)


def test_role_vocabulary_and_rank() -> None:
    assert app_roles.ROLES == ("listener", "creator", "admin")
    assert app_roles.rank("listener") < app_roles.rank("creator") < app_roles.rank("admin")
    assert app_roles.at_least("admin", "creator") is True
    assert app_roles.at_least("creator", "admin") is False
    assert app_roles.can_use_viewer("creator") and app_roles.can_use_viewer("admin")
    assert not app_roles.can_use_viewer("listener")
    assert app_roles.is_admin("admin") and not app_roles.is_admin("creator")


def test_normalize_role_coerces_unknown_to_default() -> None:
    assert app_roles.normalize_role("ADMIN") == "admin"
    assert app_roles.normalize_role("  creator ") == "creator"
    assert app_roles.normalize_role(None) == "listener"
    assert app_roles.normalize_role("wizard") == "listener"
    assert app_roles.normalize_role("wizard", default="creator") == "creator"


def test_admin_emails_from_env(monkeypatch) -> None:
    monkeypatch.setenv("APP_ADMIN_EMAILS", " Boss@X.com , admin@y.io ")
    assert app_roles.admin_emails_from_env() == frozenset({"boss@x.com", "admin@y.io"})
    monkeypatch.delenv("APP_ADMIN_EMAILS", raising=False)
    assert app_roles.admin_emails_from_env() == frozenset()


def test_resolve_login_role_admin_allowlist_wins() -> None:
    # Allowlisted email → admin, regardless of grant or current role.
    role = app_roles.resolve_login_role(
        "listener", email="boss@x.com", grant=None, admin_emails=frozenset({"boss@x.com"})
    )
    assert role == "admin"


def test_resolve_login_role_creator_grant_promotes_listener() -> None:
    role = app_roles.resolve_login_role(
        "listener", email="a@b.com", grant="creator", admin_emails=frozenset()
    )
    assert role == "creator"


def test_resolve_login_role_never_downgrades_or_grants_admin() -> None:
    # An existing admin is never knocked down by a creator grant.
    assert (
        app_roles.resolve_login_role(
            "admin", email="a@b.com", grant="creator", admin_emails=frozenset()
        )
        == "admin"
    )
    # 'admin' is never grantable via the request hint.
    assert (
        app_roles.resolve_login_role(
            "listener", email="a@b.com", grant="admin", admin_emails=frozenset()
        )
        == "listener"
    )


def test_store_new_user_defaults_to_listener(tmp_path: Path) -> None:
    u = get_or_create_user(tmp_path, provider="mock", subject="s", email="a@b.com", name="A")
    assert u.role == "listener"
    assert get_user(tmp_path, u.user_id).role == "listener"  # type: ignore[union-attr]


def test_store_create_with_role_and_set_role(tmp_path: Path) -> None:
    u = get_or_create_user(
        tmp_path, provider="mock", subject="s", email="a@b.com", name="A", role="creator"
    )
    assert u.role == "creator"
    assert set_role(tmp_path, u.user_id, "admin") is True
    assert get_user(tmp_path, u.user_id).role == "admin"  # type: ignore[union-attr]
    # an unknown role coerces to the default rather than persisting garbage
    set_role(tmp_path, u.user_id, "wizard")
    assert get_user(tmp_path, u.user_id).role == "listener"  # type: ignore[union-attr]
    assert set_role(tmp_path, "u_missing", "admin") is False


def test_store_set_role_preserves_other_fields(tmp_path: Path) -> None:
    u = create_user(
        tmp_path, provider="mock", subject="s", email="a@b.com", name="Amy", role="creator"
    )
    set_role(tmp_path, u.user_id, "admin")
    reloaded = get_user(tmp_path, u.user_id)
    assert reloaded is not None
    assert reloaded.email == "a@b.com" and reloaded.name == "Amy" and reloaded.disabled is False


def test_store_legacy_profile_without_role_reads_as_listener(tmp_path: Path) -> None:
    # A profile written before #1128 has no 'role' key → must read back as listener.
    import json

    legacy_id = "u_" + "0" * 24  # opaque user_id_for shape (pre-#1128 profile, no role)
    udir = tmp_path / "users" / legacy_id
    udir.mkdir(parents=True)
    (udir / "profile.json").write_text(
        json.dumps({"email": "old@x.com", "name": "Old", "provider": "google", "subject": "z"}),
        encoding="utf-8",
    )
    u = get_user(tmp_path, legacy_id)
    assert u is not None and u.role == "listener"
