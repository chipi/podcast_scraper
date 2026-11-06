"""Unit tests for the USERPREFS-1 preferences store (a ``user_preferences`` row per account)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import app_user_preferences
from tests._database import register_user

USER = "u_test"


@pytest.fixture(autouse=True)
def _accounts() -> None:
    """Preferences belong to an account: the row references ``users``."""
    for uid in (USER, "alice", "bob"):
        register_user(uid)


def test_get_default_is_empty_dict_when_no_file(tmp_path: Path) -> None:
    assert app_user_preferences.get_preferences(tmp_path, USER) == {}


def test_replace_and_get_roundtrip(tmp_path: Path) -> None:
    payload = {"theme": "dark", "graph": {"lens": True}}
    assert app_user_preferences.replace_preferences(tmp_path, USER, payload) == payload
    assert app_user_preferences.get_preferences(tmp_path, USER) == payload


def test_patch_shallow_merges_top_level_keys(tmp_path: Path) -> None:
    app_user_preferences.replace_preferences(tmp_path, USER, {"theme": "dark", "path": "/a"})
    merged = app_user_preferences.patch_preferences(tmp_path, USER, {"theme": "light"})
    # theme updated, path preserved.
    assert merged == {"theme": "light", "path": "/a"}


def test_patch_null_value_deletes_key(tmp_path: Path) -> None:
    app_user_preferences.replace_preferences(
        tmp_path, USER, {"theme": "dark", "path": "/a", "leftPanel": True}
    )
    merged = app_user_preferences.patch_preferences(tmp_path, USER, {"path": None})
    assert "path" not in merged
    assert merged == {"theme": "dark", "leftPanel": True}


def test_patch_deep_replaces_nested_objects_shallowly(tmp_path: Path) -> None:
    """PATCH is a top-level shallow merge only; nested values are replaced whole."""
    app_user_preferences.replace_preferences(
        tmp_path, USER, {"graph": {"lens": True, "hover": False}}
    )
    merged = app_user_preferences.patch_preferences(tmp_path, USER, {"graph": {"lens": False}})
    # `hover` inside `graph` is lost — top-level `graph` was replaced whole.
    assert merged == {"graph": {"lens": False}}


def test_replace_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        bad_payload: Any = ["not", "a", "dict"]
        app_user_preferences.replace_preferences(tmp_path, USER, bad_payload)


def test_patch_rejects_non_dict_updates(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        bad_updates: Any = [("key", "value")]
        app_user_preferences.patch_preferences(tmp_path, USER, bad_updates)


def test_users_are_isolated(tmp_path: Path) -> None:
    app_user_preferences.replace_preferences(tmp_path, "alice", {"theme": "dark"})
    app_user_preferences.replace_preferences(tmp_path, "bob", {"theme": "light"})
    assert app_user_preferences.get_preferences(tmp_path, "alice") == {"theme": "dark"}
    assert app_user_preferences.get_preferences(tmp_path, "bob") == {"theme": "light"}


def test_concurrent_first_patches_both_land(tmp_path: Path) -> None:
    """Two tabs' first writes at once: the row lock serialises them, neither key is lost."""
    keys = [f"k{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda k: app_user_preferences.patch_preferences(tmp_path, USER, {k: True}), keys
            )
        )
    assert app_user_preferences.get_preferences(tmp_path, USER) == {k: True for k in keys}


def test_preferences_go_with_the_account(tmp_path: Path) -> None:
    from podcast_scraper.server.app_user_store import delete_user

    register_user("u_" + "a" * 24)
    app_user_preferences.replace_preferences(tmp_path, "u_" + "a" * 24, {"theme": "dark"})
    assert delete_user(tmp_path, "u_" + "a" * 24)
    assert app_user_preferences.get_preferences(tmp_path, "u_" + "a" * 24) == {}
