"""Unit tests for the USERPREFS-1 file-backed preferences store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import app_user_preferences

USER = "u_test"


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


def test_malformed_json_on_disk_returns_empty(tmp_path: Path) -> None:
    """Corrupted preferences.json should not brick the user's session — the
    server returns empty and the client falls back to local defaults."""
    path = tmp_path / "users" / USER / "preferences.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json}", encoding="utf-8")
    assert app_user_preferences.get_preferences(tmp_path, USER) == {}
