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


# ── item-level ops (two devices editing one list keep both changes) ─────────────────────────────


def _op(**kw: Any) -> Any:
    from podcast_scraper.server.schemas import PreferenceOp

    return PreferenceOp(**kw)


def test_push_front_moves_an_existing_item_first_and_caps() -> None:
    out = app_user_preferences.apply_ops(
        {"r": ["a", "b", "c"]}, [_op(op="push_front", key="r", value="c", max=2)]
    )
    assert out == {"r": ["c", "a"]}


def test_push_front_matches_objects_by_fields_and_case() -> None:
    saved = [{"q": "Climate ", "scope": "all", "saved_at": 1}, {"q": "x", "scope": "mine"}]
    item = {"q": "climate", "scope": "all", "saved_at": 2}
    out = app_user_preferences.apply_ops(
        {"s": saved},
        [_op(op="push_front", key="s", value=item, match=["q", "scope"], fold_case=True)],
    )
    assert out["s"] == [item, {"q": "x", "scope": "mine"}]


def test_remove_drops_only_the_matching_item() -> None:
    viewed = [{"kind": "show", "id": "1"}, {"kind": "episode", "id": "1"}]
    out = app_user_preferences.apply_ops(
        {"v": viewed},
        [_op(op="remove", key="v", value={"kind": "show", "id": "1"}, match=["kind", "id"])],
    )
    assert out["v"] == [{"kind": "episode", "id": "1"}]


def test_clear_set_field_and_increment() -> None:
    out = app_user_preferences.apply_ops(
        {"r": ["a"], "m": {"ep-1": 2.5, "ep-2": 1}, "n": 2, "keep": True},
        [
            _op(op="clear", key="r"),
            _op(op="set_field", key="m", field="ep-3", value=4),
            _op(op="set_field", key="m", field="ep-1", value=None),
            _op(op="increment", key="n"),
            _op(op="increment", key="fresh", by=3),
        ],
    )
    assert out == {"r": [], "m": {"ep-2": 1, "ep-3": 4}, "n": 3, "fresh": 3, "keep": True}


def test_a_key_of_the_wrong_type_starts_over_empty() -> None:
    out = app_user_preferences.apply_ops(
        {"r": "not a list", "n": True},
        [
            _op(op="push_front", key="r", value="a"),
            _op(op="increment", key="n"),
        ],
    )
    assert out == {"r": ["a"], "n": 1}


def test_ops_that_cannot_apply_are_refused() -> None:
    with pytest.raises(ValueError):
        app_user_preferences.apply_ops({}, [_op(op="push_front", key="r")])
    with pytest.raises(ValueError):
        app_user_preferences.apply_ops({}, [_op(op="set_field", key="m", value=1)])
    with pytest.raises(ValueError):
        app_user_preferences.apply_ops({}, [_op(op="remove", key="r", value="a", match=["id"])])
    # A value without a matched field would have matched every item without it.
    with pytest.raises(ValueError):
        app_user_preferences.apply_ops(
            {}, [_op(op="remove", key="r", value={"kind": "show"}, match=["kind", "id"])]
        )


def test_an_item_missing_a_matched_field_is_never_dropped() -> None:
    kept = [{"kind": "show"}, {"kind": "show", "id": None}, {"id": "1"}]
    out = app_user_preferences.apply_ops(
        {"v": [*kept, {"kind": "show", "id": "1"}]},
        [_op(op="remove", key="v", value={"kind": "show", "id": "1"}, match=["kind", "id"])],
    )
    assert out["v"] == kept
    # A field present as null is not a field missing.
    out = app_user_preferences.apply_ops(
        {"v": kept},
        [_op(op="remove", key="v", value={"kind": "show", "id": None}, match=["kind", "id"])],
    )
    assert out["v"] == [{"kind": "show"}, {"id": "1"}]


def test_values_compare_with_their_json_type() -> None:
    out = app_user_preferences.apply_ops(
        {"r": [1, True, 1.5], "s": [{"id": 1}, {"id": True}]},
        [
            _op(op="remove", key="r", value=True),
            _op(op="push_front", key="s", value={"id": 1, "new": True}, match=["id"]),
        ],
    )
    assert out == {"r": [1, 1.5], "s": [{"id": 1, "new": True}, {"id": True}]}


def test_remove_from_an_absent_key_leaves_it_absent() -> None:
    out = app_user_preferences.apply_ops({"other": 1}, [_op(op="remove", key="r", value="a")])
    assert out == {"other": 1}


def test_two_devices_adding_to_one_list_at_once_keep_both(tmp_path: Path) -> None:
    """The row lock serialises the ops, so every device's search survives (db-era sweep row 29);
    each sending its whole list kept only the last one's."""
    terms = [f"term-{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda t: app_user_preferences.patch_preferences(
                    tmp_path, USER, {}, [_op(op="push_front", key="recents", value=t, max=20)]
                ),
                terms,
            )
        )
    stored = app_user_preferences.get_preferences(tmp_path, USER)["recents"]
    assert sorted(stored) == sorted(terms)
