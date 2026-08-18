"""Unit tests for the collections store (#1417, app_collections_store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_collections_store as cs
from podcast_scraper.server.app_user_state import UserStateUnreadable

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"


def test_empty_when_unset(tmp_path: Path) -> None:
    assert cs.list_collections(tmp_path, _UID) == []


def test_create_and_list(tmp_path: Path) -> None:
    c = cs.create_collection(tmp_path, _UID, "AI takes")
    assert c["name"] == "AI takes" and c["id"].startswith("col_") and c["count"] == 0
    rows = cs.list_collections(tmp_path, _UID)
    assert [r["name"] for r in rows] == ["AI takes"]


def test_create_rejects_bad_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cs.create_collection(tmp_path, _UID, "   ")
    with pytest.raises(ValueError):
        cs.create_collection(tmp_path, _UID, "x" * 200)


def test_add_item_idempotent_and_counts(tmp_path: Path) -> None:
    cid = cs.create_collection(tmp_path, _UID, "c")["id"]
    cs.add_item(tmp_path, _UID, cid, "h1")
    cs.add_item(tmp_path, _UID, cid, "h1")  # idempotent
    cs.add_item(tmp_path, _UID, cid, "h2")
    assert cs.get_items(tmp_path, _UID, cid) == ["h1", "h2"]
    assert cs.list_collections(tmp_path, _UID)[0]["count"] == 2


def test_add_item_unknown_collection(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        cs.add_item(tmp_path, _UID, "col_missing", "h1")


def test_remove_item(tmp_path: Path) -> None:
    cid = cs.create_collection(tmp_path, _UID, "c")["id"]
    cs.add_item(tmp_path, _UID, cid, "h1")
    cs.add_item(tmp_path, _UID, cid, "h2")
    assert cs.remove_item(tmp_path, _UID, cid, "h1") == ["h2"]


def test_delete_collection_drops_membership(tmp_path: Path) -> None:
    cid = cs.create_collection(tmp_path, _UID, "c")["id"]
    cs.add_item(tmp_path, _UID, cid, "h1")
    assert cs.delete_collection(tmp_path, _UID, cid) is True
    assert cs.list_collections(tmp_path, _UID) == []
    assert cs.get_items(tmp_path, _UID, cid) == []
    assert cs.delete_collection(tmp_path, _UID, cid) is False  # already gone


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert cs.list_collections(tmp_path, "../evil") == []
    with pytest.raises(ValueError):
        cs.create_collection(tmp_path, "../evil", "c")


def test_remove_item_unknown_collection_no_ghost_write(tmp_path: Path) -> None:
    cs.create_collection(tmp_path, _UID, "real")
    assert cs.remove_item(tmp_path, _UID, "col_missing", "h1") == []
    # the unknown id must NOT have been persisted as an empty membership
    assert cs.get_items(tmp_path, _UID, "col_missing") == []
    import json

    raw = json.loads((tmp_path / "users" / _UID / "collections.json").read_text())
    assert "col_missing" not in raw["items"]


# --- the wipe class (see the same block in test_app_user_state.py) -------------------------------


def _write_raw(tmp_path: Path, text: str) -> Path:
    path = cs._path(tmp_path, _UID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("payload", ["{not json", '["a list, not the doc"]'])
@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("create", lambda d: cs.create_collection(d, _UID, "Reading list")),
        ("delete", lambda d: cs.delete_collection(d, _UID, "col_x")),
        ("add_item", lambda d: cs.add_item(d, _UID, "col_x", "h1")),
        ("remove_item", lambda d: cs.remove_item(d, _UID, "col_x", "h1")),
    ],
)
def test_mutating_over_an_unreadable_doc_raises_and_changes_nothing(
    tmp_path: Path, payload: str, label: str, mutate
) -> None:
    """A bad collections.json plus one ordinary action used to replace EVERY collection and EVERY
    membership with whatever was being written. The doc holds both, so a single write loses both."""
    path = _write_raw(tmp_path, payload)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(UserStateUnreadable):
        mutate(tmp_path)
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("payload", ["{not json", '["a list, not the doc"]'])
def test_readers_still_degrade_instead_of_raising(tmp_path: Path, payload: str) -> None:
    """The other half of the rule: browsing over a bad file shows nothing, it does not 500."""
    _write_raw(tmp_path, payload)
    assert cs.list_collections(tmp_path, _UID) == []
    assert cs.get_items(tmp_path, _UID, "col_x") == []


# --- generous count caps (#51) ------------------------------------------------------------------


def test_the_collection_count_is_capped(tmp_path: Path) -> None:
    uid = "u_0123456789abcdef01234567"
    for i in range(cs._MAX_COLLECTIONS):
        cs.create_collection(tmp_path, uid, f"c{i}")
    with pytest.raises(ValueError, match="at most"):
        cs.create_collection(tmp_path, uid, "one too many")
    # The cap rejects the NEW write; everything already stored still reads.
    assert len(cs.list_collections(tmp_path, uid)) == cs._MAX_COLLECTIONS


def test_items_per_collection_are_capped(tmp_path: Path) -> None:
    uid = "u_0123456789abcdef01234567"
    col = cs.create_collection(tmp_path, uid, "c")["id"]
    for i in range(cs._MAX_ITEMS_PER_COLLECTION):
        cs.add_item(tmp_path, uid, col, f"h_{i}")
    with pytest.raises(ValueError, match="at most"):
        cs.add_item(tmp_path, uid, col, "h_overflow")


def test_re_adding_an_existing_member_still_works_at_the_cap(tmp_path: Path) -> None:
    """Idempotent re-add must keep working at the cap, or a full collection could never be tidied
    — the check belongs on the APPEND, not on the call."""
    uid = "u_0123456789abcdef01234567"
    col = cs.create_collection(tmp_path, uid, "c")["id"]
    for i in range(cs._MAX_ITEMS_PER_COLLECTION):
        cs.add_item(tmp_path, uid, col, f"h_{i}")
    members = cs.add_item(tmp_path, uid, col, "h_0")  # already a member
    assert len(members) == cs._MAX_ITEMS_PER_COLLECTION
