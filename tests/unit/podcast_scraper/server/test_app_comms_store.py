"""Unit tests for the per-user comms/consent store — the type×channel matrix (#1414 → wave-I)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_comms_store

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"  # matches the u_ + 24 hex shape


def test_defaults_when_unset(tmp_path: Path) -> None:
    c = app_comms_store.get_comms(tmp_path, _UID)
    # Outbound channels opt-in (off); in-app inbox on by default, for every type.
    for ntype in app_comms_store.TYPES:
        assert c["types"][ntype] == {"email": False, "push": False, "in_app": True}
    assert c["digest_schedule"] == {
        "cadence": "weekly",
        "day_of_week": 6,
        "hour": 13,
        "paused": False,
    }
    assert "unsubscribe_ref" not in c  # not minted until first save


def test_set_mints_ref_and_persists(tmp_path: Path) -> None:
    saved = app_comms_store.set_comms(
        tmp_path, _UID, types={"digest": {"email": True}}, digest_schedule={"cadence": "daily"}
    )
    ref = saved["unsubscribe_ref"]
    assert isinstance(ref, str) and len(ref) >= 16
    assert saved["types"]["digest"]["email"] is True
    assert saved["digest_schedule"]["cadence"] == "daily"

    reloaded = app_comms_store.get_comms(tmp_path, _UID)
    assert reloaded["types"]["digest"]["email"] is True
    assert reloaded["digest_schedule"]["cadence"] == "daily"
    assert reloaded["unsubscribe_ref"] == ref


def test_set_with_no_sections_still_mints_ref(tmp_path: Path) -> None:
    # The digest path calls set_comms() with nothing just to mint the ref.
    saved = app_comms_store.set_comms(tmp_path, _UID)
    assert isinstance(saved["unsubscribe_ref"], str)


def test_ref_is_stable_across_saves(tmp_path: Path) -> None:
    first = app_comms_store.set_comms(tmp_path, _UID, types={"digest": {"email": True}})[
        "unsubscribe_ref"
    ]
    second = app_comms_store.set_comms(tmp_path, _UID, types={"new_episodes": {"push": True}})[
        "unsubscribe_ref"
    ]
    assert first == second


def test_partial_merge_keeps_other_cells(tmp_path: Path) -> None:
    app_comms_store.set_comms(tmp_path, _UID, types={"digest": {"email": True}})
    merged = app_comms_store.set_comms(tmp_path, _UID, types={"new_episodes": {"push": True}})
    assert merged["types"]["digest"]["email"] is True  # untouched
    assert merged["types"]["digest"]["in_app"] is True  # default preserved
    assert merged["types"]["new_episodes"]["push"] is True


def test_unknown_type_and_channel_keys_are_ignored(tmp_path: Path) -> None:
    saved = app_comms_store.set_comms(
        tmp_path, _UID, types={"digest": {"email": True, "bogus": 1}, "nope": {"email": True}}
    )
    assert "bogus" not in saved["types"]["digest"]
    assert "nope" not in saved["types"]


def test_channel_enabled_helper(tmp_path: Path) -> None:
    c = app_comms_store.set_comms(tmp_path, _UID, types={"new_episodes": {"push": True}})
    assert app_comms_store.channel_enabled(c, "new_episodes", "push") is True
    assert app_comms_store.channel_enabled(c, "new_episodes", "email") is False
    assert app_comms_store.channel_enabled(c, "bogus", "push") is False


def test_set_channel_convenience(tmp_path: Path) -> None:
    saved = app_comms_store.set_channel(tmp_path, _UID, "product", "email", True)
    assert saved["types"]["product"]["email"] is True


def test_disable_push_everywhere(tmp_path: Path) -> None:
    app_comms_store.set_comms(
        tmp_path,
        _UID,
        types={"digest": {"push": True}, "new_episodes": {"push": True}},
    )
    saved = app_comms_store.disable_push_everywhere(tmp_path, _UID)
    for ntype in app_comms_store.TYPES:
        assert saved["types"][ntype]["push"] is False


def test_unsubscribe_disables_digest_email_and_is_idempotent(tmp_path: Path) -> None:
    ref = app_comms_store.set_comms(tmp_path, _UID, types={"digest": {"email": True}})[
        "unsubscribe_ref"
    ]
    assert app_comms_store.unsubscribe(tmp_path, ref) is True
    c = app_comms_store.get_comms(tmp_path, _UID)
    assert c["types"]["digest"]["email"] is False
    # The email link governs email only — in-app is untouched.
    assert c["types"]["digest"]["in_app"] is True
    # Re-hitting the same link is a no-op that still reports success.
    assert app_comms_store.unsubscribe(tmp_path, ref) is True


def test_unsubscribe_unknown_ref_is_false(tmp_path: Path) -> None:
    app_comms_store.set_comms(tmp_path, _UID, types={"digest": {"email": True}})
    assert app_comms_store.unsubscribe(tmp_path, "no-such-ref") is False
    assert app_comms_store.unsubscribe(tmp_path, "") is False


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert app_comms_store.get_comms(tmp_path, "../evil")["types"]["digest"]["email"] is False
    with pytest.raises(ValueError):
        app_comms_store.set_comms(tmp_path, "../evil", types={"digest": {"email": True}})
