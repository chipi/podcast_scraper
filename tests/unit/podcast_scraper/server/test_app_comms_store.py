"""Unit tests for the per-user comms/consent store (#1414, app_comms_store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_comms_store

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"  # matches the u_ + 24 hex shape


def test_defaults_when_unset(tmp_path: Path) -> None:
    c = app_comms_store.get_comms(tmp_path, _UID)
    assert c["digest"] == {
        "enabled": False,
        "cadence": "weekly",
        "day_of_week": 6,
        "hour": 13,
        "paused": False,
    }
    assert c["push"] == {"enabled": False}
    assert "unsubscribe_ref" not in c  # not minted until first save


def test_set_mints_ref_and_persists(tmp_path: Path) -> None:
    saved = app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True, "cadence": "daily"})
    ref = saved["unsubscribe_ref"]
    assert isinstance(ref, str) and len(ref) >= 16
    assert saved["digest"]["enabled"] is True
    assert saved["digest"]["cadence"] == "daily"

    reloaded = app_comms_store.get_comms(tmp_path, _UID)
    assert reloaded["digest"]["enabled"] is True
    assert reloaded["digest"]["cadence"] == "daily"
    assert reloaded["unsubscribe_ref"] == ref


def test_ref_is_stable_across_saves(tmp_path: Path) -> None:
    first = app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True})["unsubscribe_ref"]
    second = app_comms_store.set_comms(tmp_path, _UID, push={"enabled": True})["unsubscribe_ref"]
    assert first == second


def test_partial_merge_keeps_other_sections(tmp_path: Path) -> None:
    app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True, "cadence": "daily"})
    merged = app_comms_store.set_comms(tmp_path, _UID, push={"enabled": True})
    assert merged["digest"]["enabled"] is True
    assert merged["digest"]["cadence"] == "daily"
    assert merged["push"]["enabled"] is True


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    saved = app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True, "bogus": 1})
    assert "bogus" not in saved["digest"]


def test_unsubscribe_disables_digest_and_is_idempotent(tmp_path: Path) -> None:
    ref = app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True})["unsubscribe_ref"]
    assert app_comms_store.unsubscribe(tmp_path, ref) is True
    assert app_comms_store.get_comms(tmp_path, _UID)["digest"]["enabled"] is False
    # Re-hitting the same link is a no-op that still reports success.
    assert app_comms_store.unsubscribe(tmp_path, ref) is True


def test_unsubscribe_unknown_ref_is_false(tmp_path: Path) -> None:
    app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True})
    assert app_comms_store.unsubscribe(tmp_path, "no-such-ref") is False
    assert app_comms_store.unsubscribe(tmp_path, "") is False


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert app_comms_store.get_comms(tmp_path, "../evil")["digest"]["enabled"] is False
    with pytest.raises(ValueError):
        app_comms_store.set_comms(tmp_path, "../evil", digest={"enabled": True})
