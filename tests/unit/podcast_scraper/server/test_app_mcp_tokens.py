"""Unit tests for the MCP PAT store (RFC-112 slice 1, #1471)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_mcp_tokens as mt

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"
_UID2 = "u_fedcba9876543210fedcba98"


def test_create_returns_plaintext_once_and_hashes_at_rest(tmp_path: Path) -> None:
    plaintext, meta = mt.create_token(tmp_path, _UID, "Claude Code")
    assert plaintext.startswith("clp_mcp_")
    assert meta["label"] == "Claude Code" and meta["id"].startswith("mtk_")
    # stored file holds only the hash, never the plaintext
    stored = (tmp_path / "users" / _UID / "mcp_tokens.json").read_text()
    assert plaintext not in stored
    assert "hash" in stored
    # list never leaks the hash
    listed = mt.list_tokens(tmp_path, _UID)
    assert listed[0]["id"] == meta["id"] and "hash" not in listed[0]


def test_verify_resolves_owner_and_stamps_last_used(tmp_path: Path) -> None:
    plaintext, meta = mt.create_token(tmp_path, _UID, "a")
    assert mt.verify_token(tmp_path, plaintext) == _UID
    assert mt.list_tokens(tmp_path, _UID)[0]["last_used_at"] is not None


def test_verify_unknown_or_empty(tmp_path: Path) -> None:
    assert mt.verify_token(tmp_path, "clp_mcp_bogus") is None
    assert mt.verify_token(tmp_path, "") is None


def test_verify_is_o1_across_users(tmp_path: Path) -> None:
    p1, _ = mt.create_token(tmp_path, _UID, "a")
    p2, _ = mt.create_token(tmp_path, _UID2, "b")
    assert mt.verify_token(tmp_path, p1) == _UID
    assert mt.verify_token(tmp_path, p2) == _UID2  # index routes to the right owner


def test_revoke_removes_from_store_and_index(tmp_path: Path) -> None:
    plaintext, meta = mt.create_token(tmp_path, _UID, "a")
    assert mt.revoke_token(tmp_path, _UID, meta["id"]) is True
    assert mt.list_tokens(tmp_path, _UID) == []
    assert mt.verify_token(tmp_path, plaintext) is None  # index entry gone
    assert mt.revoke_token(tmp_path, _UID, meta["id"]) is False  # already gone


def test_multiple_tokens_per_user(tmp_path: Path) -> None:
    mt.create_token(tmp_path, _UID, "laptop")
    mt.create_token(tmp_path, _UID, "desktop")
    assert {t["label"] for t in mt.list_tokens(tmp_path, _UID)} == {"laptop", "desktop"}


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert mt.list_tokens(tmp_path, "../evil") == []
    with pytest.raises(ValueError):
        mt.create_token(tmp_path, "../evil", "a")
