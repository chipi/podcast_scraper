"""Unit tests for the corpus revision counter + change log (RFC-114 Phase 1, #1470)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_corpus_revision as rev

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"
_ROOT = Path("/unused")


@pytest.fixture(autouse=True)
def _stub_membership(monkeypatch: pytest.MonkeyPatch):
    """Drive membership from a mutable dict so tests control adds/removes deterministically."""
    state: dict[str, set[str]] = {"experienced": set(), "saved": set()}
    monkeypatch.setattr(
        rev.app_user_corpus, "experienced_episode_set", lambda r, d, u: set(state["experienced"])
    )
    monkeypatch.setattr(rev.app_user_corpus, "saved_episode_set", lambda d, u: set(state["saved"]))
    return state


def test_empty_revision_zero(tmp_path: Path) -> None:
    assert rev.current(_ROOT, tmp_path, _UID) == 0


def test_add_bumps_revision_and_logs(tmp_path: Path, _stub_membership) -> None:
    _stub_membership["experienced"] = {"ep-a"}
    r1 = rev.reconcile(_ROOT, tmp_path, _UID)
    assert r1 == 1
    ch = rev.changes_since(_ROOT, tmp_path, _UID, 0)
    assert ch["revision"] == 1
    assert ch["events"] == [{"seq": 1, "kind": "added", "facet": "experienced", "ref": "ep-a"}]


def test_no_change_no_bump(tmp_path: Path, _stub_membership) -> None:
    _stub_membership["experienced"] = {"ep-a"}
    assert rev.reconcile(_ROOT, tmp_path, _UID) == 1
    assert rev.reconcile(_ROOT, tmp_path, _UID) == 1  # idempotent, no new event


def test_removal_emits_tombstone(tmp_path: Path, _stub_membership) -> None:
    _stub_membership["experienced"] = {"ep-a"}
    rev.reconcile(_ROOT, tmp_path, _UID)
    _stub_membership["experienced"] = set()  # ep-a removed (e.g. highlight deleted)
    r2 = rev.reconcile(_ROOT, tmp_path, _UID)
    assert r2 == 2
    ch = rev.changes_since(_ROOT, tmp_path, _UID, 1)
    assert ch["events"] == [{"seq": 2, "kind": "removed", "facet": "experienced", "ref": "ep-a"}]


def test_since_filters_delta(tmp_path: Path, _stub_membership) -> None:
    _stub_membership["experienced"] = {"ep-a"}
    rev.reconcile(_ROOT, tmp_path, _UID)  # rev 1
    _stub_membership["experienced"] = {"ep-a", "ep-b"}
    rev.reconcile(_ROOT, tmp_path, _UID)  # rev 2 (ep-b added)
    ch = rev.changes_since(_ROOT, tmp_path, _UID, 1)
    assert [e["ref"] for e in ch["events"]] == ["ep-b"]


def test_saved_facet_tracked_distinctly(tmp_path: Path, _stub_membership) -> None:
    _stub_membership["saved"] = {"ep-fav"}
    rev.reconcile(_ROOT, tmp_path, _UID)
    ch = rev.changes_since(_ROOT, tmp_path, _UID, 0)
    assert ch["events"] == [{"seq": 1, "kind": "added", "facet": "saved", "ref": "ep-fav"}]


def test_saved_minus_experienced(tmp_path: Path, _stub_membership) -> None:
    # An episode both favorited AND experienced is `experienced` only, never double-counted.
    _stub_membership["experienced"] = {"ep-a"}
    _stub_membership["saved"] = {"ep-a"}
    rev.reconcile(_ROOT, tmp_path, _UID)
    ch = rev.changes_since(_ROOT, tmp_path, _UID, 0)
    assert ch["events"] == [{"seq": 1, "kind": "added", "facet": "experienced", "ref": "ep-a"}]


def test_unsafe_user_id(tmp_path: Path) -> None:
    assert rev.current(_ROOT, tmp_path, "../evil") == 0
