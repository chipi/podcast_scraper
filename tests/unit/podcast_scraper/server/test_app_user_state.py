"""Unit tests for per-user state files — playback, queue, library (#1065)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_user_state as st

UID = "u_test"


def test_playback_roundtrip(tmp_path: Path) -> None:
    assert st.get_playback(tmp_path, UID, "ep") is None
    rec = st.set_playback(tmp_path, UID, "ep", 42.5, 1000)
    assert rec == {"position_seconds": 42.5, "updated_at": 1000}
    loaded = st.get_playback(tmp_path, UID, "ep")
    assert loaded is not None and loaded["position_seconds"] == 42.5
    # a second episode coexists without clobbering the first
    st.set_playback(tmp_path, UID, "ep2", 5.0, 1001)
    first = st.get_playback(tmp_path, UID, "ep")
    assert first is not None and first["position_seconds"] == 42.5


def test_queue_roundtrip(tmp_path: Path) -> None:
    assert st.get_queue(tmp_path, UID) == []
    assert st.set_queue(tmp_path, UID, ["a", "b"]) == ["a", "b"]
    assert st.get_queue(tmp_path, UID) == ["a", "b"]


def test_library_add_dedupe_remove(tmp_path: Path) -> None:
    assert st.get_library(tmp_path, UID) == []
    st.add_subscription(tmp_path, UID, {"feed_id": "f1", "title": "One"})
    st.add_subscription(tmp_path, UID, {"feed_id": "f2", "title": "Two"})
    st.add_subscription(
        tmp_path, UID, {"feed_id": "f1", "title": "One-updated"}
    )  # dedupe on feed_id
    library = st.get_library(tmp_path, UID)
    assert {x["feed_id"] for x in library} == {"f1", "f2"}
    assert next(x for x in library if x["feed_id"] == "f1")["title"] == "One-updated"
    st.remove_subscription(tmp_path, UID, "f1")
    assert {x["feed_id"] for x in st.get_library(tmp_path, UID)} == {"f2"}
