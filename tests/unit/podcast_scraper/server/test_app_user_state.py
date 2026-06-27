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


def test_list_playback_newest_first(tmp_path: Path) -> None:
    assert st.list_playback(tmp_path, UID) == []
    st.set_playback(tmp_path, UID, "ep1", 10.0, 1000)
    st.set_playback(tmp_path, UID, "ep2", 20.0, 2000)
    items = st.list_playback(tmp_path, UID)
    assert [i["slug"] for i in items] == ["ep2", "ep1"]  # newest updated_at first
    assert items[0]["position_seconds"] == 20.0


def test_queue_roundtrip(tmp_path: Path) -> None:
    assert st.get_queue(tmp_path, UID) == []
    assert st.set_queue(tmp_path, UID, ["a", "b"]) == ["a", "b"]
    assert st.get_queue(tmp_path, UID) == ["a", "b"]


def test_favorites_roundtrip_idempotent_and_remove(tmp_path: Path) -> None:
    assert st.get_favorites(tmp_path, UID) == []
    st.add_favorite(tmp_path, UID, {"kind": "episode", "ref": "ep1", "label": "A"})
    st.add_favorite(tmp_path, UID, {"kind": "insight", "ref": "ep1#i1", "label": "claim"})
    # idempotent on kind+ref (re-add replaces, no dup)
    favs = st.add_favorite(tmp_path, UID, {"kind": "episode", "ref": "ep1", "label": "A2"})
    eps = [f for f in favs if f["kind"] == "episode"]
    assert len(eps) == 1 and eps[0]["label"] == "A2"
    # remove by kind+ref
    favs = st.remove_favorite(tmp_path, UID, "episode", "ep1")
    assert all(f["ref"] != "ep1" for f in favs)
    assert any(f["ref"] == "ep1#i1" for f in favs)  # insight survives
    # malformed entries (missing kind/ref) are filtered on read
    st._write(tmp_path, UID, "favorites", [{"kind": "episode"}, {"ref": "x"}, "bad"])
    assert st.get_favorites(tmp_path, UID) == []


def test_interests_roundtrip_dedup_and_isolation(tmp_path: Path) -> None:
    assert st.get_interests(tmp_path, UID) == []
    # de-dup + blank-drop, order preserved
    assert st.set_interests(tmp_path, UID, ["tc:a", "tc:b", "tc:a", ""]) == ["tc:a", "tc:b"]
    assert st.get_interests(tmp_path, UID) == ["tc:a", "tc:b"]
    # users are isolated
    st.set_interests(tmp_path, "other", ["tc:z"])
    assert st.get_interests(tmp_path, UID) == ["tc:a", "tc:b"]


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


def test_add_remove_interest(tmp_path: Path) -> None:
    assert st.get_interests(tmp_path, UID) == []
    st.add_interest(tmp_path, UID, "tc:ai")
    st.add_interest(tmp_path, UID, "person:jane")
    st.add_interest(tmp_path, UID, "tc:ai")  # idempotent
    assert st.get_interests(tmp_path, UID) == ["tc:ai", "person:jane"]
    st.remove_interest(tmp_path, UID, "tc:ai")
    assert st.get_interests(tmp_path, UID) == ["person:jane"]
    st.remove_interest(tmp_path, UID, "topic:absent")  # no-op
    assert st.get_interests(tmp_path, UID) == ["person:jane"]


def test_listen_events_append_and_list(tmp_path: Path) -> None:
    assert st.list_listen_events(tmp_path, UID) == []
    st.append_listen_event(tmp_path, UID, "ep1", "feedX", 1000)
    st.append_listen_event(tmp_path, UID, "ep1", "feedX", 1086400)
    st.append_listen_event(tmp_path, UID, "ep2", None, 1100)
    events = st.list_listen_events(tmp_path, UID)
    assert [e["slug"] for e in events] == ["ep1", "ep1", "ep2"]  # append order preserved
    assert events[0] == {"slug": "ep1", "feed_id": "feedX", "ts": 1000}
    assert events[2]["feed_id"] is None


def test_listen_events_skip_corrupt_lines(tmp_path: Path) -> None:
    st.append_listen_event(tmp_path, UID, "ep1", "feedX", 1000)
    path = tmp_path / "users" / UID / "listen_events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n\n")  # a garbage line + a blank line
    st.append_listen_event(tmp_path, UID, "ep2", "feedY", 2000)
    assert [e["slug"] for e in st.list_listen_events(tmp_path, UID)] == ["ep1", "ep2"]


def test_iter_user_ids(tmp_path: Path) -> None:
    assert st.iter_user_ids(tmp_path) == []
    st.append_listen_event(tmp_path, "alice", "ep1", "f", 1000)
    st.set_playback(tmp_path, "bob", "ep2", 5.0, 1000)
    assert set(st.iter_user_ids(tmp_path)) == {"alice", "bob"}
