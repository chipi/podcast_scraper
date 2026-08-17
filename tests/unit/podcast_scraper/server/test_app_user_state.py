"""Unit tests for per-user state files — playback, queue, library (#1065)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server import app_user_state as st

UID = "u_test"


def test_playback_roundtrip(tmp_path: Path) -> None:
    assert st.get_playback(tmp_path, UID, "ep") is None
    rec = st.set_playback(tmp_path, UID, "ep", 42.5, 1000)
    assert rec == {"position_seconds": 42.5, "updated_at": 1000, "finished": False}
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
    # Canonical envelope (ADR-119): {ts (ISO-8601), schema, event_type, slug, feed_id?}.
    assert events[0]["slug"] == "ep1" and events[0]["feed_id"] == "feedX"
    assert events[0]["event_type"] == "listen" and events[0]["schema"] == 1
    assert events[0]["ts"] == "1970-01-01T00:16:40+00:00"  # epoch 1000 -> canonical ISO
    # feed_id=None is dropped from the lean envelope; the event is still recorded.
    assert events[2]["slug"] == "ep2" and "feed_id" not in events[2]


def test_listen_events_skip_corrupt_lines(tmp_path: Path) -> None:
    st.append_listen_event(tmp_path, UID, "ep1", "feedX", 1000)
    path = tmp_path / "users" / UID / "listen_events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n\n")  # a garbage line + a blank line
    st.append_listen_event(tmp_path, UID, "ep2", "feedY", 2000)
    assert [e["slug"] for e in st.list_listen_events(tmp_path, UID)] == ["ep1", "ep2"]


def test_listen_events_unreadable_file_is_empty(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    # The file exists but read_text raises OSError (e.g. permissions/IO) → treated as no events.
    st.append_listen_event(tmp_path, UID, "ep1", "feedX", 1000)
    real_read_text = Path.read_text

    def boom(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "listen_events.jsonl":
            raise OSError("unreadable")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", boom)
    assert st.list_listen_events(tmp_path, UID) == []


def test_iter_user_ids(tmp_path: Path) -> None:
    assert st.iter_user_ids(tmp_path) == []
    st.append_listen_event(tmp_path, "alice", "ep1", "f", 1000)
    st.set_playback(tmp_path, "bob", "ep2", 5.0, 1000)
    assert set(st.iter_user_ids(tmp_path)) == {"alice", "bob"}


def _write_raw(tmp_path: Path, name: str, text: str) -> None:
    path = st._state_path(tmp_path, UID, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_read_corrupt_json_falls_back_to_default(tmp_path: Path) -> None:
    # A corrupt state file is treated as "unset" (the default), not an error.
    _write_raw(tmp_path, "queue", "{not json")
    assert st.get_queue(tmp_path, UID) == []
    _write_raw(tmp_path, "interests", "}}bad")
    assert st.get_interests(tmp_path, UID) == []


def test_playback_readers_tolerate_non_dict_payload_but_the_writer_refuses(tmp_path: Path) -> None:
    """READING a wrong-shaped playback.json degrades; WRITING over it must not.

    This test used to assert "set_playback rebuilds a fresh dict over the bad payload" — it pinned
    the data loss. Rebuilding is indistinguishable from wiping: the file is either hand-corrupted or
    written by a schema version this build does not know, and overwriting destroys it either way.
    """
    _write_raw(tmp_path, "playback", json.dumps([1, 2, 3]))
    assert st.get_playback(tmp_path, UID, "ep") is None
    assert st.list_playback(tmp_path, UID) == []  # non-dict → empty list, for display only

    before = st._state_path(tmp_path, UID, "playback").read_text(encoding="utf-8")
    with pytest.raises(st.UserStateUnreadable):
        st.set_playback(tmp_path, UID, "ep", 9.0, 1000)
    assert st._state_path(tmp_path, UID, "playback").read_text(encoding="utf-8") == before


def test_list_playback_skips_non_dict_records(tmp_path: Path) -> None:
    # A dict whose values are not all records: the non-dict value is skipped.
    _write_raw(
        tmp_path,
        "playback",
        json.dumps({"ep1": {"position_seconds": 3.0, "updated_at": 1}, "ep2": "garbage"}),
    )
    items = st.list_playback(tmp_path, UID)
    assert [i["slug"] for i in items] == ["ep1"]


def test_favorites_non_list_payload_is_empty(tmp_path: Path) -> None:
    _write_raw(tmp_path, "favorites", json.dumps({"kind": "episode"}))
    assert st.get_favorites(tmp_path, UID) == []


def test_get_queue_non_list_payload_is_empty(tmp_path: Path) -> None:
    _write_raw(tmp_path, "queue", json.dumps({"a": 1}))
    assert st.get_queue(tmp_path, UID) == []


def test_get_library_non_list_payload_is_empty(tmp_path: Path) -> None:
    _write_raw(tmp_path, "library", json.dumps("nope"))
    assert st.get_library(tmp_path, UID) == []


# --- the wipe class: a mutator must never persist over a file it could not read -----------------
#
# Every mutator in this module is a read-modify-write. When the read answered a corrupt or
# unrecognised file with the empty default, the write that followed replaced the user's entire
# history with whatever single row was being added. One bad byte plus one ordinary interaction
# (saving a position, following a show, creating a collection) was total, silent, permanent loss.
#
# The rule these tests pin: absent is safe to default; unreadable and unrecognised are not. Readers
# stay lenient — they only decide what renders — and each of those is covered separately above.

_MUTATORS = [
    # (state file, bad payload, callable that mutates it)
    ("playback", "{not json", lambda d: st.set_playback(d, UID, "ep", 9.0, 1000)),
    ("playback", json.dumps([1, 2, 3]), lambda d: st.set_playback(d, UID, "ep", 9.0, 1000)),
    ("favorites", "{not json", lambda d: st.add_favorite(d, UID, {"kind": "episode", "ref": "e"})),
    ("favorites", json.dumps({"kind": "episode"}), lambda d: st.remove_favorite(d, UID, "k", "r")),
    ("library", "]]bad", lambda d: st.add_subscription(d, UID, {"feed_id": "f1"})),
    ("library", json.dumps("nope"), lambda d: st.remove_subscription(d, UID, "f1")),
    ("interests", "{not json", lambda d: st.add_interest(d, UID, "topic:ai")),
    ("interests", json.dumps({"a": 1}), lambda d: st.remove_interest(d, UID, "topic:ai")),
    ("highlights", "{not json", lambda d: st.add_note(d, UID, {"id": "n1"})),
    ("resurfacing", "{not json", lambda d: st.mark_surfaced(d, UID, "h1", 1000)),
]


@pytest.mark.parametrize(
    ("name", "payload", "mutate"),
    _MUTATORS,
    ids=[f"{n}-{i}" for i, (n, _p, _m) in enumerate(_MUTATORS)],
)
def test_mutating_over_an_unreadable_file_raises_and_changes_nothing(
    tmp_path: Path, name: str, payload: str, mutate
) -> None:
    target = name if name != "highlights" else "notes"  # add_note writes notes.json
    _write_raw(tmp_path, target, payload)
    before = st._state_path(tmp_path, UID, target).read_text(encoding="utf-8")
    with pytest.raises(st.UserStateUnreadable):
        mutate(tmp_path)
    assert st._state_path(tmp_path, UID, target).read_text(encoding="utf-8") == before


def test_absent_file_is_still_safe_to_default(tmp_path: Path) -> None:
    """The other half of the rule: a file that was never written must NOT raise."""
    assert st.set_playback(tmp_path, UID, "ep", 1.0, 1)["position_seconds"] == 1.0
    assert st.add_favorite(tmp_path, UID, {"kind": "episode", "ref": "e"})
    assert st.add_subscription(tmp_path, UID, {"feed_id": "f1"})
    assert st.add_interest(tmp_path, UID, "topic:ai") == ["topic:ai"]


@pytest.mark.parametrize(
    ("name", "seed", "mutate", "survivor_key"),
    [
        (
            "favorites",
            [{"kind": "episode", "ref": "keep"}, {"schema_v2_only": "row"}],
            lambda d: st.add_favorite(d, UID, {"kind": "insight", "ref": "new"}),
            "schema_v2_only",
        ),
        (
            "library",
            [{"feed_id": "keep"}, {"schema_v2_only": "row"}],
            lambda d: st.add_subscription(d, UID, {"feed_id": "new"}),
            "schema_v2_only",
        ),
    ],
)
def test_a_mutation_does_not_purge_rows_the_getter_filters_out(
    tmp_path: Path, name: str, seed, mutate, survivor_key: str
) -> None:
    """Dropping an unrenderable row on READ is a display decision; persisting the drop is data loss.

    The getters filter rows missing the fields their response model needs. Mutators used to read
    THROUGH those getters and write the filtered list back, so adding one favorite permanently
    deleted every row a different schema version had written.
    """
    _write_raw(tmp_path, name, json.dumps(seed))
    mutate(tmp_path)
    raw = json.loads(st._state_path(tmp_path, UID, name).read_text(encoding="utf-8"))
    assert any(survivor_key in row for row in raw), raw


def test_set_interests_is_locked_like_every_other_writer_of_this_file(tmp_path: Path) -> None:
    """An unlocked replace over a file that ALSO has read-modify-write writers is not
    last-write-wins.

    add_interest reads under the lock; a PUT /interests landing between that read and its write used
    to make add_interest persist a list derived from the PRE-PUT state, silently discarding the
    replacement. Asserted by observing the lock is held at the moment of the write, which is
    deterministic — unlike trying to schedule the actual interleaving.
    """
    from filelock import FileLock, Timeout

    held: list[bool] = []
    real_write = st._write

    def spy(data_dir: Path, user_id: str, name: str, obj: object) -> None:
        if name == "interests":
            # A SECOND lock object on the same file. flock is per file descriptor, so acquiring it
            # fails exactly when the writer holds the lock. `is_locked` would not do: that is
            # per-instance state, and a freshly minted instance always reports False.
            path = st._state_path(data_dir, user_id, name).with_name(f".{name}.lock")
            probe = FileLock(str(path))
            try:
                probe.acquire(timeout=0.01)
                probe.release()
                held.append(False)
            except Timeout:
                held.append(True)
        real_write(data_dir, user_id, name, obj)

    st._write = spy  # type: ignore[assignment]
    try:
        st.set_interests(tmp_path, UID, ["topic:ai"])
        st.add_interest(tmp_path, UID, "person:jane")
        st.remove_interest(tmp_path, UID, "topic:ai")
    finally:
        st._write = real_write  # type: ignore[assignment]

    assert held == [True, True, True], held
    # And the lock is genuinely released each time — a follow-up write must not time out.
    assert st.get_interests(tmp_path, UID) == ["person:jane"]


def test_resaving_a_favorite_keeps_its_place_and_its_original_added_at(tmp_path: Path) -> None:
    """ "Idempotent on kind+ref" has to mean the list is unchanged, not just un-duplicated.

    Re-saving used to remove the row and append the new one, so the item jumped to the end and got
    a fresh added_at (the route always stamps time.time()). The user saw their favorites reorder
    after an action that changed nothing, and RFC-103's weekly momentum series counted the re-save
    as a second engagement.
    """
    st.add_favorite(tmp_path, UID, {"kind": "episode", "ref": "a", "added_at": 100})
    st.add_favorite(tmp_path, UID, {"kind": "episode", "ref": "b", "added_at": 200})
    st.add_favorite(tmp_path, UID, {"kind": "episode", "ref": "c", "added_at": 300})

    favs = st.add_favorite(
        tmp_path, UID, {"kind": "episode", "ref": "a", "label": "renamed", "added_at": 999}
    )
    assert [f["ref"] for f in favs] == ["a", "b", "c"]  # position held
    first = next(f for f in favs if f["ref"] == "a")
    assert first["added_at"] == 100, "a re-save is the same save, not a new one"
    assert first["label"] == "renamed"  # everything else DOES update


def test_resubscribing_keeps_its_place_and_its_original_added_at(tmp_path: Path) -> None:
    st.add_subscription(tmp_path, UID, {"feed_id": "f1", "added_at": 100})
    st.add_subscription(tmp_path, UID, {"feed_id": "f2", "added_at": 200})
    library = st.add_subscription(
        tmp_path, UID, {"feed_id": "f1", "title": "Renamed", "added_at": 999}
    )
    assert [x["feed_id"] for x in library] == ["f1", "f2"]
    f1 = next(x for x in library if x["feed_id"] == "f1")
    assert f1["added_at"] == 100 and f1["title"] == "Renamed"
