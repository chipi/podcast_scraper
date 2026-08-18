"""Unit tests for the P2 Capture store — highlights, notes, re-anchor (#1114, RFC-098 §7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server import app_user_state as st

UID = "u_test"


def _span(over: dict | None = None) -> dict:
    """A transcript-span highlight with a time window + positional fields."""
    rec = {
        "id": "h1",
        "episode_slug": "show-ep01",
        "kind": "span",
        "start_ms": 10_000,
        "end_ms": 14_000,
        "char_start": 100,
        "char_end": 180,
        "segment_ids": ["s5", "s6"],
        "quote_text": "the stable anchor is the timestamp",
        "speaker": "Guest",
        "source_insight_id": None,
        "color": "amber",
        "created_at": 1000,
    }
    if over:
        rec.update(over)
    return rec


# --- highlights store ---------------------------------------------------------


def test_highlights_roundtrip_idempotent_and_scoped(tmp_path: Path) -> None:
    assert st.get_highlights(tmp_path, UID) == []
    st.add_highlight(tmp_path, UID, _span())
    st.add_highlight(
        tmp_path, UID, _span({"id": "h2", "episode_slug": "show-ep02", "kind": "moment"})
    )
    # idempotent on id — re-add replaces, no dup
    favs = st.add_highlight(tmp_path, UID, _span({"color": "blue"}))
    h1 = [h for h in favs if h["id"] == "h1"]
    assert len(h1) == 1 and h1[0]["color"] == "blue"
    # scoping by episode
    assert [h["id"] for h in st.get_highlights(tmp_path, UID, "show-ep01")] == ["h1"]
    assert [h["id"] for h in st.get_highlights(tmp_path, UID, "show-ep02")] == ["h2"]


def test_highlights_update_merges_and_protects_immutable_fields(tmp_path: Path) -> None:
    st.add_highlight(tmp_path, UID, _span())
    updated = st.update_highlight(
        tmp_path,
        UID,
        "h1",
        {
            "color": "rose",
            "quote_text": "edited",
            "episode_slug": "HACKED",
            "id": "HACKED",
            "created_at": 9,
        },
    )
    assert updated is not None
    assert updated["color"] == "rose" and updated["quote_text"] == "edited"
    # id / episode_slug / created_at are immutable — the attempted overwrite is ignored
    assert updated["id"] == "h1"
    assert updated["episode_slug"] == "show-ep01"
    assert updated["created_at"] == 1000
    # persisted
    assert st.get_highlights(tmp_path, UID)[0]["color"] == "rose"
    # no-op on an absent id
    assert st.update_highlight(tmp_path, UID, "nope", {"color": "x"}) is None


def test_highlights_remove(tmp_path: Path) -> None:
    st.add_highlight(tmp_path, UID, _span())
    st.add_highlight(tmp_path, UID, _span({"id": "h2"}))
    remaining = st.remove_highlight(tmp_path, UID, "h1")
    assert [h["id"] for h in remaining] == ["h2"]
    # removing an absent id is a no-op
    assert [h["id"] for h in st.remove_highlight(tmp_path, UID, "ghost")] == ["h2"]


def test_highlights_malformed_entries_filtered_on_read(tmp_path: Path) -> None:
    st._write(
        tmp_path, UID, "highlights", [{"id": "x"}, {"episode_slug": "e", "kind": "span"}, "bad"]
    )
    assert st.get_highlights(tmp_path, UID) == []


def test_highlights_non_list_payload_is_empty(tmp_path: Path) -> None:
    st._write(tmp_path, UID, "highlights", {"id": "h1"})
    assert st.get_highlights(tmp_path, UID) == []


# --- re-anchor (pure) ---------------------------------------------------------


# The PLAYER TRANSCRIPT CONTRACT — {id, start, end, text} with start/end in SECONDS
# (segments_view.to_contract_segments). The previous fixtures here used
# {segment_id, start_ms, end_ms, char_start, char_end}, a shape nothing in the codebase produces,
# so these tests validated the function against an imaginary contract while production never called
# it at all.
def _seg(sid: str, start_s: float, end_s: float, text: str) -> dict:
    return {"id": sid, "start": start_s, "end": end_s, "text": text}


def _segments() -> list[dict]:
    return [
        _seg("n1", 0.0, 5.0, "Opening chatter before the good part. "),
        _seg("n2", 5.0, 12.0, "the stable anchor is the timestamp"),
        _seg("n3", 12.0, 20.0, " and the rest of the discussion follows."),
    ]


def test_reanchor_span_finds_the_quote_and_recomputes_offsets(tmp_path: Path) -> None:
    """Time picks the candidates; the QUOTE decides whether they are the right ones."""
    out = st.reanchor_highlight(_span(), _segments())
    assert out["anchor_status"] == "anchored"
    assert out["segment_ids"] == ["n2", "n3"]  # window 10s-14s overlaps n2 and n3
    # Offsets are recomputed from where the quote actually is, in the client's coordinate
    # system (relative to the first candidate segment) — not widened to segment boundaries.
    joined = "the stable anchor is the timestamp and the rest of the discussion follows."
    assert joined[out["char_start"] : out["char_end"]] == "the stable anchor is the timestamp"
    # The anchor itself survives untouched.
    assert out["start_ms"] == 10_000 and out["end_ms"] == 14_000
    assert out["quote_text"] == "the stable anchor is the timestamp"


def test_reanchor_marks_drift_when_the_timeline_moved_under_the_quote(tmp_path: Path) -> None:
    """The window still exists but no longer contains the quote — the passage moved.

    This is the case time-only matching gets WRONG: it would stamp "anchored" on whatever text now
    happens to sit at those timestamps. On an adfree segment file (minutes shorter by its own
    docstring) that is a different passage entirely.
    """
    moved = [
        _seg("n1", 0.0, 5.0, "Completely different opening. "),
        _seg("n2", 5.0, 12.0, "An advert for a mattress company. "),
        _seg("n3", 12.0, 20.0, "And now something unrelated."),
    ]
    out = st.reanchor_highlight(_span(), moved)
    assert (
        out["anchor_status"] == "drifted"
    ), "the quote is absent from the new text, so this anchor was not earned"
    assert out["quote_text"] == "the stable anchor is the timestamp"  # never dropped


def test_reanchor_moment_without_a_quote_is_time_only(tmp_path: Path) -> None:
    """A bare moment has nothing to verify against — say so rather than claim verification."""
    moment = _span({"id": "m1", "kind": "moment", "start_ms": 6_000, "end_ms": None})
    moment.pop("quote_text", None)
    out = st.reanchor_highlight(moment, _segments())
    assert out["anchor_status"] == "time_only"
    assert out["segment_ids"] == ["n2"]  # 6s falls inside n2 only


def test_reanchor_is_idempotent(tmp_path: Path) -> None:
    once = st.reanchor_highlight(_span(), _segments())
    twice = st.reanchor_highlight(once, _segments())
    assert once == twice


def test_reanchor_recovers_a_drifted_highlight_when_the_text_returns(tmp_path: Path) -> None:
    gone = st.reanchor_highlight(_span(), [_seg("x", 0.0, 1.0, "nothing here")])
    assert gone["anchor_status"] == "drifted"
    back = st.reanchor_highlight(gone, _segments())
    assert back["anchor_status"] == "anchored"


def test_reanchor_handles_an_inverted_window(tmp_path: Path) -> None:
    weird = _span({"start_ms": 14_000, "end_ms": 10_000})
    out = st.reanchor_highlight(weird, _segments())
    assert out["anchor_status"] == "anchored"


def test_reanchor_drift_keeps_positional_fields_and_never_drops(tmp_path: Path) -> None:
    # a window past the end of the new (shortened) transcript → nothing overlaps
    drifted = _span({"start_ms": 90_000, "end_ms": 95_000})
    out = st.reanchor_highlight(drifted, _segments())
    assert out["anchor_status"] == "drifted"
    # prior positional fields are preserved (not zeroed); the highlight is returned, not dropped
    assert out["segment_ids"] == ["s5", "s6"]
    assert out["char_start"] == 100 and out["char_end"] == 180


def test_reanchor_insight_passes_through_unchanged(tmp_path: Path) -> None:
    insight = _span({"id": "i1", "kind": "insight", "source_insight_id": "show-ep01#gi-3"})
    out = st.reanchor_highlight(insight, _segments())
    # anchored by source_insight_id, not time → no anchor_status, fields untouched
    assert "anchor_status" not in out
    assert out["segment_ids"] == ["s5", "s6"]


def test_reanchor_missing_start_ms_is_drift(tmp_path: Path) -> None:
    out = st.reanchor_highlight(_span({"start_ms": None}), _segments())
    assert out["anchor_status"] == "drifted"


# --- notes store --------------------------------------------------------------


def _note(over: dict | None = None) -> dict:
    rec = {
        "id": "n1",
        "target": "highlight",
        "target_id": "h1",
        "text": "this reframed how I think about sleep",
        "created_at": 1000,
        "updated_at": 1000,
    }
    if over:
        rec.update(over)
    return rec


def test_notes_roundtrip_idempotent_and_scoped(tmp_path: Path) -> None:
    assert st.get_notes(tmp_path, UID) == []
    st.add_note(tmp_path, UID, _note())
    st.add_note(tmp_path, UID, _note({"id": "n2", "target": "episode", "target_id": "show-ep01"}))
    # idempotent on id
    notes = st.add_note(tmp_path, UID, _note({"text": "updated"}))
    n1 = [n for n in notes if n["id"] == "n1"]
    assert len(n1) == 1 and n1[0]["text"] == "updated"
    # scoping by target / target_id
    assert [n["id"] for n in st.get_notes(tmp_path, UID, target="episode")] == ["n2"]
    scoped = st.get_notes(tmp_path, UID, target="highlight", target_id="h1")
    assert [n["id"] for n in scoped] == ["n1"]


def test_notes_update_text_and_timestamp(tmp_path: Path) -> None:
    st.add_note(tmp_path, UID, _note())
    updated = st.update_note(tmp_path, UID, "n1", "second thoughts", 2000)
    assert updated is not None
    assert updated["text"] == "second thoughts" and updated["updated_at"] == 2000
    assert updated["created_at"] == 1000  # created stays put
    assert st.get_notes(tmp_path, UID)[0]["text"] == "second thoughts"
    assert st.update_note(tmp_path, UID, "absent", "x", 3000) is None


def test_notes_remove(tmp_path: Path) -> None:
    st.add_note(tmp_path, UID, _note())
    st.add_note(tmp_path, UID, _note({"id": "n2"}))
    assert [n["id"] for n in st.remove_note(tmp_path, UID, "n1")] == ["n2"]


def test_notes_malformed_and_non_list_payloads(tmp_path: Path) -> None:
    st._write(tmp_path, UID, "notes", [{"id": "x"}, {"target": "episode"}, 5])
    assert st.get_notes(tmp_path, UID) == []
    st._write(tmp_path, UID, "notes", json.dumps("nope"))  # a JSON string, not a list
    assert st.get_notes(tmp_path, UID) == []


# --- data-loss guards (found in review, 2026-08-16) --------------------------------------------
#
# Every mutator here persists what it just read. Two consequences the store used to have, both
# violating this module's own "NEVER drops a highlight" promise:
#
#   1. _read answered ANY read error with the empty default, so one unreadable highlights.json
#      plus one new capture replaced the user's entire history with a single row.
#   2. get_highlights filters rows missing a field the response model needs; mutators wrote that
#      filtered list back, so a row from another schema version was purged as a side effect of
#      changing an unrelated row.
#
# The invariant both tests defend: mutating row X must not rewrite any other row.


class TestMutationsNeverDestroyOtherRows:
    def test_corrupt_file_is_not_overwritten_by_a_new_capture(self, tmp_path) -> None:
        user_dir = tmp_path / "users" / "u1"
        user_dir.mkdir(parents=True)
        corrupt = user_dir / "highlights.json"
        corrupt.write_text('[{"id": "h_old", "episode_slug"', encoding="utf-8")  # truncated JSON

        with pytest.raises(st.UserStateUnreadable):
            st.add_highlight(
                tmp_path,
                "u1",
                {"id": "h_new", "episode_slug": "ep1", "kind": "moment", "created_at": 1},
            )
        # The damaged bytes are still there — a human can recover them.
        assert corrupt.read_text(encoding="utf-8").startswith('[{"id": "h_old"')

    def test_absent_file_still_reads_as_empty(self, tmp_path) -> None:
        """Absent is not the same as unreadable: a first capture must work."""
        got = st.add_highlight(
            tmp_path, "u1", {"id": "h1", "episode_slug": "ep1", "kind": "moment", "created_at": 1}
        )
        assert [h["id"] for h in got] == ["h1"]

    def test_a_row_the_reader_cannot_render_survives_an_unrelated_add(self, tmp_path) -> None:
        user_dir = tmp_path / "users" / "u1"
        user_dir.mkdir(parents=True)
        # Missing created_at → get_highlights filters it out, but it is still the user's data.
        (user_dir / "highlights.json").write_text(
            json.dumps([{"id": "h_legacy", "episode_slug": "ep1", "kind": "moment"}]),
            encoding="utf-8",
        )

        st.add_highlight(
            tmp_path,
            "u1",
            {"id": "h_new", "episode_slug": "ep2", "kind": "moment", "created_at": 9},
        )

        on_disk = json.loads((user_dir / "highlights.json").read_text(encoding="utf-8"))
        assert {r["id"] for r in on_disk} == {
            "h_legacy",
            "h_new",
        }, "adding a highlight deleted a row it merely could not render"
        # The API view still hides it, which is the correct display behaviour.
        assert [h["id"] for h in st.get_highlights(tmp_path, "u1")] == ["h_new"]

    def test_unrenderable_row_survives_remove_of_another(self, tmp_path) -> None:
        user_dir = tmp_path / "users" / "u1"
        user_dir.mkdir(parents=True)
        (user_dir / "highlights.json").write_text(
            json.dumps(
                [
                    {"id": "h_legacy", "episode_slug": "ep1", "kind": "moment"},
                    {"id": "h_ok", "episode_slug": "ep2", "kind": "moment", "created_at": 3},
                ]
            ),
            encoding="utf-8",
        )
        st.remove_highlight(tmp_path, "u1", "h_ok")
        on_disk = json.loads((user_dir / "highlights.json").read_text(encoding="utf-8"))
        assert [r["id"] for r in on_disk] == ["h_legacy"]

    def test_unrenderable_row_survives_update_of_another(self, tmp_path) -> None:
        user_dir = tmp_path / "users" / "u1"
        user_dir.mkdir(parents=True)
        (user_dir / "highlights.json").write_text(
            json.dumps(
                [
                    {"id": "h_legacy", "episode_slug": "ep1", "kind": "moment"},
                    {"id": "h_ok", "episode_slug": "ep2", "kind": "moment", "created_at": 3},
                ]
            ),
            encoding="utf-8",
        )
        st.update_highlight(tmp_path, "u1", "h_ok", {"color": "amber"})
        on_disk = json.loads((user_dir / "highlights.json").read_text(encoding="utf-8"))
        assert {r["id"] for r in on_disk} == {"h_legacy", "h_ok"}
        assert [r for r in on_disk if r["id"] == "h_ok"][0]["color"] == "amber"

    def test_notes_get_the_same_protection(self, tmp_path) -> None:
        user_dir = tmp_path / "users" / "u1"
        user_dir.mkdir(parents=True)
        (user_dir / "notes.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(st.UserStateUnreadable):
            st.add_note(
                tmp_path,
                "u1",
                {"id": "n1", "target": "highlight", "target_id": "h1", "text": "x"},
            )


def test_deleting_a_highlight_takes_its_notes_with_it(tmp_path: Path) -> None:
    """A note that LOOKS deleted and then comes back is worse than either outcome alone.

    Notes on a deleted highlight used to survive server-side while the client pruned them locally
    (capture.ts), so the user was shown the note was gone — and it reappeared on the next full load.
    The client's own filter was the intent the server had never implemented.
    """
    from podcast_scraper.server import app_user_state as st

    uid = "u_0123456789abcdef01234567"
    st.add_highlight(
        tmp_path, uid, {"id": "h1", "episode_slug": "ep", "kind": "span", "created_at": 1}
    )
    st.add_highlight(
        tmp_path, uid, {"id": "h2", "episode_slug": "ep", "kind": "span", "created_at": 2}
    )
    for nid, target_id in (("n1", "h1"), ("n2", "h1"), ("n3", "h2")):
        st.add_note(
            tmp_path,
            uid,
            {
                "id": nid,
                "target": "highlight",
                "target_id": target_id,
                "text": "t",
                "created_at": 1,
                "updated_at": 1,
            },
        )

    assert st.remove_notes_for_target(tmp_path, uid, "highlight", "h1") == 2
    remaining = st.get_notes(tmp_path, uid)
    assert [n["id"] for n in remaining] == ["n3"], "only h1's notes go"


def test_the_sweep_is_a_no_op_when_nothing_matches(tmp_path: Path) -> None:
    from podcast_scraper.server import app_user_state as st

    uid = "u_0123456789abcdef01234567"
    st.add_note(
        tmp_path,
        uid,
        {
            "id": "n1",
            "target": "episode",
            "target_id": "ep",
            "text": "t",
            "created_at": 1,
            "updated_at": 1,
        },
    )
    # Same id, different target kind — a highlight sweep must not take an episode note.
    assert st.remove_notes_for_target(tmp_path, uid, "highlight", "ep") == 0
    assert len(st.get_notes(tmp_path, uid)) == 1
