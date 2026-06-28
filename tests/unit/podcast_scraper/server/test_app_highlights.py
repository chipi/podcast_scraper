"""Unit tests for the P2 Capture store — highlights, notes, re-anchor (#1114, RFC-098 §7)."""

from __future__ import annotations

import json
from pathlib import Path

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


def _seg(sid: str, start_ms: int, end_ms: int, char_start: int, char_end: int) -> dict:
    return {
        "segment_id": sid,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "char_start": char_start,
        "char_end": char_end,
    }


def _segments() -> list[dict]:
    return [
        _seg("n1", 0, 5_000, 0, 40),
        _seg("n2", 5_000, 12_000, 40, 95),
        _seg("n3", 12_000, 20_000, 95, 150),
    ]


def test_reanchor_span_recomputes_positional_fields_from_overlap(tmp_path: Path) -> None:
    # span window 10_000–14_000 overlaps n2 (5k–12k) and n3 (12k–20k)
    out = st.reanchor_highlight(_span(), _segments())
    assert out["anchor_status"] == "anchored"
    assert out["segment_ids"] == ["n2", "n3"]
    assert out["char_start"] == 40  # min of overlapping
    assert out["char_end"] == 150  # max of overlapping
    # the input is not mutated; timestamps + quote survive as the anchor
    assert out["start_ms"] == 10_000 and out["end_ms"] == 14_000
    assert out["quote_text"] == "the stable anchor is the timestamp"


def test_reanchor_moment_point_overlap(tmp_path: Path) -> None:
    moment = _span({"id": "m1", "kind": "moment", "start_ms": 6_000, "end_ms": None})
    out = st.reanchor_highlight(moment, _segments())
    assert out["anchor_status"] == "anchored"
    assert out["segment_ids"] == ["n2"]  # 6_000 falls inside n2 only


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
