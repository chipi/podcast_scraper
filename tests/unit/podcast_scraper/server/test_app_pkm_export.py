"""Unit tests for the graph-aware Obsidian emitter + incremental export (RFC-113, #1472)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server import app_pkm_export as ex

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"
_ROOT = Path("/unused")

_HL = {
    "id": "h_1",
    "episode_slug": "acquired-nvidia",
    "kind": "span",
    "start_ms": 3921000,
    "quote_text": "the bottleneck was never compute",
    "source": "user",
    "graph_refs": [
        {"id": "person:jensen-huang", "kind": "person", "label": "Jensen Huang"},
        {"id": "topic:scaling", "kind": "topic", "label": "Scaling"},
    ],
}


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ex, "resolve_slug", lambda root, slug: SimpleNamespace(episode_title="NVIDIA")
    )


def _vault(monkeypatch, highlights) -> dict[str, str]:
    monkeypatch.setattr(ex.app_user_state, "get_highlights", lambda d, u: highlights)
    return ex._current_vault(_ROOT, Path("/d"), _UID)


# --- note rendering (pure) ---


def test_safe_id_strips_separators() -> None:
    assert ex._safe_id("person:jensen-huang") == "person_jensen-huang"
    assert "/" not in ex._safe_id("../evil")


def test_yaml_scalar_escapes_quotes_and_newlines() -> None:
    assert ex._yaml_scalar('He said "hi"') == '"He said \\"hi\\""'
    assert "\n" not in ex._yaml_scalar("line1\nline2")


def test_frontmatter_survives_quotes_in_title_and_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ex, "resolve_slug", lambda root, slug: SimpleNamespace(episode_title='The "Real" Show')
    )
    hl = {**_HL, "quote_text": 'a quote with "quotes" and\na newline', "graph_refs": []}
    note = _vault(monkeypatch, [hl])["closelistening/Highlights/h_1.md"]
    # The alias line must be valid single-line double-quoted YAML (escaped quotes, no newline).
    alias_line = next(ln for ln in note.splitlines() if ln.startswith("aliases:"))
    assert '\\"' in alias_line and "\n" not in alias_line


def test_traversal_shaped_ids_cannot_escape_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    hl = {**_HL, "id": "../../etc/passwd", "episode_slug": "../../../x", "graph_refs": []}
    paths = list(_vault(monkeypatch, [hl]).keys())
    assert all(p.startswith("closelistening/") and ".." not in p for p in paths)


def test_highlight_note_wikilinks_and_deep_link(monkeypatch: pytest.MonkeyPatch) -> None:
    note = _vault(monkeypatch, [_HL])["closelistening/Highlights/h_1.md"]
    assert "> the bottleneck was never compute" in note
    assert "[[closelistening/People/person_jensen-huang|Jensen Huang]]" in note
    assert "[[closelistening/Topics/topic_scaling|Scaling]]" in note
    assert "/player/acquired-nvidia?t=3921" in note  # deep-link with jump
    assert "source: user" in note


def test_entity_and_episode_notes_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _vault(monkeypatch, [_HL])
    assert "closelistening/People/person_jensen-huang.md" in files
    assert "closelistening/Topics/topic_scaling.md" in files
    assert "closelistening/Episodes/acquired-nvidia.md" in files
    assert "# NVIDIA" in files["closelistening/Episodes/acquired-nvidia.md"]


def test_entity_notes_deduped_across_highlights(monkeypatch: pytest.MonkeyPatch) -> None:
    h2 = {**_HL, "id": "h_2", "quote_text": "another point"}
    files = _vault(monkeypatch, [_HL, h2])
    assert "closelistening/Highlights/h_1.md" in files
    assert "closelistening/Highlights/h_2.md" in files
    people = [p for p in files if p.startswith("closelistening/People/")]
    assert people == ["closelistening/People/person_jensen-huang.md"]


def test_no_audio_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = "\n".join(_vault(monkeypatch, [_HL]).values()).lower()
    assert ".mp3" not in blob and "audio" not in blob and "enclosure" not in blob


# --- incremental export (stateful, real tmp state dir) ---


def _hls(monkeypatch, highlights) -> None:
    monkeypatch.setattr(ex.app_user_state, "get_highlights", lambda d, u: highlights)


def test_first_export_is_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    b = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    assert b["mode"] == "full"
    assert b["revision"] == 1
    assert "closelistening/Highlights/h_1.md" in b["files"]
    assert b["removed"] == []
    # Full mode tells the client to replace the whole namespace (it can't be sent per-file
    # tombstones for notes it never saw) — review M8.
    assert b["replace_namespace"] is True


def test_incremental_only_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)  # full, rev 1
    # add a second highlight (new episode) → next export returns ONLY the new files
    h2 = {**_HL, "id": "h_2", "episode_slug": "dwarkesh-ep", "graph_refs": []}
    _hls(monkeypatch, [_HL, h2])
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"])
    assert b2["mode"] == "incremental"
    assert b2["replace_namespace"] is False  # incremental never wipes the namespace
    assert "closelistening/Highlights/h_2.md" in b2["files"]
    assert "closelistening/Highlights/h_1.md" not in b2["files"]  # unchanged → not re-sent


def test_incremental_tombstone_on_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    _hls(monkeypatch, [])  # highlight deleted → all its notes gone
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"])
    assert b2["mode"] == "incremental"
    assert "closelistening/Highlights/h_1.md" in b2["removed"]  # tombstone


def test_unchanged_reexport_no_cursor_bump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"])  # nothing changed
    assert b2["revision"] == b1["revision"]  # cursor did not advance
    assert b2["files"] == {} and b2["removed"] == []


def test_stale_cursor_falls_back_to_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    ex.export_bundle(_ROOT, tmp_path, _UID, since=0)  # rev 1
    h2 = {**_HL, "id": "h_2", "episode_slug": "ep2", "graph_refs": []}
    _hls(monkeypatch, [_HL, h2])
    # a client stuck at since=0 (behind) gets a FULL export, not a partial delta
    b = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    assert b["mode"] == "full"
    assert "closelistening/Highlights/h_1.md" in b["files"]
