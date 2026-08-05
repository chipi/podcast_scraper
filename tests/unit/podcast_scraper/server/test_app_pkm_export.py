"""Unit tests for the graph-aware Obsidian emitter (RFC-113, #1472)."""

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
    monkeypatch.setattr(ex.app_corpus_revision, "current", lambda r, d, u: 7)


def _bundle(monkeypatch, highlights):
    monkeypatch.setattr(ex.app_user_state, "get_highlights", lambda d, u: highlights)
    return ex.build_obsidian_bundle(_ROOT, Path("/d"), _UID)


def test_safe_id_strips_separators() -> None:
    assert ex._safe_id("person:jensen-huang") == "person_jensen-huang"
    assert "/" not in ex._safe_id("../evil")


def test_highlight_note_wikilinks_and_deep_link(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _bundle(monkeypatch, [_HL])["files"]
    note = files["closelistening/Highlights/h_1.md"]
    assert "> the bottleneck was never compute" in note
    assert "[[closelistening/People/person_jensen-huang|Jensen Huang]]" in note
    assert "[[closelistening/Topics/topic_scaling|Scaling]]" in note
    assert "/player/acquired-nvidia?t=3921" in note  # deep-link with jump
    assert "source: user" in note


def test_entity_and_episode_notes_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _bundle(monkeypatch, [_HL])["files"]
    assert "closelistening/People/person_jensen-huang.md" in files
    assert "closelistening/Topics/topic_scaling.md" in files
    assert "closelistening/Episodes/acquired-nvidia.md" in files
    assert "# NVIDIA" in files["closelistening/Episodes/acquired-nvidia.md"]


def test_entity_notes_deduped_across_highlights(monkeypatch: pytest.MonkeyPatch) -> None:
    h2 = {**_HL, "id": "h_2", "quote_text": "another point"}  # same entity refs
    files = _bundle(monkeypatch, [_HL, h2])["files"]
    # two highlight notes, but ONE person note (deduped by id)
    assert "closelistening/Highlights/h_1.md" in files
    assert "closelistening/Highlights/h_2.md" in files
    people = [p for p in files if p.startswith("closelistening/People/")]
    assert people == ["closelistening/People/person_jensen-huang.md"]


def test_no_audio_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(monkeypatch, [_HL])
    blob = "\n".join(bundle["files"].values()).lower()
    assert ".mp3" not in blob and "audio" not in blob and "enclosure" not in blob


def test_manifest_carries_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _bundle(monkeypatch, [_HL])["manifest"]
    assert manifest["revision"] == 7
    assert manifest["namespace"] == "closelistening"
    assert manifest["removed"] == []
    assert "closelistening/Highlights/h_1.md" in manifest["written"]


def test_empty_when_no_highlights(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(monkeypatch, [])
    assert bundle["files"] == {}
    assert bundle["manifest"]["written"] == []
