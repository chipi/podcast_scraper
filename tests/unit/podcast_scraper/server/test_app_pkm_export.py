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
    # Quoted since #43 — YAML-equivalent, and consistent with every other string field.
    assert 'source: "user"' in note


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


# --- the LINK BODY was never escaped, only the frontmatter (#43) ----------------------------------
#
# _yaml_scalar is applied carefully everywhere a title/quote/label enters frontmatter. The wikilink
# display text after `|` was emitted raw — and `]]` ends the link while `|` starts a new field, so
# either one truncates the link mid-note and spills the remainder as literal text. Podcast titles
# really do contain brackets. This is a vault the user opens in Obsidian, where a broken note is
# not an error message, just a note that quietly does not work.


def test_a_label_containing_link_syntax_cannot_truncate_the_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hl = {
        **_HL,
        "graph_refs": [{"id": "topic:x", "kind": "topic", "label": "Weird ]]title|x"}],
    }
    note = _vault(monkeypatch, [hl])["closelistening/Highlights/h_1.md"]
    line = next(ln for ln in note.splitlines() if ln.startswith("Discusses "))
    body = line[len("Discusses [[") : line.rindex("]]")]
    assert "]]" not in body, line
    assert "|" not in body.split("|", 1)[1] if "|" in body else True
    # The link still resolves to the right note, and the label is still readable.
    assert "closelistening/Topics/topic_x|" in line
    assert "Weird" in line and "title" in line


def test_an_episode_title_containing_link_syntax_cannot_truncate_the_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ex, "resolve_slug", lambda root, slug: SimpleNamespace(episode_title="Ep ]] two|three")
    )
    note = _vault(monkeypatch, [{**_HL, "graph_refs": []}])["closelistening/Highlights/h_1.md"]
    line = next(ln for ln in note.splitlines() if ln.startswith("— [["))
    # Count over the WHOLE line. The first version sliced to the first "]]" and asserted the
    # fragment was well-formed — but under a raw title that first "]]" IS the injected one, so it
    # measured the truncation and pronounced it healthy. Sabotage caught it: reverting the fix left
    # this test green. One wikilink on this line means exactly one "[[" and one "]]".
    assert line.count("[[") == 1, line
    assert line.count("]]") == 1, line
    assert "closelistening/Episodes/acquired-nvidia|" in line
    assert "[▶ jump](/player/acquired-nvidia" in line  # the rest of the line survived intact


def test_a_multi_line_quote_stays_inside_the_blockquote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown ends a blockquote at the first unprefixed line.

    A two-line capture rendered its opening line as a quote and the rest as body text attributed to
    nobody — the note said something the speaker did not.
    """
    hl = {**_HL, "quote_text": "first line\nsecond line", "graph_refs": []}
    note = _vault(monkeypatch, [hl])["closelistening/Highlights/h_1.md"]
    body = note.split("---\n", 2)[2]
    quoted = [ln for ln in body.splitlines() if ln.startswith("> ")]
    assert len(quoted) == 2, body
    assert "second line" in quoted[1]


def test_every_note_in_the_vault_has_parseable_frontmatter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check the string asserts above could never make: run a YAML parser over ALL of it.

    Asserting `'\\\\"' in alias_line` proves an escape was emitted, not that the result parses.
    Includes the shapes that break YAML rather than merely look odd — a ": " (which turns a scalar
    into a mapping if unquoted), a leading "-", a tab, and a raw \\x07 control character, which is
    illegal inside a double-quoted scalar and made exactly one note unopenable.
    """
    yaml = pytest.importorskip("yaml")
    monkeypatch.setattr(
        ex,
        "resolve_slug",
        lambda root, slug: SimpleNamespace(episode_title='- Title: with "colon"\tand \x07 bell'),
    )
    hl = {
        **_HL,
        "quote_text": "- leading dash: and a colon\ttab \x07 bell \\ backslash",
        "graph_refs": [
            {"id": "person:a", "kind": "person", "label": '- Name: "quoted"\x01'},
            {"id": "topic:b", "kind": "topic", "label": "]]weird|label"},
        ],
    }
    vault = _vault(monkeypatch, [hl])
    assert vault, "no notes were produced, so this test asserts nothing"
    for path, text in vault.items():
        assert text.startswith("---\n"), path
        frontmatter = text.split("---\n", 2)[1]
        parsed = yaml.safe_load(frontmatter)  # raises → the user's vault has a broken note
        assert isinstance(parsed, dict), (path, parsed)
