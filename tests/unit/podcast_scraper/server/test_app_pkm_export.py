"""Unit tests for the graph-aware Obsidian emitter + incremental export (RFC-113, #1472)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml

from podcast_scraper.server import app_pkm_export as ex

pytestmark = pytest.mark.unit


def _recording_walk(calls: list):
    """Stand in for ``cached_catalog``, recording that it walked the catalog.

    The point of the test is that the walk happens ONCE, so the recording is the assertion —
    it should not be tucked inside an ``append(...) or []`` expression.
    """

    def _walk(root):
        calls.append("walk")
        return []

    return _walk


_UID = "u_0123456789abcdef01234567"
#: The real indexed catalog walk, grabbed before the autouse stub replaces it — the #42
#: counter test needs the genuine implementation, not the fixture's stand-in.
_REAL_TITLE_INDEX = ex._title_index
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


class _AnyTitle(dict):
    """A slug -> title index that answers the same title for every slug.

    The export used to call ``resolve_slug`` once per highlighted episode; since #42 it builds ONE
    indexed catalog walk instead, so the stub is now a mapping rather than a function. Answering
    any slug keeps each test asserting what it always asserted.
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def get(self, key, default=None):  # noqa: D102 - dict protocol
        return self._title


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ex, "_title_index", lambda root: _AnyTitle("NVIDIA"))


def _vault(monkeypatch, highlights) -> dict[str, str]:
    monkeypatch.setattr(ex.app_user_state, "get_highlights", lambda d, u: highlights)
    return cast("dict[str, str]", ex._current_vault(_ROOT, Path("/d"), _UID))


# --- note rendering (pure) ---


def test_safe_id_strips_separators() -> None:
    assert ex._safe_id("person:jensen-huang") == "person_jensen-huang"
    assert "/" not in ex._safe_id("../evil")


def test_yaml_scalar_escapes_quotes_and_newlines() -> None:
    assert ex._yaml_scalar('He said "hi"') == '"He said \\"hi\\""'
    assert "\n" not in ex._yaml_scalar("line1\nline2")


def test_frontmatter_survives_quotes_in_title_and_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "_title_index", lambda root: _AnyTitle('The "Real" Show'))
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
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"], epoch=b1["epoch"])
    assert b2["mode"] == "incremental"
    assert b2["replace_namespace"] is False  # incremental never wipes the namespace
    assert "closelistening/Highlights/h_2.md" in b2["files"]
    assert "closelistening/Highlights/h_1.md" not in b2["files"]  # unchanged → not re-sent


def test_incremental_tombstone_on_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    _hls(monkeypatch, [])  # highlight deleted → all its notes gone
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"], epoch=b1["epoch"])
    assert b2["mode"] == "incremental"
    assert "closelistening/Highlights/h_1.md" in b2["removed"]  # tombstone


def test_unchanged_reexport_no_cursor_bump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    b2 = ex.export_bundle(
        _ROOT, tmp_path, _UID, since=b1["revision"], epoch=b1["epoch"]
    )  # nothing changed
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
    monkeypatch.setattr(ex, "_title_index", lambda root: _AnyTitle("Ep ]] two|three"))
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
    monkeypatch.setattr(
        ex,
        "_title_index",
        lambda root: _AnyTitle('- Title: with "colon"\tand \x07 bell'),
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


# --- a bare cursor cannot identify a snapshot across a state reset (#41) --------------------------
#
# `since == prev_cursor` treated an integer as identifying a snapshot. But the counter restarts at
# 0 whenever export_state.json is lost — and _load_state ALSO resets on corruption, so this is not
# limited to restore-from-backup. Sequence: state resets; exports resume and the counter climbs
# back to N; a device still holding the pre-reset N (whose vault reflects the OLD world) asks for
# since=N; the server sees a match and computes a delta against ITS OWN snapshot, not against what
# the client holds. The client applies a nonsense delta, keeps its orphans, and because the cursors
# now advance in lockstep it NEVER receives a full export again. Silent and permanent.


def test_a_cursor_from_before_a_state_reset_gets_a_full_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline scenario. Same integer, different world."""
    _hls(monkeypatch, [_HL])
    before = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)  # rev 1, epoch A

    # The server loses its export state (restore from backup, wiped volume, fresh container).
    ex._state_path(tmp_path, _UID).unlink()

    # A different world: the user captured more while the client was away.
    h2 = {**_HL, "id": "h_2", "episode_slug": "ep2", "graph_refs": []}
    _hls(monkeypatch, [_HL, h2])
    ex.export_bundle(_ROOT, tmp_path, _UID, since=0)  # counter climbs back to 1, epoch B

    # The stale device echoes its pre-reset cursor — which now MATCHES the new counter.
    stale = ex.export_bundle(_ROOT, tmp_path, _UID, since=before["revision"], epoch=before["epoch"])
    assert stale["mode"] == "full", stale["mode"]
    assert stale["replace_namespace"] is True
    assert "closelistening/Highlights/h_1.md" in stale["files"]


def test_a_corrupt_state_file_also_re_mints_the_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_state resets on OSError/ValueError too, so corruption is the same failure as loss."""
    _hls(monkeypatch, [_HL])
    before = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    ex._state_path(tmp_path, _UID).write_text("{not json", encoding="utf-8")

    after = ex.export_bundle(_ROOT, tmp_path, _UID, since=before["revision"], epoch=before["epoch"])
    assert after["mode"] == "full"
    assert after["epoch"] != before["epoch"], "a reset must mint a new vault identity"


def test_a_client_that_sends_no_epoch_gets_a_full_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older client echoes only the revision. Full is the always-valid answer, so it self-heals
    on its first export instead of silently applying deltas it cannot verify."""
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"])  # no epoch
    assert b2["mode"] == "full"


def test_a_cursor_from_the_future_gets_a_full_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Correct before this change, untested. Pinned so the epoch work cannot regress it."""
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    ahead = ex.export_bundle(_ROOT, tmp_path, _UID, since=999, epoch=b1["epoch"])
    assert ahead["mode"] == "full"
    assert ahead["replace_namespace"] is True


def test_the_epoch_is_stable_while_the_state_file_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It identifies the VAULT, not the export. Re-minting per call would mean never incremental."""
    _hls(monkeypatch, [_HL])
    first = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    h2 = {**_HL, "id": "h_2", "episode_slug": "ep2", "graph_refs": []}
    _hls(monkeypatch, [_HL, h2])
    second = ex.export_bundle(_ROOT, tmp_path, _UID, since=first["revision"], epoch=first["epoch"])
    assert second["epoch"] == first["epoch"]
    assert second["mode"] == "incremental"


# --- one catalog walk per export, not one per episode (#42) ---------------------------------------


def test_the_catalog_is_walked_once_per_export_not_once_per_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resolve_slug` documented itself as O(episodes) per CALL — but each call rebuilt the whole
    catalog, walking every run's metadata and JSON-parsing all of it, uncached.

    Highlights across 300 episodes of a 1000-episode corpus meant ~300 full walks and ~300k JSON
    loads, all INSIDE the export lock, whose timeout is 5s. A concurrent export — web plus native
    shell, the very case that lock exists for — then raised filelock.Timeout and 500'd.
    """
    calls: list[str] = []

    def counting_walk(root: Path):
        calls.append("walk")
        return [SimpleNamespace(slug=f"slug-{i}", episode_title=f"Ep {i}") for i in range(10)]

    monkeypatch.setattr(ex, "_title_index", _REAL_TITLE_INDEX)  # the real one, not the stub
    monkeypatch.setattr(ex, "cached_catalog", counting_walk)
    monkeypatch.setattr(ex, "slug_for_row", lambda row: row.slug)

    # 50 highlights spread over 10 distinct episodes: 10 walks before, 1 after.
    _hls(
        monkeypatch,
        [
            {**_HL, "id": f"h_{i}", "episode_slug": f"slug-{i % 10}", "graph_refs": []}
            for i in range(50)
        ],
    )
    bundle = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)

    assert len(calls) == 1, f"the catalog was walked {len(calls)} times for 10 episodes"
    # And the titles still landed, so this measures a hoist rather than a lost lookup.
    assert "closelistening/Episodes/slug-0.md" in bundle["files"]
    assert "Ep 0" in bundle["files"]["closelistening/Episodes/slug-0.md"]


def test_no_highlights_costs_no_catalog_walk_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index is built on first need. Hoisting it to the top of the function would have made an
    empty vault pay for a full corpus scan — a regression for every user who has captured nothing.
    """
    calls: list[str] = []
    monkeypatch.setattr(ex, "_title_index", _REAL_TITLE_INDEX)
    monkeypatch.setattr(ex, "cached_catalog", _recording_walk(calls))
    _hls(monkeypatch, [])
    ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    assert calls == []


# --- tombstone coverage, refs and filenames (#44) ------------------------------------------------
#
# The tombstone CORE is sound and worth stating as such: there is no reference counting to get
# wrong. Every export rebuilds the whole vault from current highlights and set-diffs content hashes
# against the snapshot, so edits, shared entities and label renames on stable ids all fall out
# correctly. These pin the cases that were never tested.


def test_deleting_one_of_two_highlights_keeps_their_shared_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case reference counting would get wrong — and which the set-diff gets right for free."""
    shared = [{"id": "topic:ai", "kind": "topic", "label": "AI"}]
    a = {**_HL, "id": "h_a", "graph_refs": shared}
    b = {**_HL, "id": "h_b", "graph_refs": shared}
    _hls(monkeypatch, [a, b])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)

    _hls(monkeypatch, [a])  # delete ONE
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"], epoch=b1["epoch"])
    assert "closelistening/Highlights/h_b.md" in b2["removed"]
    assert (
        "closelistening/Topics/topic_ai.md" not in b2["removed"]
    ), "the surviving highlight still links this entity — tombstoning it would break that link"


def test_an_edit_ships_exactly_one_changed_path_and_no_tombstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hls(monkeypatch, [_HL])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    _hls(monkeypatch, [{**_HL, "quote_text": "an edited quote"}])
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"], epoch=b1["epoch"])
    assert list(b2["files"]) == ["closelistening/Highlights/h_1.md"]
    assert b2["removed"] == []


def test_a_label_rename_on_a_stable_id_changes_the_note_but_never_tombstones_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why filenames are ids, not labels: the note survives the rename in place."""
    _hls(monkeypatch, [{**_HL, "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}]}])
    b1 = ex.export_bundle(_ROOT, tmp_path, _UID, since=0)
    _hls(
        monkeypatch,
        [{**_HL, "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "Machine Learning"}]}],
    )
    b2 = ex.export_bundle(_ROOT, tmp_path, _UID, since=b1["revision"], epoch=b1["epoch"])
    assert "closelistening/Topics/topic_ai.md" in b2["files"]
    assert b2["removed"] == []
    assert "Machine Learning" in b2["files"]["closelistening/Topics/topic_ai.md"]


def test_entity_ids_differing_only_in_case_share_one_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS and Windows vaults are case-insensitive by default (#44).

    `person:Sam` and `person:sam` used to be two zip entries that collide on extraction — and a
    tombstone for one would delete the file the other still needs. The KG lowercases ids at mint,
    so this is speculative for pipeline artifacts, but refs are frozen onto the highlight at
    capture and the export trusts them verbatim, so nothing structural enforced it.
    """
    vault = _vault(
        monkeypatch,
        [
            {
                **_HL,
                "id": "h_a",
                "graph_refs": [{"id": "person:Sam", "kind": "person", "label": "S"}],
            },
            {
                **_HL,
                "id": "h_b",
                "graph_refs": [{"id": "person:sam", "kind": "person", "label": "S"}],
            },
        ],
    )
    entity_paths = [p for p in vault if "/People/" in p]
    assert entity_paths == ["closelistening/People/person_sam.md"], entity_paths
    # The STEM is folded; the directory stays "People" on purpose, so this checks the filename
    # rather than the whole path (the first version asserted the latter and failed on the dir).
    stems = [path.rsplit("/", 1)[-1] for path in entity_paths]
    assert all(s == s.lower() for s in stems), stems


def test_refs_frozen_empty_at_capture_are_backfilled_on_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moment captured while its episode had no KG exported ZERO entity links, for ever — even
    after the KG landed. That is the one case where "the vault mirrors your graph" was untrue."""
    monkeypatch.setattr(
        ex.app_graph_refs,
        "refs_for_slug",
        lambda root, slug, **kw: [{"id": "topic:late", "kind": "topic", "label": "Late"}],
    )
    vault = _vault(monkeypatch, [{**_HL, "graph_refs": []}])
    assert "closelistening/Topics/topic_late.md" in vault
    assert "topic_late" in vault["closelistening/Highlights/h_1.md"]


def test_a_highlight_that_captured_refs_keeps_exactly_those(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backfill fills a GAP; it does not re-resolve. Otherwise a later KG rewrite would silently
    restate what an old capture was about."""
    monkeypatch.setattr(
        ex.app_graph_refs,
        "refs_for_slug",
        lambda root, slug, **kw: [{"id": "topic:rewritten", "kind": "topic", "label": "New"}],
    )
    vault = _vault(monkeypatch, [_HL])  # _HL already carries jensen-huang + scaling
    assert "closelistening/Topics/topic_rewritten.md" not in vault
    assert "closelistening/Topics/topic_scaling.md" in vault


class TestPlaceholderPeopleNeverReachTheVault:
    """An unresolved person is an episode-local label, not somebody a reader can look up (#1685).

    Two shapes: `person:speaker-{ep}-03` (a diarization voice, #1b) and
    `person:unresolved-{name}-{ep}` (a bare first name with no surname anywhere in the episode).
    Every in-app surface already drops these via `is_unresolved_speaker_placeholder` — the check
    lives inside `entities_from_kg` itself — but the export did not, so a vault grew a
    `People/person_speaker-...md` note per anonymous voice plus wikilinks pointing at it.

    The `speaker-NN` half was a live bug before #1685 existed. `graph_refs` are FROZEN onto a
    highlight at capture and the export deliberately trusts them, so highlights captured earlier
    already carry placeholder refs — filtering at the export boundary repairs those too, which
    matters because a vault cannot be migrated once the user has downloaded it.
    """

    @staticmethod
    def _refs():
        return [
            {"id": "person:elon-musk", "kind": "person", "label": "Elon Musk"},
            {"id": "person:speaker-ep-1-03", "kind": "person", "label": "SPEAKER_03"},
            {"id": "person:unresolved-jensen-ep-1", "kind": "person", "label": "Jensen"},
            {"id": "topic:ai-safety", "kind": "topic", "label": "AI safety"},
        ]

    def test_only_the_real_entities_survive(self) -> None:
        kept = [r["id"] for r in ex._usable_refs(self._refs())]
        assert kept == ["person:elon-musk", "topic:ai-safety"]

    def test_a_highlight_note_links_no_placeholder(self) -> None:
        note = ex._highlight_note(
            {"graph_refs": self._refs(), "episode_slug": "s", "quote_text": "q"}, "Ep Title"
        )
        assert "person_speaker-ep-1-03" not in note
        assert "person_unresolved-jensen-ep-1" not in note
        assert "person_elon-musk" in note, "a real person must still be linked"

    def test_malformed_refs_are_dropped_without_raising(self) -> None:
        assert ex._usable_refs([{"kind": "person"}, "nonsense", None, {"id": ""}]) == []

    def test_none_is_safe(self) -> None:
        assert ex._usable_refs(None) == []
