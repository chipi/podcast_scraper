"""m0024: a removed speaker name 0023 had to refuse leaves the transcripts too (#2294).

Prod shape (2026-10-07, 17 episodes): the name sat on BOTH host voices, so the transcript line holds
both hosts' turns; a GI quote may span them; ``.cleaned.txt`` has no voice ids; and two quotes were
extracted up to a removed name's prefix and cut inside it (``"…deployed\\nAnd"``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0024_shared_removed_speaker_prefixes import (
    SharedRemovedSpeakerPrefixesMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

ORG = "Andreessen Horowitz"
GUEST = "Will Bryk"
TX = "transcripts/ep.txt"
ADFREE_REF = "transcripts/ep.adfree.txt"
# Not a pure render of the segments (a stray "[music]" line), as on the episodes 0023 refused, and
# one line holding BOTH hosts' turns under the removed name.
OLD = (
    f"{ORG}: Search is the gateway. Welcome, Will.\n"
    f"{GUEST}: Thanks for having me.\n"
    "[music]\n"
    f"{ORG}: Google fails deep topics.\n"
)


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _quote(i: int, text: str, start: int, end: int, ref: str = ADFREE_REF) -> Dict[str, Any]:
    return {
        "id": f"quote:{i}",
        "type": "Quote",
        "properties": {
            "text": text,
            "char_start": start,
            "char_end": end,
            "transcript_ref": ref,
        },
    }


def _corpus(root: Path, *, voices: int = 2, quotes: List[Dict[str, Any]] | None = None) -> Dict:
    run = root / "feeds" / "f" / "run_1"
    p = {
        "meta": run / "metadata" / "ep.metadata.json",
        "gi": run / "metadata" / "ep.gi.json",
        "seg": run / "transcripts" / "ep.segments.json",
        "aseg": run / "transcripts" / "ep.adfree.segments.json",
        "txt": run / "transcripts" / "ep.txt",
        "atxt": run / "transcripts" / "ep.adfree.txt",
        "ctxt": run / "transcripts" / "ep.cleaned.txt",
    }
    _w(p["meta"], {"episode": {"episode_id": "ep"}, "content": {"transcript_file_path": TX}})
    p["txt"].parent.mkdir(parents=True, exist_ok=True)
    for key in ("txt", "atxt"):
        p[key].write_text(OLD, encoding="utf-8")
    p["ctxt"].write_text(f"{ORG}: Search is the gateway.\n{GUEST}: Thanks.\n", encoding="utf-8")
    host_b = "SPEAKER_02" if voices == 2 else "SPEAKER_00"
    rows = [
        ("SPEAKER_00", "Search is the gateway."),
        (host_b, "Welcome, Will."),
        ("SPEAKER_01", "Thanks for having me."),
        ("SPEAKER_00", "Google fails deep topics."),
    ]
    segs = []
    for i, (voice, text) in enumerate(rows):
        row: Dict[str, Any] = {"start": float(i), "end": i + 1.0, "speaker": voice, "text": text}
        if voice == "SPEAKER_01":
            row["speaker_label"] = GUEST
        start = OLD.index(text)
        segs.append({**row, "char_start": start, "char_end": start + len(text)})
    _w(p["seg"], [{k: v for k, v in r.items() if not k.startswith("char_")} for r in segs])
    _w(p["aseg"], segs)
    named = ["SPEAKER_00", "SPEAKER_02"] if voices == 2 else ["SPEAKER_00"]
    _w(
        run / "transcripts" / "ep.speakers.diagnostics.json",
        {"voices": [{"voice": v, "resolved_name": ORG, "named": True} for v in named]},
    )
    if quotes is None:
        span = "Search is the gateway. Welcome, Will."  # both hosts' turns, one shared line
        last = "Google fails deep topics."
        quotes = [
            _quote(1, span, OLD.index(span), OLD.index(span) + len(span)),
            _quote(2, last, OLD.index(last), OLD.index(last) + len(last)),
        ]
    _w(p["gi"], {"nodes": quotes, "edges": []})
    return p


def _assert_offsets_hold(p: Dict[str, Path]) -> None:
    atxt = p["atxt"].read_text(encoding="utf-8")
    for node in _r(p["gi"])["nodes"]:
        q = node["properties"]
        assert atxt[q["char_start"] : q["char_end"]] == q["text"], q
    for row in _r(p["aseg"]):
        assert atxt[row["char_start"] : row["char_end"]] == row["text"], row


def test_registered_after_0023() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0024_shared_removed_speaker_prefixes") == (
        ids.index("0023_transcript_speaker_prefixes_resynced") + 1
    )


def test_a_name_on_two_voices_becomes_speaker_and_nothing_is_split(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    before = _r(p["gi"])["nodes"][0]["properties"]["text"]
    m = SharedRemovedSpeakerPrefixesMigration()
    assert not m.verify(MigrationContext(corpus_root=tmp_path))[0]

    m.apply(MigrationContext(corpus_root=tmp_path))

    for key in ("txt", "atxt", "ctxt"):
        text = p[key].read_text(encoding="utf-8")
        assert ORG not in text
        # Neither host's id: the line holds both hosts' turns and stays ONE line.
        assert text.startswith("SPEAKER: Search is the gateway.")
    assert p["atxt"].read_text(encoding="utf-8").splitlines()[0] == (
        "SPEAKER: Search is the gateway. Welcome, Will."
    )
    # The quote over both hosts' turns keeps its exact text — it gained no label.
    assert _r(p["gi"])["nodes"][0]["properties"]["text"] == before
    _assert_offsets_hold(p)
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_a_name_on_one_voice_becomes_that_voice(tmp_path: Path) -> None:
    p = _corpus(tmp_path, voices=1)
    SharedRemovedSpeakerPrefixesMigration().apply(MigrationContext(corpus_root=tmp_path))
    atxt = p["atxt"].read_text(encoding="utf-8")
    assert atxt.startswith("SPEAKER_00: ") and "\nSPEAKER: " not in atxt
    _assert_offsets_hold(p)


def test_a_quote_cut_inside_the_removed_label_is_trimmed(tmp_path: Path) -> None:
    # Extracted up to the next turn and cut inside its prefix: "…for having me.\n[music]\nAnd".
    start = OLD.index("Thanks for having me.")
    end = OLD.index(f"{ORG}: Google") + len("And")
    cut = _quote(1, OLD[start:end], start, end)
    p = _corpus(tmp_path, quotes=[cut])
    result = SharedRemovedSpeakerPrefixesMigration().apply(MigrationContext(corpus_root=tmp_path))
    assert result.details["totals"].get("quotes_trimmed") == 1
    q = _r(p["gi"])["nodes"][0]["properties"]
    assert q["text"] == "Thanks for having me.\n[music]"
    _assert_offsets_hold(p)


def test_a_quote_ending_in_ordinary_words_is_not_trimmed(tmp_path: Path) -> None:
    # "Google" is not the start of a removed name: nothing to trim, the quote only moves.
    start = OLD.index("Thanks for having me.")
    end = OLD.index("[music]") + len("[mus")
    p = _corpus(tmp_path, quotes=[_quote(1, OLD[start:end], start, end)])
    result = SharedRemovedSpeakerPrefixesMigration().apply(MigrationContext(corpus_root=tmp_path))
    assert not result.details["totals"].get("quotes_trimmed")
    assert _r(p["gi"])["nodes"][0]["properties"]["text"] == OLD[start:end]
    _assert_offsets_hold(p)


def test_an_unsafe_episode_is_refused_untouched_and_verify_still_fails(tmp_path: Path) -> None:
    # A quote ACROSS a renamed prefix whose stored text is not its slice: where its speech lands
    # cannot be checked, so nothing is written. (A quote the rename only shifts moves exactly.)
    span_start = OLD.index("[music]")
    span_end = OLD.index("Google fails deep topics.") + len("Google fails deep topics.")
    bad = _quote(1, "something else", span_start, span_end)
    p = _corpus(tmp_path, quotes=[bad])
    before = {k: v.read_bytes() for k, v in p.items()}
    m = SharedRemovedSpeakerPrefixesMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert result.details["totals"].get("refused") == 1
    assert result.details["refused"] == ["feeds/f/run_1/metadata/ep.metadata.json"]
    assert {k: v.read_bytes() for k, v in p.items()} == before
    # 0023's verify only looked at what it would write, so a refusal read as done. Not here.
    ok, msg = m.verify(MigrationContext(corpus_root=tmp_path))
    assert not ok and ORG in msg


def test_dry_run_writes_nothing_and_undo_restores(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    before = {k: v.read_bytes() for k, v in p.items()}
    m = SharedRemovedSpeakerPrefixesMigration()
    m.apply(MigrationContext(corpus_root=tmp_path, dry_run=True))
    assert {k: v.read_bytes() for k, v in p.items()} == before

    m.apply(MigrationContext(corpus_root=tmp_path))
    restored, refused = undo(tmp_path)
    assert restored == 5 and not refused  # txt, adfree.txt, cleaned.txt, adfree segments, gi
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_second_apply_finds_nothing(tmp_path: Path) -> None:
    _corpus(tmp_path)
    m = SharedRemovedSpeakerPrefixesMigration()
    m.apply(MigrationContext(corpus_root=tmp_path))
    again = m.apply(MigrationContext(corpus_root=tmp_path))
    assert again.details["files_written"] == 0 and not again.details["refused"]
