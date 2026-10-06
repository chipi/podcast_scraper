"""m0023: the text transcripts are re-rendered from the repaired segments, offsets carried across.

Prod shape (2026-10-06): m0012 removed "Andreessen Horowitz" from both host voices' segment labels,
and the transcripts still read "Andreessen Horowitz: …" — one line holding BOTH voices' turns, and
GI quotes pointing into it by character offset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0023_transcript_speaker_prefixes_resynced import (
    TranscriptSpeakerPrefixesResyncedMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

ORG = "Andreessen Horowitz"
GUEST = "Will Bryk"
TX = "transcripts/ep.txt"
ADFREE_REF = "transcripts/ep.adfree.txt"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(labelled: bool, both: bool = True) -> List[Dict[str, Any]]:
    """Host A, guest, host B; the hosts carry ORG while `labelled` (before the repair).

    ``both=False``: only host A ever carried it — host B was never named.
    """
    org = {"speaker_label": ORG} if labelled else {}
    org_b = org if both else {}
    return [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "Search is the gateway.",
            **org,
        },
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02", "text": "Welcome, Will.", **org_b},
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "SPEAKER_01",
            "speaker_label": GUEST,
            "text": "Thanks for having me.",
        },
        {
            "start": 3.0,
            "end": 4.0,
            "speaker": "SPEAKER_00",
            "text": "Google fails deep topics.",
            **org,
        },
    ]


def _corpus(root: Path, *, corrupt: bool = False, both: bool = True) -> Dict[str, Path]:
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
    # The transcripts as the pipeline wrote them BEFORE the repair: with the org label.
    old_txt, old_emitted = format_diarized_screenplay_with_offsets(_rows(labelled=True, both=both))
    if corrupt:
        old_txt = old_txt.replace("Welcome, Will.", "Welcome, Will. [music]")
    for key in ("txt", "atxt"):
        p[key].parent.mkdir(parents=True, exist_ok=True)
        p[key].write_text(old_txt, encoding="utf-8")
    p["ctxt"].write_text(f"{ORG}: Search is the gateway.\n{GUEST}: Thanks.\n", encoding="utf-8")
    # ...and the segments AFTER it: the label is gone, the stored offsets still the old ones.
    _w(p["seg"], _rows(labelled=False, both=both))
    _w(
        p["aseg"],
        [
            {**r, "char_start": e["char_start"], "char_end": e["char_end"]}
            for r, e in zip(_rows(labelled=False, both=both), old_emitted)
        ],
    )
    quote = "Google fails deep topics."
    start = old_txt.index(quote)
    _w(
        p["gi"],
        {
            "nodes": [
                {
                    "id": "quote:1",
                    "type": "Quote",
                    "properties": {
                        "text": quote,
                        "char_start": start,
                        "char_end": start + len(quote),
                        "transcript_ref": ADFREE_REF,
                    },
                }
            ],
            "edges": [],
        },
    )
    return p


def test_registered() -> None:
    assert "0023_transcript_speaker_prefixes_resynced" in [m.id for m in get_migrations()]


def test_transcripts_follow_the_segments_and_every_offset_follows(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    m = TranscriptSpeakerPrefixesResyncedMigration()
    assert not m.verify(MigrationContext(corpus_root=tmp_path))[0]

    m.apply(MigrationContext(corpus_root=tmp_path))

    atxt = p["atxt"].read_text(encoding="utf-8")
    assert ORG not in atxt and ORG not in p["txt"].read_text(encoding="utf-8")
    # The line two voices shared under one removed name is split, one per voice.
    assert atxt.splitlines()[:2] == [
        "SPEAKER_00: Search is the gateway.",
        "SPEAKER_02: Welcome, Will.",
    ]
    q = _r(p["gi"])["nodes"][0]["properties"]
    assert atxt[q["char_start"] : q["char_end"]] == q["text"]
    for row in _r(p["aseg"]):
        assert atxt[row["char_start"] : row["char_end"]] == row["text"]
    # .cleaned.txt holds no voice ids: one removed name on TWO voices cannot be split there,
    # so it is left and counted rather than guessed.
    assert p["ctxt"].read_text(encoding="utf-8").startswith(f"{ORG}: ")
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_cleaned_transcript_renamed_when_one_voice_held_the_name(tmp_path: Path) -> None:
    p = _corpus(tmp_path, both=False)
    TranscriptSpeakerPrefixesResyncedMigration().apply(MigrationContext(corpus_root=tmp_path))
    cleaned = p["ctxt"].read_text(encoding="utf-8")
    assert cleaned.startswith("SPEAKER_00: ") and f"{GUEST}: Thanks." in cleaned
    atxt = p["atxt"].read_text(encoding="utf-8")
    q = _r(p["gi"])["nodes"][0]["properties"]
    assert atxt[q["char_start"] : q["char_end"]] == q["text"]


def test_a_transcript_that_is_not_a_render_is_refused_and_untouched(tmp_path: Path) -> None:
    p = _corpus(tmp_path, corrupt=True)
    before = {k: v.read_bytes() for k, v in p.items()}
    result = TranscriptSpeakerPrefixesResyncedMigration().apply(
        MigrationContext(corpus_root=tmp_path)
    )
    assert result.details["totals"].get("refused") == 1
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_dry_run_writes_nothing_and_undo_restores(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    before = {k: v.read_bytes() for k, v in p.items()}
    m = TranscriptSpeakerPrefixesResyncedMigration()
    m.apply(MigrationContext(corpus_root=tmp_path, dry_run=True))
    assert {k: v.read_bytes() for k, v in p.items()} == before

    m.apply(MigrationContext(corpus_root=tmp_path))
    restored, refused = undo(tmp_path)
    assert restored == 4 and not refused  # txt, adfree.txt, adfree segments, gi
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_quote_across_the_split_line_refuses_the_episode(tmp_path: Path) -> None:
    # One quote over both host turns that shared the removed name's line. Splitting that line puts
    # a newline and a prefix inside the quote: it can no longer slice to its text, so nothing is
    # written rather than a quote pointing at the wrong characters.
    p = _corpus(tmp_path)
    gi = _r(p["gi"])
    old = p["atxt"].read_text(encoding="utf-8")
    span = "Search is the gateway. Welcome, Will."
    props = gi["nodes"][0]["properties"]
    props.update(text=span, char_start=old.index(span), char_end=old.index(span) + len(span))
    _w(p["gi"], gi)
    before = {k: v.read_bytes() for k, v in p.items()}
    result = TranscriptSpeakerPrefixesResyncedMigration().apply(
        MigrationContext(corpus_root=tmp_path)
    )
    assert result.details["totals"].get("refused: a quote would not slice to its text") == 1
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_an_absolute_transcript_ref_is_moved_too(tmp_path: Path) -> None:
    # Prod writes transcript_ref run-relative on some episodes and ABSOLUTE on others. Matched as a
    # run-relative string, the absolute ones were neither moved nor checked: 2,038 quotes pointed
    # at the wrong text until the write was undone (2026-10-06).
    p = _corpus(tmp_path)
    gi = _r(p["gi"])
    gi["nodes"][0]["properties"]["transcript_ref"] = str(p["atxt"])
    _w(p["gi"], gi)
    TranscriptSpeakerPrefixesResyncedMigration().apply(MigrationContext(corpus_root=tmp_path))
    q = _r(p["gi"])["nodes"][0]["properties"]
    atxt = p["atxt"].read_text(encoding="utf-8")
    assert atxt[q["char_start"] : q["char_end"]] == q["text"]
