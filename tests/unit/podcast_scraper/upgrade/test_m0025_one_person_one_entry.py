"""m0025: one person is one entry on an episode, whatever their title or spelling.

Synthetic fixtures shaped like the prod cases of 2026-10-08: a host's own "I'm Professor Hannah
Fry" beside the feed's "Hannah Fry"; Odd Lots' "Traci Alloway" cast as a guest beside the feed's
host Tracy Alloway; "Bernard Leong" stated and "Bernard Leung" hinted, both unplaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import (
    OnePersonOneEntryMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations
from podcast_scraper.workflow.turns_artifact import build_turns_document

pytestmark = [pytest.mark.unit]

TX = "transcripts/ep.txt"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pid(name: str) -> str:
    return "person:" + "-".join(name.lower().replace(".", "").split())


def _corpus(
    root: Path,
    *,
    speakers: List[Dict[str, Any]],
    turns: Sequence[Tuple[str, str, str, str]],
    known_hosts: Sequence[str],
    with_turns: bool = True,
) -> Dict[str, Path]:
    """One episode. ``turns`` = (voice, label, role, text); the transcript is their exact render."""
    run = root / "feeds" / "f" / "run_1"
    p = {
        "meta": run / "metadata" / "ep.metadata.json",
        "gi": run / "metadata" / "ep.gi.json",
        "kg": run / "metadata" / "ep.kg.json",
        "seg": run / "transcripts" / "ep.segments.json",
        "txt": run / "transcripts" / "ep.txt",
        "diag": run / "transcripts" / "ep.speakers.diagnostics.json",
        "turns": run / "transcripts" / "ep.turns.json",
    }
    _w(
        p["meta"],
        {
            "episode": {"episode_id": "ep"},
            "content": {"transcript_file_path": TX, "speakers": speakers},
        },
    )
    rows = [
        {
            "start": float(i),
            "end": i + 1.0,
            "speaker": v,
            "speaker_label": lab,
            "speaker_role": role,
            "text": t,
        }
        for i, (v, lab, role, t) in enumerate(turns)
    ]
    _w(p["seg"], rows)
    text, offsets = format_diarized_screenplay_with_offsets(rows)
    p["txt"].write_text(text, encoding="utf-8")
    labels = sorted({lab for _v, lab, _r, _t in turns})
    persons = [
        {"id": _pid(lab), "type": "Person", "properties": {"name": lab, "label": lab, "role": role}}
        for lab, role in {lab: role for _v, lab, role, _t in turns}.items()
    ]
    quotes = [
        {
            "id": f"quote:{i}",
            "type": "Quote",
            "properties": {
                "text": o["text"],
                "char_start": o["char_start"],
                "char_end": o["char_end"],
                "transcript_ref": TX,
                "speaker_id": _pid(o["speaker_label"]),
                "speaker_name": o["speaker_label"],
            },
        }
        for i, o in enumerate(offsets)
    ]
    spoken = [
        {"type": "SPOKEN_BY", "from": f"quote:{i}", "to": _pid(o["speaker_label"])}
        for i, o in enumerate(offsets)
    ]
    _w(p["gi"], {"nodes": persons + quotes, "edges": spoken})
    _w(p["kg"], {"nodes": [json.loads(json.dumps(n)) for n in persons], "edges": []})
    _w(
        p["diag"],
        {
            "tried": {"known_hosts": list(known_hosts)},
            "voices": [
                {"voice": v, "resolved_name": lab, "named": True, "role": role}
                for v, lab, role in {(v, lab, role) for v, lab, role, _t in turns}
            ],
        },
    )
    doc = build_turns_document(
        text, rows, rel_transcript_path=TX, episode_slug="ep", language="en", segments_sha256="x"
    )
    assert doc is not None and labels
    if with_turns:
        p["turns"].write_text(json.dumps(doc, indent=0), encoding="utf-8")
    else:
        del p["turns"]
    return p


def _apply(root: Path, dry_run: bool = False):
    return OnePersonOneEntryMigration().apply(MigrationContext(corpus_root=root, dry_run=dry_run))


def _quotes_hold(p: Dict[str, Path]) -> None:
    text = p["txt"].read_text(encoding="utf-8")
    for node in _r(p["gi"])["nodes"]:
        q = node["properties"]
        if node["type"] == "Quote":
            assert text[q["char_start"] : q["char_end"]] == q["text"], q


def _speaker(name: str, role: str, voices: List[str], source: str, sid: str) -> Dict[str, Any]:
    return {
        "id": sid,
        "name": name,
        "role": role,
        "placed": bool(voices),
        "voices": voices,
        "source": source,
    }


TITLED = [
    (
        "SPEAKER_00",
        "Professor Hannah Fry",
        "host",
        "Welcome to the podcast. I'm Professor Hannah Fry.",
    ),
    ("SPEAKER_01", "Joelle Barral", "guest", "Thank you for having me."),
    ("SPEAKER_00", "Professor Hannah Fry", "host", "So tell me about health."),
]


def test_registered_last_after_0024() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0025_one_person_one_entry") == len(ids) - 1
    assert ids.index("0024_shared_removed_speaker_prefixes") == len(ids) - 2


def test_a_titled_host_takes_the_stated_spelling_on_every_surface(tmp_path: Path) -> None:
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Professor Hannah Fry", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Joelle Barral", "guest", ["SPEAKER_01"], "llm_resolution", "guest"),
            _speaker("Hannah Fry", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=TITLED,
        known_hosts=["Hannah Fry"],
    )
    m = OnePersonOneEntryMigration()
    assert not m.verify(MigrationContext(corpus_root=tmp_path))[0]

    result = _apply(tmp_path)

    assert result.details["episodes"][0]["renames"] == {"Professor Hannah Fry": "Hannah Fry"}
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], s["voices"]) for s in speakers] == [
        ("Hannah Fry", "host", ["SPEAKER_00"]),
        ("Joelle Barral", "guest", ["SPEAKER_01"]),
    ]
    assert {r["speaker_label"] for r in _r(p["seg"])} == {"Hannah Fry", "Joelle Barral"}
    text = p["txt"].read_text(encoding="utf-8")
    assert text.startswith("Hannah Fry: Welcome") and "\nProfessor Hannah Fry: " not in text
    # The speech keeps what was said.
    assert "I'm Professor Hannah Fry." in text
    _quotes_hold(p)
    kg_names = {(n["properties"]["name"], n["properties"]["label"]) for n in _r(p["kg"])["nodes"]}
    assert ("Hannah Fry", "Hannah Fry") in kg_names
    assert {v["resolved_name"] for v in _r(p["diag"])["voices"]} == {"Hannah Fry", "Joelle Barral"}
    turns = _r(p["turns"])["turns"]
    assert [t["speaker_label"] for t in turns][0] == "Hannah Fry"
    for t in turns:
        line = text[: t["char_start"]].rsplit("\n", 1)[-1]
        assert line == t["speaker_label"] + ": "
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_a_respelt_host_cast_as_a_guest_becomes_the_host(tmp_path: Path) -> None:
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Joe Weisenthal", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Traci Alloway", "guest", ["SPEAKER_01"], "self_intro", "guest_1"),
            _speaker("Francis Fukuyama", "guest", ["SPEAKER_02"], "llm_resolution", "guest_2"),
            _speaker("Tracy Alloway", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_00", "Joe Weisenthal", "host", "I'm Joe Weisenthal."),
            ("SPEAKER_01", "Traci Alloway", "guest", "And I'm Traci Alloway."),
            ("SPEAKER_02", "Francis Fukuyama", "guest", "Glad to be here."),
        ],
        known_hosts=["Joe Weisenthal", "Tracy Alloway"],
    )
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["id"], s["name"], s["role"]) for s in speakers] == [
        ("host_1", "Joe Weisenthal", "host"),
        ("host_2", "Tracy Alloway", "host"),
        ("guest", "Francis Fukuyama", "guest"),
    ]
    rows = {r["speaker"]: r for r in _r(p["seg"])}
    assert (rows["SPEAKER_01"]["speaker_label"], rows["SPEAKER_01"]["speaker_role"]) == (
        "Tracy Alloway",
        "host",
    )
    kg = {n["properties"]["name"]: n["properties"]["role"] for n in _r(p["kg"])["nodes"]}
    assert kg["Tracy Alloway"] == "host" and "Traci Alloway" not in kg
    _quotes_hold(p)


def test_unplaced_duplicates_go_and_different_people_stay(tmp_path: Path) -> None:
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Bernard Leong", "host", [], "feed_statement", "unplaced_1"),
            _speaker("Bernard Leung", "guest", [], "episode_metadata", "unplaced_2"),
            _speaker("Bernard Leong", "host", [], "hint", "unplaced_3"),
            _speaker("Anna Jones", "guest", [], "episode_metadata", "unplaced_4"),
            _speaker("Anna Smith", "guest", [], "hint", "unplaced_5"),
        ],
        turns=[("SPEAKER_00", "SPEAKER_00", "host", "Hello.")],
        known_hosts=["Bernard Leong"],
    )
    txt_before = p["txt"].read_bytes()
    result = _apply(tmp_path)
    assert result.details["episodes"][0]["dropped"] == ["Bernard Leung", "Bernard Leong"]
    names = [(s["id"], s["name"]) for s in _r(p["meta"])["content"]["speakers"]]
    assert names == [
        ("unplaced_1", "Bernard Leong"),
        ("unplaced_2", "Anna Jones"),
        ("unplaced_3", "Anna Smith"),
    ]
    assert p["txt"].read_bytes() == txt_before  # nothing was renamed, no transcript touched


@pytest.mark.parametrize("with_turns", [True, False])
def test_a_titled_guest_whose_name_sits_on_the_host_voice_is_refused_untouched(
    tmp_path: Path, with_turns: bool
) -> None:
    # DeepMind 0014: the introduction reader put "Hannah Fry" on the guest's voice and her own
    # voice says "I'm Professor Hannah Fry" as a guest. Only the roster can say who is who.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Hannah Fry", "host", ["SPEAKER_04"], "introduced", "host"),
            _speaker("Professor Hannah Fry", "guest", ["SPEAKER_02"], "self_intro", "guest"),
            _speaker("Paige Bailey", "guest", [], "episode_metadata", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_04", "Hannah Fry", "host", "Welcome to Google DeepMind,"),
            (
                "SPEAKER_02",
                "Professor Hannah Fry",
                "guest",
                "the podcast. I'm Professor Hannah Fry.",
            ),
        ],
        known_hosts=["Hannah Fry"],
        with_turns=with_turns,
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert result.details["files_written"] == 0
    assert "held in another role" in result.details["refused"][0]["why"]
    assert {k: v.read_bytes() for k, v in p.items()} == before
    ok, msg = m.verify(MigrationContext(corpus_root=tmp_path))
    assert not ok and "1 refused" in msg


def test_two_placed_voices_with_two_names_are_left_to_the_roster(tmp_path: Path) -> None:
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Luke Timmerman", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Andy Ratcliffe", "guest", ["SPEAKER_01"], "forced", "guest_1"),
            _speaker("Andy Rachleff", "guest", ["SPEAKER_02"], "llm_resolution", "guest_2"),
        ],
        turns=[
            ("SPEAKER_00", "Luke Timmerman", "host", "Welcome."),
            ("SPEAKER_01", "Andy Ratcliffe", "guest", "Thanks."),
            ("SPEAKER_02", "Andy Rachleff", "guest", "Glad to be here."),
        ],
        known_hosts=["Luke Timmerman"],
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert result.details["left"][0]["pairs"] == [("Andy Ratcliffe", "Andy Rachleff")]
    assert {k: v.read_bytes() for k, v in p.items()} == before
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_a_title_with_no_stated_host_is_not_a_rename(tmp_path: Path) -> None:
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Kaiser Kuo", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Dr. Ruby Wang", "guest", ["SPEAKER_01"], "self_intro", "guest"),
        ],
        turns=[
            ("SPEAKER_00", "Kaiser Kuo", "host", "Welcome."),
            ("SPEAKER_01", "Dr. Ruby Wang", "guest", "Thank you."),
        ],
        known_hosts=["Kaiser Kuo"],
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    assert _apply(tmp_path).details["files_written"] == 0
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_dry_run_writes_nothing_undo_restores_and_a_second_apply_finds_nothing(
    tmp_path: Path,
) -> None:
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Professor Hannah Fry", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Joelle Barral", "guest", ["SPEAKER_01"], "llm_resolution", "guest"),
        ],
        turns=TITLED,
        known_hosts=["Hannah Fry"],
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    _apply(tmp_path, dry_run=True)
    assert {k: v.read_bytes() for k, v in p.items()} == before

    first = _apply(tmp_path)
    assert first.details["files_written"] == 7  # meta, segments, txt, gi, kg, diagnostics, turns
    assert _apply(tmp_path).details["files_written"] == 0

    restored, refused = undo(tmp_path)
    assert restored == 7 and not refused
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_respelling_this_fix_did_not_cause_is_left_alone(tmp_path: Path) -> None:
    # No title, and the host is not listed beside the voice: the roster's own rules already
    # applied to this name when it was written; the migration repairs only what the fix changed.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Tracy Allaway", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Francis Fukuyama", "guest", ["SPEAKER_01"], "llm_resolution", "guest"),
        ],
        turns=[
            ("SPEAKER_00", "Tracy Allaway", "host", "I'm Tracy Allaway."),
            ("SPEAKER_01", "Francis Fukuyama", "guest", "Glad to be here."),
        ],
        known_hosts=["Tracy Alloway"],
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    assert _apply(tmp_path).details["files_written"] == 0
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_respelt_unplaced_twin_on_one_episode_is_dropped() -> None:
    # The cross-source predicate needs one surname; on one episode a respelt surname is one person.
    from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import same_person

    assert same_person("Misha Glennie", "Misha Glenny")
    assert same_person("Professor Graham Pearson", "Grahame Pearson")
    assert not same_person("Anna Smith", "Anna Jones")
