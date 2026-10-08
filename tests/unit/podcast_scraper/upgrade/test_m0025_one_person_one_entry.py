"""m0025: one person is one entry on an episode, whatever their title or spelling.

Synthetic fixtures shaped like the prod cases of 2026-10-08: a host's own "I'm Professor Hannah
Fry" beside the feed's "Hannah Fry"; Odd Lots' "Traci Alloway" cast as a guest beside the feed's
host Tracy Alloway; "Bernard Leong" stated and "Bernard Leung" hinted, both unplaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    feed_title: str = "Some Show",
    participants: Sequence[str] = (),
    llm_voice_names: Optional[Dict[str, str]] = None,
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
            "feed": {"title": feed_title},
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
            "tried": {"known_hosts": list(known_hosts), "metadata_named": list(participants)},
            "decision_trace": {"inputs": {"llm_voice_names": dict(llm_voice_names or {})}},
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


DEEPMIND_0014: Dict[str, Any] = dict(
    speakers=[
        _speaker("Hannah Fry", "host", ["SPEAKER_04"], "introduced", "host"),
        _speaker("Professor Hannah Fry", "guest", ["SPEAKER_02"], "self_intro", "guest"),
        _speaker("Paige Bailey", "guest", [], "episode_metadata", "unplaced_1"),
    ],
    turns=[
        ("SPEAKER_04", "Hannah Fry", "host", "Welcome to Google DeepMind,"),
        ("SPEAKER_02", "Professor Hannah Fry", "guest", "the podcast. I'm Professor Hannah Fry."),
        ("SPEAKER_04", "Hannah Fry", "host", "Thank you so much for having me."),
    ],
    known_hosts=["Hannah Fry"],
)


@pytest.mark.parametrize("with_turns", [True, False])
def test_the_host_and_the_guest_the_reader_swapped_are_swapped_back(
    tmp_path: Path, with_turns: bool
) -> None:
    # DeepMind 0014: the introduction reader put "Hannah Fry" on the guest's voice while her own
    # voice said "I'm Professor Hannah Fry"; the LLM had named the guest's voice Paige Bailey.
    p = _corpus(
        tmp_path,
        **DEEPMIND_0014,
        with_turns=with_turns,
        llm_voice_names={"SPEAKER_04": "Paige Bailey", "SPEAKER_02": "Hannah Fry"},
    )
    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert not result.details["refused"]
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], s["voices"]) for s in speakers] == [
        ("Hannah Fry", "host", ["SPEAKER_02"]),
        ("Paige Bailey", "guest", ["SPEAKER_04"]),
    ]
    rows = {r["speaker"]: (r["speaker_label"], r["speaker_role"]) for r in _r(p["seg"])}
    assert rows == {"SPEAKER_02": ("Hannah Fry", "host"), "SPEAKER_04": ("Paige Bailey", "guest")}
    text = p["txt"].read_text(encoding="utf-8")
    assert text.startswith("Paige Bailey: Welcome") and "\nHannah Fry: the podcast." in text
    _quotes_hold(p)
    # Each quote is credited to the voice its line belongs to.
    by_text = {
        n["properties"]["text"]: n["properties"]["speaker_id"]
        for n in _r(p["gi"])["nodes"]
        if n["type"] == "Quote"
    }
    assert by_text["Welcome to Google DeepMind,"] == "person:paige-bailey"
    assert by_text["the podcast. I'm Professor Hannah Fry."] == "person:hannah-fry"
    spoken = {e["from"]: e["to"] for e in _r(p["gi"])["edges"] if e["type"] == "SPOKEN_BY"}
    assert sorted(set(spoken.values())) == ["person:hannah-fry", "person:paige-bailey"]
    kg = {n["id"]: n["properties"]["role"] for n in _r(p["kg"])["nodes"]}
    assert kg["person:paige-bailey"] == "guest"
    if with_turns:
        turns = _r(p["turns"])["turns"]
        assert [t["speaker_label"] for t in turns] == ["Paige Bailey", "Hannah Fry", "Paige Bailey"]
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_without_the_llms_name_the_guests_voice_is_left_unnamed(tmp_path: Path) -> None:
    p = _corpus(tmp_path, **DEEPMIND_0014)
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], s["voices"]) for s in speakers] == [
        ("Hannah Fry", "host", ["SPEAKER_02"]),
        ("Paige Bailey", "guest", []),
    ]
    assert p["txt"].read_text(encoding="utf-8").startswith("SPEAKER_04: Welcome")
    _quotes_hold(p)


def test_a_conflict_without_a_self_introduced_host_is_still_refused(tmp_path: Path) -> None:
    case: Dict[str, Any] = dict(DEEPMIND_0014)
    case["speakers"] = [
        _speaker("Hannah Fry", "host", ["SPEAKER_04"], "llm_resolution", "host"),
        _speaker("Professor Hannah Fry", "guest", ["SPEAKER_02"], "self_intro", "guest"),
    ]
    p = _corpus(tmp_path, **case)
    before = {k: v.read_bytes() for k, v in p.items()}
    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert "held in another role" in result.details["refused"][0]["why"]
    assert {k: v.read_bytes() for k, v in p.items()} == before
    ok, msg = m.verify(MigrationContext(corpus_root=tmp_path))
    assert not ok and "1 refused" in msg


ANDY: Dict[str, Any] = dict(
    turns=[
        ("SPEAKER_00", "Luke Timmerman", "host", "Welcome. Today's guests are Yung and Andy."),
        (
            "SPEAKER_01",
            "Andy Ratcliffe",
            "guest",
            "So, as you know, Luke, we fund young scientists.",
        ),
        ("SPEAKER_02", "Andy Rachleff", "guest", "Well, science is under attack."),
    ],
    known_hosts=["Luke Timmerman"],
)


def test_a_forced_twin_of_a_placed_person_is_unnamed(tmp_path: Path) -> None:
    # The Long Run: the host's spoken "Andy Ratcliffe" was FORCED onto the co-guest's voice while
    # the LLM placed Andy Rachleff on his own.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Luke Timmerman", "host", ["SPEAKER_00"], "known_hosts", "host"),
            _speaker("Andy Ratcliffe", "guest", ["SPEAKER_01"], "forced", "guest_1"),
            _speaker("Andy Rachleff", "guest", ["SPEAKER_02"], "llm_resolution", "guest_2"),
        ],
        **ANDY,
    )
    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert not result.details["left"] and not result.details["refused"]
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["voices"]) for s in speakers] == [
        ("Luke Timmerman", ["SPEAKER_00"]),
        ("Andy Rachleff", ["SPEAKER_02"]),
    ]
    row = next(r for r in _r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    assert "speaker_label" not in row and row["voice_type"] == "unknown"
    assert "\nSPEAKER_01: So, as you know" in p["txt"].read_text(encoding="utf-8")
    quote = next(
        n["properties"]
        for n in _r(p["gi"])["nodes"]
        if n["type"] == "Quote" and "young" in n["properties"]["text"]
    )
    assert quote["speaker_id"] is None
    _quotes_hold(p)
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_two_placed_spellings_of_one_person_become_one_name_by_the_pipelines_rule(
    tmp_path: Path,
) -> None:
    # Two evidence-named voices, two spellings of one person, not in conversation with each other:
    # the pipeline's `_one_name_per_person` gives both one name (Talk Eastern Europe, In Our Time).
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Luke Timmerman", "host", ["SPEAKER_00"], "known_hosts", "host"),
            _speaker("Andy Ratcliffe", "guest", ["SPEAKER_01"], "introduced", "guest_1"),
            _speaker("Andy Rachleff", "guest", ["SPEAKER_02"], "llm_resolution", "guest_2"),
        ],
        participants=["Andy Rachleff"],
        **ANDY,
    )
    m = OnePersonOneEntryMigration()
    result = m.apply(MigrationContext(corpus_root=tmp_path))
    assert not result.details["left"] and not result.details["refused"]
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], sorted(s["voices"])) for s in speakers] == [
        ("Luke Timmerman", ["SPEAKER_00"]),
        ("Andy Rachleff", ["SPEAKER_01", "SPEAKER_02"]),
    ]
    _quotes_hold(p)
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_two_voices_in_conversation_stay_two_people(tmp_path: Path) -> None:
    # They trade the floor: two people who merely spell alike (the pipeline keeps them apart).
    turns = [("SPEAKER_00", "Luke Timmerman", "host", "Welcome to the show today.")]
    for i in range(12):
        turns += [
            ("SPEAKER_01", "Andy Ratcliffe", "guest", f"Point number {i} from me."),
            ("SPEAKER_02", "Andy Rachleff", "guest", f"And my reply number {i}."),
        ]
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Luke Timmerman", "host", ["SPEAKER_00"], "known_hosts", "host"),
            _speaker("Andy Ratcliffe", "guest", ["SPEAKER_01"], "introduced", "guest_1"),
            _speaker("Andy Rachleff", "guest", ["SPEAKER_02"], "llm_resolution", "guest_2"),
        ],
        turns=turns,
        known_hosts=["Luke Timmerman"],
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    result = OnePersonOneEntryMigration().apply(MigrationContext(corpus_root=tmp_path))
    assert result.details["left"][0]["pairs"] == [("Andy Ratcliffe", "Andy Rachleff")]
    assert {k: v.read_bytes() for k, v in p.items()} == before


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


def test_a_presenter_published_as_a_guest_becomes_the_unclaimed_host(tmp_path: Path) -> None:
    # Empire: the voice presenting the show said "Anita Arnond" and was published as a guest,
    # with the feed's host listed again beside it.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Anita Arnond", "guest", ["SPEAKER_03"], "self_intro", "guest_1"),
            _speaker("Fiona Hill", "guest", ["SPEAKER_02"], "self_intro", "guest_2"),
            _speaker("Anita Anand", "host", [], "feed_statement", "unplaced_1"),
            _speaker("William Dalrymple", "host", [], "feed_statement", "unplaced_2"),
        ],
        turns=[
            (
                "SPEAKER_03",
                "Anita Arnond",
                "guest",
                "Hello and welcome to Empire with me, Anita Arnond.",
            ),
            ("SPEAKER_02", "Fiona Hill", "guest", "Thank you so much for having me here."),
            ("SPEAKER_02", "Fiona Hill", "guest", "It is great to be here with you, and so on."),
        ],
        known_hosts=["Anita Anand", "William Dalrymple"],
        feed_title="Empire: World History",
        participants=["Fiona Hill"],
    )
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["id"], s["name"], s["role"], s["voices"]) for s in speakers] == [
        ("host", "Anita Anand", "host", ["SPEAKER_03"]),
        ("guest", "Fiona Hill", "guest", ["SPEAKER_02"]),
        ("unplaced_1", "William Dalrymple", "host", []),
    ]
    row = next(r for r in _r(p["seg"]) if r["speaker"] == "SPEAKER_03")
    assert (row["speaker_label"], row["speaker_role"]) == ("Anita Anand", "host")
    assert p["txt"].read_text(encoding="utf-8").startswith("Anita Anand: Hello and welcome")
    _quotes_hold(p)


def test_a_voice_that_does_not_present_the_show_is_not_made_its_host(tmp_path: Path) -> None:
    # The same names, but the voice never presents the show: only the duplicate entry goes.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Anita Arnond", "guest", ["SPEAKER_03"], "self_intro", "guest_1"),
            _speaker("Fiona Hill", "guest", ["SPEAKER_02"], "self_intro", "guest_2"),
            _speaker("Anita Anand", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_03", "Anita Arnond", "guest", "Thanks, glad to be on."),
            ("SPEAKER_02", "Fiona Hill", "guest", "Thank you so much for having me here."),
            ("SPEAKER_02", "Fiona Hill", "guest", "It is great to be here with you, and so on."),
        ],
        known_hosts=["Anita Anand"],
        feed_title="Empire: World History",
        participants=["Fiona Hill"],
    )
    result = _apply(tmp_path)
    assert result.details["episodes"][0]["renames"] == {}
    names = [(s["name"], s["role"]) for s in _r(p["meta"])["content"]["speakers"]]
    assert names == [("Anita Arnond", "guest"), ("Fiona Hill", "guest")]


def test_a_stated_participant_takes_its_spelling_and_keeps_the_guest_role(tmp_path: Path) -> None:
    # The a16z Show lists its interviewee among nine "hosts"; his voice said "Lucas Kaiser".
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Sophia Dew", "host", ["SPEAKER_00"], "llm_resolution", "host"),
            _speaker("Lucas Kaiser", "guest", ["SPEAKER_02"], "self_intro", "guest"),
            _speaker("Ben Horowitz", "host", [], "feed_statement", "unplaced_1"),
            _speaker("Lukasz Kaiser", "host", [], "feed_statement", "unplaced_2"),
        ],
        turns=[
            ("SPEAKER_00", "Sophia Dew", "host", "Joining me is an AI researcher."),
            ("SPEAKER_02", "Lucas Kaiser", "guest", "There was a very important moment."),
        ],
        known_hosts=["Sophia Dew", "Ben Horowitz", "Lukasz Kaiser"],
        participants=["Lukasz Kaiser"],
    )
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], bool(s["voices"])) for s in speakers] == [
        ("Sophia Dew", "host", True),
        ("Lukasz Kaiser", "guest", True),
        ("Ben Horowitz", "host", False),
    ]
    row = next(r for r in _r(p["seg"]) if r["speaker"] == "SPEAKER_02")
    assert (row["speaker_label"], row["speaker_role"]) == ("Lukasz Kaiser", "guest")
    _quotes_hold(p)


def test_a_respelt_voice_beside_the_hosts_own_voice_is_unified_not_made_a_second_host_twice(
    tmp_path: Path,
) -> None:
    # "Anita Arnond" presenting and "Anita Anand" placed as host on another voice: the presenter
    # rule does not claim a host another voice holds; the pipeline's one-name rule then gives both
    # voices the stated name, and the consecutive lines become one, as the formatter writes them.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Anita Anand", "host", ["SPEAKER_01"], "known_hosts", "host"),
            _speaker("Anita Arnond", "guest", ["SPEAKER_03"], "self_intro", "guest_1"),
            _speaker("Fiona Hill", "guest", ["SPEAKER_02"], "self_intro", "guest_2"),
        ],
        turns=[
            (
                "SPEAKER_03",
                "Anita Arnond",
                "guest",
                "Hello and welcome to Empire with me, Anita Arnond.",
            ),
            ("SPEAKER_01", "Anita Anand", "host", "And our guest today."),
            ("SPEAKER_02", "Fiona Hill", "guest", "Thank you so much for having me here."),
            ("SPEAKER_02", "Fiona Hill", "guest", "It is great to be here with you, and so on."),
        ],
        known_hosts=["Anita Anand"],
        feed_title="Empire: World History",
        participants=["Fiona Hill"],
    )
    result = _apply(tmp_path)
    assert not result.details["refused"] and not result.details["left"]
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], sorted(s["voices"])) for s in speakers] == [
        ("Anita Anand", "host", ["SPEAKER_01", "SPEAKER_03"]),
        ("Fiona Hill", "guest", ["SPEAKER_02"]),
    ]
    assert (
        p["txt"]
        .read_text(encoding="utf-8")
        .startswith(
            "Anita Anand: Hello and welcome to Empire with me, Anita Arnond. And our guest today."
        )
    )
    _quotes_hold(p)
    turns = _r(p["turns"])["turns"]
    assert [t["speaker_label"] for t in turns] == ["Anita Anand", "Fiona Hill"]


def test_the_presenter_rule_takes_only_one_unclaimed_unstated_host() -> None:
    from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import _unclaimed_host

    hosts = ["Anita Anand", "William Dalrymple"]
    assert _unclaimed_host("Anita Arnond", hosts, set(), set()) == "Anita Anand"
    # "Anita Anant" respells both stated hosts: ambiguity keeps the spoken form.
    assert _unclaimed_host("Anita Anant", hosts + ["Anita Anaut"], set(), set()) is None
    assert _unclaimed_host("Anita Anant", hosts, set(), set()) == "Anita Anand"
    assert _unclaimed_host("Anita Arnond", hosts, {"Anita Anand"}, set()) is None
    assert _unclaimed_host("Anita Arnond", hosts, set(), {"anita anand"}) is None


def test_a_rename_that_only_restores_a_credential_is_none(tmp_path: Path) -> None:
    # The Peter Attia Drive states "Peter Attia, MD"; the record publishes the canonical form.
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Peter Attia", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Jane Doe", "guest", ["SPEAKER_01"], "llm_resolution", "guest"),
        ],
        turns=[
            ("SPEAKER_00", "Peter Attia", "host", "Welcome, I'm Peter Attia."),
            ("SPEAKER_01", "Jane Doe", "guest", "Thanks for having me."),
        ],
        known_hosts=["Peter Attia, MD"],
        participants=["Peter Attia, MD"],
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    assert _apply(tmp_path).details["files_written"] == 0
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_swap_needs_the_voice_to_have_introduced_itself(tmp_path: Path) -> None:
    case: Dict[str, Any] = dict(DEEPMIND_0014)
    case["speakers"] = [
        _speaker("Hannah Fry", "host", ["SPEAKER_04"], "introduced", "host"),
        _speaker("Professor Hannah Fry", "guest", ["SPEAKER_02"], "llm_resolution", "guest"),
    ]
    p = _corpus(tmp_path, **case, llm_voice_names={"SPEAKER_04": "Paige Bailey"})
    before = {k: v.read_bytes() for k, v in p.items()}
    result = _apply(tmp_path)
    assert "held in another role" in result.details["refused"][0]["why"]
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_the_guests_voice_takes_only_a_person_the_record_states(tmp_path: Path) -> None:
    # The LLM's name for the guest's voice must be someone the record lists, unplaced.
    p = _corpus(tmp_path, **DEEPMIND_0014, llm_voice_names={"SPEAKER_04": "Somebody Else"})
    _apply(tmp_path)
    voices = {
        tuple(s["voices"]): s["name"] for s in _r(p["meta"])["content"]["speakers"] if s["voices"]
    }
    assert voices == {("SPEAKER_02",): "Hannah Fry"}


def test_a_mentioned_person_who_speaks_becomes_a_guest_in_the_graph(tmp_path: Path) -> None:
    p = _corpus(
        tmp_path,
        **DEEPMIND_0014,
        llm_voice_names={"SPEAKER_04": "Paige Bailey"},
    )
    kg = _r(p["kg"])
    kg["nodes"].append(
        {
            "id": "person:paige-bailey",
            "type": "Person",
            "properties": {"name": "Paige Bailey", "label": "Paige Bailey", "role": "mentioned"},
        }
    )
    _w(p["kg"], kg)
    _apply(tmp_path)
    roles = [
        n["properties"]["role"] for n in _r(p["kg"])["nodes"] if n["id"] == "person:paige-bailey"
    ]
    assert roles == ["guest"]


def test_an_absolute_transcript_path_is_read_and_written_inside_this_corpus(tmp_path: Path) -> None:
    # Some metadata stores the transcript path ABSOLUTE, as written at ingest. Upgrading a copy or
    # a restore must touch that copy only — never the path the ingest wrote (2026-10-08: a copy
    # test reached for the live corpus and `write_with_backup` refused).
    corpus = tmp_path / "copy"
    p = _corpus(
        corpus,
        speakers=[
            _speaker("Professor Hannah Fry", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Joelle Barral", "guest", ["SPEAKER_01"], "llm_resolution", "guest"),
        ],
        turns=TITLED,
        known_hosts=["Hannah Fry"],
    )
    elsewhere = tmp_path / "live" / "feeds" / "f" / "run_1" / "transcripts" / "ep.txt"
    meta = _r(p["meta"])
    meta["content"]["transcript_file_path"] = str(elsewhere)
    _w(p["meta"], meta)
    result = _apply(corpus)
    assert result.details["files_written"] == 7
    assert p["txt"].read_text(encoding="utf-8").startswith("Hannah Fry: Welcome")
    assert not (tmp_path / "live").exists()
    _quotes_hold(p)


def test_an_older_record_without_voices_is_read_through_its_segments(tmp_path: Path) -> None:
    # Half the corpus predates `voices` / `placed`: every entry is placed on the voices its label
    # is on. Dropping one as "unplaced" would leave the record disagreeing with its own segments.
    p = _corpus(
        tmp_path,
        speakers=[
            {"id": "host", "name": "Tim Romero", "role": "host"},
            {"id": "guest", "name": "Charming Lai", "role": "guest"},
        ],
        turns=[
            ("SPEAKER_00", "Tim Romero", "host", "Welcome to Disrupting Japan."),
            ("SPEAKER_01", "Charming Lai", "guest", "Hi, my name is Charming Lai."),
        ],
        known_hosts=["Tim Romero"],
        participants=["Chiamin Lai"],
    )
    _apply(tmp_path)
    assert [s["name"] for s in _r(p["meta"])["content"]["speakers"]] == [
        "Tim Romero",
        "Chiamin Lai",
    ]
    assert {r["speaker_label"] for r in _r(p["seg"])} == {"Tim Romero", "Chiamin Lai"}
    _quotes_hold(p)


def test_a_listed_forced_ad_voice_is_unnamed(tmp_path: Path, monkeypatch) -> None:
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    monkeypatch.setattr(m25, "REPLAYED_VOICES", {("ep", "SPEAKER_02"): (None, None, "an ad read")})
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Sean Illing", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker(
                "Anna Louie Sussman",
                "guest",
                ["SPEAKER_01", "SPEAKER_02"],
                "llm_resolution",
                "guest",
            ),
        ],
        turns=[
            ("SPEAKER_00", "Sean Illing", "host", "Welcome to the show."),
            (
                "SPEAKER_02",
                "Anna Louie Sussman",
                "guest",
                "Support for the show comes from an advertiser.",
            ),
            ("SPEAKER_01", "Anna Louie Sussman", "guest", "Thanks for having me on."),
        ],
        known_hosts=["Sean Illing"],
    )
    gi = _r(p["gi"])
    gi["nodes"].append(
        {
            "id": "insight:1",
            "type": "Insight",
            "properties": {"text": "x", "speaker": "Anna Louie Sussman"},
        }
    )
    _w(p["gi"], gi)
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["voices"]) for s in speakers] == [
        ("Sean Illing", ["SPEAKER_00"]),
        ("Anna Louie Sussman", ["SPEAKER_01"]),
    ]
    insight = next(n for n in _r(p["gi"])["nodes"] if n["id"] == "insight:1")
    assert insight["properties"]["speaker"] == "Anna Louie Sussman"  # she still speaks
    row = next(r for r in _r(p["seg"]) if r["speaker"] == "SPEAKER_02")
    assert "speaker_label" not in row
    assert "\nSPEAKER_02: Support for the show" in p["txt"].read_text(encoding="utf-8")
    _quotes_hold(p)
    # Her own quote keeps her; only the ad's loses the name — by WHERE each quote sits, since
    # both voices carried the same label.
    credited = {
        n["properties"]["text"]: n["properties"]["speaker_id"]
        for n in _r(p["gi"])["nodes"]
        if n["type"] == "Quote"
    }
    assert credited["Thanks for having me on."] == _pid("Anna Louie Sussman")
    assert credited["Support for the show comes from an advertiser."] is None
    kg = {n["id"] for n in _r(p["kg"])["nodes"]}
    assert _pid("Anna Louie Sussman") in kg


def test_a_listed_voice_older_code_left_unnamed_takes_the_replayed_name(
    tmp_path: Path, monkeypatch
) -> None:
    # Empire '400. Stalin': the co-host's voice was stored unnamed; today's roster names him.
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    monkeypatch.setattr(
        m25, "REPLAYED_VOICES", {("ep", "SPEAKER_01"): ("William Dalrymple", "host", "co-host")}
    )
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Anita Anand", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Fiona Hill", "guest", ["SPEAKER_02"], "self_intro", "guest"),
        ],
        turns=[
            ("SPEAKER_00", "Anita Anand", "host", "Welcome to Empire."),
            ("SPEAKER_01", "SPEAKER_01", "host", "And we've got you someone who sat with Putin."),
            ("SPEAKER_02", "Fiona Hill", "guest", "Thank you for having me."),
        ],
        known_hosts=["Anita Anand", "William Dalrymple"],
    )
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], s["voices"]) for s in speakers] == [
        ("Anita Anand", "host", ["SPEAKER_00"]),
        ("William Dalrymple", "host", ["SPEAKER_01"]),
        ("Fiona Hill", "guest", ["SPEAKER_02"]),
    ]
    row = next(r for r in _r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    assert (row["speaker_label"], row["speaker_role"]) == ("William Dalrymple", "host")
    assert "\nWilliam Dalrymple: And we've got you" in p["txt"].read_text(encoding="utf-8")
    _quotes_hold(p)


def test_the_unplaced_entry_for_that_person_takes_the_named_voice(
    tmp_path: Path, monkeypatch
) -> None:
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    monkeypatch.setattr(
        m25, "REPLAYED_VOICES", {("ep", "SPEAKER_01"): ("William Dalrymple", "host", "co-host")}
    )
    p = _corpus(
        tmp_path,
        speakers=[
            _speaker("Anita Anand", "host", ["SPEAKER_00"], "self_intro", "host"),
            _speaker("Fiona Hill", "guest", ["SPEAKER_02"], "self_intro", "guest"),
            _speaker("William Dalrymple", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_00", "Anita Anand", "host", "Welcome to Empire."),
            ("SPEAKER_01", "SPEAKER_01", "host", "And we've got you someone who sat with Putin."),
            ("SPEAKER_02", "Fiona Hill", "guest", "Thank you for having me."),
        ],
        known_hosts=["Anita Anand", "William Dalrymple"],
    )
    _apply(tmp_path)
    speakers = _r(p["meta"])["content"]["speakers"]
    assert [(s["name"], s["role"], s["voices"]) for s in speakers] == [
        ("Anita Anand", "host", ["SPEAKER_00"]),
        ("William Dalrymple", "host", ["SPEAKER_01"]),
        ("Fiona Hill", "guest", ["SPEAKER_02"]),
    ]


def test_a_replayed_voice_is_applied_once(tmp_path: Path, monkeypatch) -> None:
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    monkeypatch.setattr(
        m25, "REPLAYED_VOICES", {("ep", "SPEAKER_01"): ("William Dalrymple", "host", "co-host")}
    )
    _corpus(
        tmp_path,
        speakers=[_speaker("Anita Anand", "host", ["SPEAKER_00"], "self_intro", "host")],
        turns=[
            ("SPEAKER_00", "Anita Anand", "host", "Welcome to Empire."),
            ("SPEAKER_01", "SPEAKER_01", "host", "And we've got you someone who sat with Putin."),
        ],
        known_hosts=["Anita Anand", "William Dalrymple"],
    )
    assert _apply(tmp_path).details["files_written"] > 0
    assert _apply(tmp_path).details["files_written"] == 0
    assert OnePersonOneEntryMigration().verify(MigrationContext(corpus_root=tmp_path))[0]


def test_the_host_takes_its_spelling_once_its_name_leaves_the_guests_voice(
    tmp_path: Path, monkeypatch
) -> None:
    # Analyse: the guest's voice was stored as the host "Bernard Leong" (unnamed by the replayed
    # list); the host's own voice said "Bernard Leung" and takes the feed's spelling.
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    monkeypatch.setattr(m25, "REPLAYED_VOICES", {("ep", "SPEAKER_00"): (None, None, "the guest")})
    p = _corpus(
        tmp_path,
        speakers=[
            {"id": "host", "name": "Bernard Leong", "role": "host"},
            {"id": "guest", "name": "Bernard Leung", "role": "guest"},
        ],
        turns=[
            ("SPEAKER_00", "Bernard Leong", "host", "I started my career as a journalist."),
            (
                "SPEAKER_01",
                "Bernard Leung",
                "guest",
                "Welcome to Analyse Podcast. I'm Bernard Leung.",
            ),
        ],
        known_hosts=["Bernard Leong"],
    )
    _apply(tmp_path)
    labels = {r["speaker"]: r.get("speaker_label") for r in _r(p["seg"])}
    assert labels == {"SPEAKER_00": None, "SPEAKER_01": "Bernard Leong"}
    assert [s["name"] for s in _r(p["meta"])["content"]["speakers"]] == ["Bernard Leong"]
    _quotes_hold(p)
