"""The per-voice decision trace (#2276): a pure observer of the naming ladder.

Two guarantees:
1. A roster built WITH a trace is identical to one built without — the trace never decides.
2. The trace records, per voice and per name, which rung proposed / accepted / refused / overrode,
   so a published roster can be reconstructed from the sidecar.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.naming_trace import (
    NamingTrace,
    NullTrace,
    TRACE_VERSION,
)
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

pytestmark = pytest.mark.unit

HOST = "Tobias Wren"
GUEST = "Maria Lindqvist"


def _as_production(turns: List[Tuple[str, str, float]]):
    segs: List[DiarizationSegment] = []
    chunks: Dict[str, List[str]] = {}
    ordered: List[Tuple[str, str]] = []
    t = 30.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        chunks.setdefault(spk, []).append(" " + text)
        ordered.append((spk, " " + text))
    voice_texts = {v: " ".join(c) for v, c in chunks.items()}
    return DiarizationResult(segments=segs, num_speakers=len(voice_texts)), voice_texts, ordered


_INTERVIEW = [
    ("SPEAKER_00", "Hello and welcome to the show. I'm Tobias Wren.", 6.0),
    ("SPEAKER_00", "Today I'm chatting with Maria Lindqvist about the Baltic ports.", 6.0),
    ("SPEAKER_01", "Thanks so much for having me, Tobias. It is a pleasure to be here.", 8.0),
    ("SPEAKER_01", "The merchant family I follow kept ledgers for three centuries.", 120.0),
    ("SPEAKER_00", "What surprised you most in those ledgers?", 5.0),
    ("SPEAKER_01", "How often a silted river decided who got rich and who did not.", 120.0),
    # A short third voice the LLM will try to name with a name the episode already bound.
    ("SPEAKER_02", "Our show is brought to you by readers like you.", 25.0),
]


def _resolve(turns, trace=None, **kw):
    dz, vt, ordered = _as_production(turns)
    return resolve_speaker_roster(
        dz,
        " ".join(t for _, t in ordered),
        known_hosts=[HOST],
        metadata_named=[GUEST],
        voice_texts=vt,
        ordered_turns=ordered,
        trace=trace,
        **kw,
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"llm_voice_names": {"SPEAKER_02": GUEST}, "llm_voice_roles": {"SPEAKER_02": "guest"}},
        {"llm_voice_names": {"SPEAKER_01": GUEST, "SPEAKER_02": "Before Gene"}},
        {"detected_guests": [GUEST], "feed_title": "The Tobias Show"},
    ],
    ids=["rules_only", "llm_reuses_a_bound_name", "llm_junk_name", "corroborated_guest"],
)
def test_a_traced_roster_is_identical_to_an_untraced_one(extra) -> None:
    plain = _resolve(_INTERVIEW, None, **extra)
    traced = _resolve(_INTERVIEW, NamingTrace(), **extra)
    assert traced.by_voice == plain.by_voice
    assert traced.num_speakers == plain.num_speakers


_SETS_A_NAME = {"named", "renamed", "set", "changed", "accepted", "added", "restored"}


def test_the_trace_records_inputs_seats_and_names() -> None:
    trace = NamingTrace()
    roster = _resolve(_INTERVIEW, trace)
    d = trace.to_dict()
    assert d["version"] == 1
    assert d["inputs"]["known_hosts"] == [HOST]
    assert d["inputs"]["metadata_named"] == [GUEST]
    seats = next(s for s in d["episode"] if s["rung"] == "host_seats")
    assert seats["voices"] == [v for v, r in roster.by_voice.items() if r.role == "host"]
    # Every voice the roster published by name has a step that SET that name (a skipped proposal
    # carries `proposed`, never `name`, so it cannot satisfy this).
    for v, r in roster.by_voice.items():
        if r.named:
            steps = d["voices"].get(v, [])
            assert any(s.get("name") == r.name and s["decision"] in _SETS_A_NAME for s in steps), (
                v,
                r.name,
                steps,
            )


def test_an_llm_name_on_an_already_named_voice_is_recorded_as_skipped() -> None:
    """The self-introduced voice keeps its own name; the LLM's proposal for it is on record."""
    trace = NamingTrace()
    _resolve(_INTERVIEW, trace, llm_voice_names={"SPEAKER_00": "Somebody Else"})
    steps = trace.to_dict()["voices"]["SPEAKER_00"]
    skipped = [s for s in steps if s["rung"] == "llm_merge"]
    assert skipped and skipped[0]["decision"] == "skipped"
    assert skipped[0]["reason"] == "already_named" and skipped[0]["proposed"] == "Somebody Else"
    assert "name" not in skipped[0]


def test_a_publish_gate_refusal_records_the_rejected_name() -> None:
    trace = NamingTrace()
    roster = _resolve(_INTERVIEW, trace, llm_voice_names={"SPEAKER_02": "Before Gene"})
    assert not roster.by_voice["SPEAKER_02"].named  # the gate refused the junk name
    d = trace.to_dict()
    refused = [s for s in d["voices"].get("SPEAKER_02", []) if s["rung"] == "publish_gate"]
    assert refused and refused[0]["decision"] == "refused" and refused[0]["name"] == "Before Gene"
    assert d["names"]["Before Gene"][0]["rung"] == "publish_gate"


def test_diff_helpers_report_only_changes() -> None:
    t = NamingTrace()
    t.diff_names("stage", {"A": "x", "B": "y"}, {"A": "x", "B": "z", "C": "w"})
    v = t.to_dict()["voices"]
    assert "A" not in v
    assert v["B"][0] == {"rung": "stage", "decision": "renamed", "name": "z", "previous": "y"}
    assert v["C"][0] == {"rung": "stage", "decision": "named", "name": "w"}


def test_a_role_prefix_the_publish_gate_strips_is_recorded() -> None:
    trace = NamingTrace()
    roster = _resolve(_INTERVIEW, trace, llm_voice_names={"SPEAKER_02": "Your Host Anna Berg"})
    assert roster.by_voice["SPEAKER_02"].name == "Anna Berg"
    steps = [s for s in trace.to_dict()["voices"]["SPEAKER_02"] if s["rung"] == "publish_gate"]
    assert steps[0] == {
        "rung": "publish_gate",
        "decision": "prefix_stripped",
        "name": "Anna Berg",
        "previous": "Your Host Anna Berg",
    }


def test_a_publisher_label_is_recorded_even_when_it_changes_nothing() -> None:
    trace = NamingTrace()
    _resolve(_INTERVIEW, trace, stated_voice_names={"SPEAKER_00": HOST})
    steps = trace.to_dict()["voices"]["SPEAKER_00"]
    assert {"rung": "publisher_label", "decision": "stated", "name": HOST} in steps


def test_a_recorder_error_degrades_the_trace_instead_of_breaking_the_roster() -> None:
    trace = NamingTrace()
    trace.voices = None  # type: ignore[assignment]  # every per-voice record now raises
    plain = _resolve(_INTERVIEW, None)
    traced = _resolve(_INTERVIEW, trace)
    assert traced.by_voice == plain.by_voice
    assert trace.degraded
    assert trace.to_dict()["degraded"] is True


def test_the_null_trace_records_nothing_and_never_raises() -> None:
    t = NullTrace()
    t.voices = None  # type: ignore[assignment]
    t.voice("A", "r", "named", name="x")
    t.guest_pool(None, None, None, None)  # type: ignore[arg-type]
    t.host_pool(None, None)  # type: ignore[arg-type]
    assert not t.degraded and t.episode == [] and t.inputs == {}


def test_to_dict_is_a_json_safe_copy() -> None:
    t = NamingTrace()
    t.input("pool", {"b", "a"})
    t.note("stage", ratio=float("nan"))
    d = t.to_dict()
    assert d["degraded"] is True and "error" in d  # NaN: stub, the sidecar still writes
    t2 = NamingTrace()
    t2.input("pool", {"b", "a"})
    d2 = t2.to_dict()
    assert d2["inputs"]["pool"] == ["a", "b"]
    d2["inputs"]["pool"].append("c")
    assert t2.inputs["pool"] == {"a", "b"}


def test_known_hosts_none_is_accepted() -> None:
    dz, vt, ordered = _as_production(_INTERVIEW)
    text = " ".join(t for _, t in ordered)
    resolve_speaker_roster(dz, text, known_hosts=None, voice_texts=vt)  # type: ignore[arg-type]


def test_the_design_doc_states_the_trace_schema_the_code_writes() -> None:
    """A schema change must land with the doc that explains how to read it."""
    doc = Path(__file__).parents[6] / "docs" / "wip" / "NAMING_DECISION_TRACE.md"
    m = re.search(r"^\| Trace schema \| `TRACE_VERSION = (\d+)`", doc.read_text(), re.M)
    assert m, "the doc's 'Trace schema' row is missing"
    assert int(m.group(1)) == TRACE_VERSION
