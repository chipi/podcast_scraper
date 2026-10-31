"""Raw diarization labels are never shown as a speaker (RFC-131, S3; operator decision 2026-10-30).

Measured on the served corpus (2026-10-30): 66 insights in 21 episodes showed a raw label as their
speaker, and 222,461 transcript segments in 732 episodes showed ``SPEAKER_NN``. The operator's
rule: such labels are hidden, and an insight whose speaker is one is unattributed; an unnamed
voice in the transcript reads "Unidentified speaker 1", "… 2", numbered by first appearance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.graph_id_utils import is_bare_speaker_label, is_raw_speaker_label
from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.segments_view import to_contract_segments


def _gi(quotes: List[Dict[str, Any]], persons: Dict[str, str]) -> Dict[str, Any]:
    """One insight per quote spec ``{"id", "person"}``; synthetic names only."""
    nodes: List[Dict[str, Any]] = [
        {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in persons.items()
    ]
    edges: List[Dict[str, Any]] = []
    for q in quotes:
        iid, qid = f"insight:{q['id']}", f"quote:{q['id']}"
        nodes.append({"id": iid, "type": "Insight", "properties": {"text": f"claim {q['id']}"}})
        nodes.append({"id": qid, "type": "Quote", "properties": {"text": f"said {q['id']}"}})
        edges.append({"type": "SUPPORTED_BY", "from": iid, "to": qid})
        for person in q["persons"]:
            edges.append({"type": "SPOKEN_BY", "from": qid, "to": person})
    return {"nodes": nodes, "edges": edges}


# ---- the predicate -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,raw",
    [
        ("SPEAKER_07", True),
        ("Speaker 3", True),
        ("unknown_guest_1", True),  # the speaker detector's failure default
        ("Unknown Guest 2", True),
        ("Host", False),  # a role word: readable in its own episode
        ("Unidentified speaker", False),  # a deliberate label
        ("Speaker John Knight", False),  # a real name
        ("Jorna Velt", False),
        (None, False),
    ],
)
def test_is_raw_speaker_label(name: Optional[str], raw: bool) -> None:
    assert is_raw_speaker_label(name) is raw


def test_the_detector_default_is_a_placeholder_everywhere() -> None:
    # Every filter built on is_bare_speaker_label (speaker cards, people surfaces, follow targets)
    # now drops it too; before, "unknown_guest_1" rendered as a person chip on a global id.
    assert is_bare_speaker_label("unknown_guest_1")


# ---- the insight view ----------------------------------------------------------------------------


def test_a_raw_label_speaker_is_no_name_and_its_insight_is_unattributed() -> None:
    gi = _gi(
        [{"id": "a", "persons": ["person:speaker-ep1-07"]}],
        {"person:speaker-ep1-07": "SPEAKER_07"},
    )
    assert insights_from_gi(gi) == []  # off highlights and every default surface
    (fallback,) = insights_from_gi(gi, unattributed_fallback=True)
    assert fallback.attributed is False
    assert [q.speaker for q in fallback.quotes] == [None]


def test_the_detector_default_on_a_global_id_is_hidden_too() -> None:
    gi = _gi(
        [{"id": "a", "persons": ["person:unknown-guest-1"]}],
        {"person:unknown-guest-1": "unknown_guest_1"},
    )
    assert insights_from_gi(gi) == []


def test_a_named_speaker_and_host_still_render() -> None:
    gi = _gi(
        [
            {"id": "a", "persons": ["person:jorna-velt"]},
            {"id": "b", "persons": ["person:unresolved-host-ep1"]},
        ],
        {"person:jorna-velt": "Jorna Velt", "person:unresolved-host-ep1": "Host"},
    )
    speakers = sorted(str(q.speaker) for ins in insights_from_gi(gi) for q in ins.quotes)
    assert speakers == ["Host", "Jorna Velt"]


def test_one_named_quote_keeps_an_insight_attributed() -> None:
    gi = _gi(
        [{"id": "a", "persons": ["person:speaker-ep1-07"]}],
        {"person:speaker-ep1-07": "SPEAKER_07", "person:jorna-velt": "Jorna Velt"},
    )
    # Add a second, named quote to the same insight.
    gi["nodes"].append({"id": "quote:b", "type": "Quote", "properties": {"text": "said b"}})
    gi["edges"] += [
        {"type": "SUPPORTED_BY", "from": "insight:a", "to": "quote:b"},
        {"type": "SPOKEN_BY", "from": "quote:b", "to": "person:jorna-velt"},
    ]
    (ins,) = insights_from_gi(gi)
    assert ins.attributed is True
    assert sorted(q.speaker or "" for q in ins.quotes) == ["", "Jorna Velt"]


# ---- the transcript ------------------------------------------------------------------------------


def _seg(start: float, label: str, voice_type: Optional[str] = None, role: Optional[str] = None):
    row: Dict[str, Any] = {"start": start, "end": start + 1, "text": "x", "speaker_label": label}
    if voice_type:
        row["voice_type"] = voice_type
    if role:
        row["speaker_role"] = role
    return row


def test_unnamed_voices_are_numbered_by_first_appearance() -> None:
    segs = to_contract_segments(
        [
            _seg(0, "Jorna Velt"),
            _seg(1, "SPEAKER_03", "unknown"),
            _seg(2, "SPEAKER_01", "unknown"),
            _seg(3, "SPEAKER_03", "unknown"),
            _seg(4, "SPEAKER_02", "cameo"),
            _seg(5, "SPEAKER_00", role="host"),
            _seg(6, "SPEAKER_05", "unidentified"),
            _seg(7, "SPEAKER_06"),  # no voice_type at all: still not shown raw
        ]
    )
    assert [s.speaker for s in segs] == [
        "Jorna Velt",
        "Unidentified speaker 1",
        "Unidentified speaker 2",
        "Unidentified speaker 1",
        "Brief speaker",
        "Host",
        "Unidentified speaker 3",
        "Unidentified speaker 4",
    ]
    # Typed for the client's i18n (RFC-131 L7): kind and number, not only the English text.
    assert [(s.speaker_kind, s.speaker_index) for s in segs] == [
        ("person", None),
        ("unidentified", 1),
        ("unidentified", 2),
        ("unidentified", 1),
        ("brief", None),
        ("host", None),
        ("unidentified", 3),
        ("unidentified", 4),
    ]
    assert not any(s.speaker and is_raw_speaker_label(s.speaker) for s in segs)


def test_a_voice_typed_unknown_is_unidentified_whatever_its_label() -> None:
    # The roster's voice type decides: a substantive voice we failed to name is unidentified even
    # when its stored label is not raw (an old artifact can carry a leftover string there).
    (seg,) = to_contract_segments([_seg(0, "voice a", "unknown")])
    assert (seg.speaker, seg.speaker_kind, seg.speaker_index) == (
        "Unidentified speaker 1",
        "unidentified",
        1,
    )
