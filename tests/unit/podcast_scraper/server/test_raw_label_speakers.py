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


# ---- the transcript ------------------------------------------------------------------------------


def _seg(start: float, label: str, voice_type: Optional[str] = None, role: Optional[str] = None):
    row: Dict[str, Any] = {"start": start, "end": start + 1, "text": "x", "speaker_label": label}
    if voice_type:
        row["voice_type"] = voice_type
    if role:
        row["speaker_role"] = role
    return row
