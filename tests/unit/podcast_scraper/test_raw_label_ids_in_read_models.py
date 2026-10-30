"""Raw-label voices stay unnamed in the GI read model, search hits and MCP rosters (RFC-131, S3).

The insight view and the transcript hide raw labels; these are the other routes a speaker name
leaves the server by: ``gi.load`` (search insight hits, ``explore`` top speakers, MCP insight
tools), the lifted transcript speaker on search hits, and the MCP speaker roster.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.graph_id_utils import is_raw_label_person_id


@pytest.mark.parametrize(
    "pid,raw",
    [
        ("person:speaker-ep1-07", True),
        ("person:speaker-d99b4720-283b-4f07-89c4-791b12acba2d-07", True),
        ("person:speaker-ep1-unknown-guest-1", True),
        ("person:unknown-guest-1", True),  # the legacy global id of the detector default
        ("person:speaker-ep1-host", False),  # a role voice: "Host" stays readable in its episode
        ("person:unresolved-host-ep1", False),
        ("person:speaker-john-knight", False),  # a real person named "Speaker John Knight"
        ("person:jorna-velt", False),
        ("", False),
        (None, False),
    ],
)
def test_is_raw_label_person_id(pid: Any, raw: bool) -> None:
    assert is_raw_label_person_id(pid) is raw


def _gi(name: str, pid: str) -> Dict[str, Any]:
    return {
        "episode_id": "ep1",
        "nodes": [
            {
                "id": "insight:a",
                "type": "Insight",
                "properties": {"text": "claim", "grounded": True},
            },
            {
                "id": "quote:a",
                "type": "Quote",
                "properties": {"text": "said", "char_start": 0, "char_end": 4, "speaker_id": pid},
            },
            {"id": pid, "type": "Person", "properties": {"name": name}},
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:a", "to": "quote:a"},
            {"type": "SPOKEN_BY", "from": "quote:a", "to": pid},
        ],
    }


def _supporting(gi: Dict[str, Any]):
    from podcast_scraper.gi.load import build_inspect_output

    out = build_inspect_output(gi, transcript_text="said")
    return out.insights[0].supporting_quotes[0]


def test_the_read_model_gives_a_raw_label_voice_no_name_and_no_id() -> None:
    q = _supporting(_gi("SPEAKER_07", "person:speaker-ep1-07"))
    assert (q.speaker_name, q.speaker_id) == (None, None)


def test_the_read_model_keeps_a_named_speaker_and_a_host() -> None:
    q = _supporting(_gi("Jorna Velt", "person:jorna-velt"))
    assert (q.speaker_name, q.speaker_id) == ("Jorna Velt", "person:jorna-velt")
    h = _supporting(_gi("Host", "person:unresolved-host-ep1"))
    assert h.speaker_name == "Host"


@pytest.mark.parametrize(
    "pid,name,want",
    [
        (
            "person:jorna-velt",
            "Jorna Velt",
            {"id": "person:jorna-velt", "display_name": "Jorna Velt"},
        ),
        ("person:speaker-ep1-07", "", None),  # raw voice, no bridge name: no speaker
        ("person:speaker-ep1-07", "SPEAKER_07", None),  # raw voice named by its label
        ("person:jorna-velt", "unknown_guest_1", None),
        (
            "person:unresolved-host-ep1",
            "Host",
            {"id": "person:unresolved-host-ep1", "display_name": "Host"},
        ),
        (None, "Jorna Velt", None),
    ],
)
def test_the_lifted_search_speaker_is_never_a_raw_label(pid, name, want) -> None:
    from podcast_scraper.search.transcript_chunk_lift import _lifted_speaker

    assert _lifted_speaker(pid, name) == want


def test_the_mcp_roster_lists_no_raw_label_as_a_speaker() -> None:
    from podcast_scraper.mcp.tools.enrichment import _roster_from_record

    content = {
        "speakers": [
            {"name": "Jorna Velt", "role": "host", "placed": True, "voices": ["SPEAKER_00"]},
            {"name": "SPEAKER_01", "role": "guest", "placed": True, "voices": ["SPEAKER_01"]},
            {"name": "unknown_guest_1", "role": "guest", "placed": False, "voices": []},
        ]
    }
    diag = {"voices": [{"voice": "SPEAKER_00", "talk_time_s": 10.0}]}
    assert [r["name"] for r in _roster_from_record(content, diag)] == ["Jorna Velt"]
