"""Raw-label voices stay unnamed in the GI read model, search hits and MCP rosters (RFC-131, S3).

The insight view and the transcript hide raw labels; these are the other routes a speaker name
leaves the server by: ``gi.load`` (search insight hits, ``explore`` top speakers, MCP insight
tools), the lifted transcript speaker on search hits, and the MCP speaker roster.
"""

from __future__ import annotations

from typing import Any

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
