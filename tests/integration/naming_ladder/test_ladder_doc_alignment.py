"""The ladder document and the scenario library describe the same ladder (#2304, segment L).

Every rung listed in ``docs/guides/NAMING_LADDER.md`` ("Rungs recorded, in ladder order") must
be exercised by at least one scenario's trace, or appear in ``NOT_YET_EXERCISED`` with the reason.
A rung added to the document without a scenario, or a scenario run that stops reaching a rung,
fails here, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

import pytest

from . import harness

pytestmark = pytest.mark.integration

DOC = Path(__file__).resolve().parents[3] / "docs" / "guides" / "NAMING_LADDER.md"

#: Rungs the library does not reach yet, each with why. Shrinks as scenarios are added; a rung that
#: becomes reached must leave this list (the test below says so).
NOT_YET_EXERCISED: dict = {
    # Both rungs are off by default and no scenario profile switches them on; each is covered by
    # its own unit tests (test_voice_identity.py, test_talk_floor.py).
    "voice_identity": "off by default (voice_identity_rung); needs a feed fingerprint bank",
    "talk_floor": "off by default (speaker_name_min_talk_share = 0)",
    "transcript_naming": "off by default; needs an LLM call the scenario harness does not make",
}


def _doc_rungs() -> Set[str]:
    text = DOC.read_text(encoding="utf-8")
    table = text.split("### Rungs recorded, in ladder order")[1].split("**Sources a voice")[0]
    keys: Set[str] = set()
    for line in table.splitlines():
        if line.startswith("| `"):
            keys |= set(re.findall(r"`([a-z_.]+)`", line.split("|")[1]))
    return keys


def _seen() -> Set[str]:
    seen: Set[str] = set()
    for sc in harness.load_scenarios():
        seen |= harness.rungs_seen(harness.run(sc))
    return seen


def test_every_documented_rung_is_exercised_or_listed() -> None:
    missing = sorted(_doc_rungs() - _seen() - set(NOT_YET_EXERCISED))
    assert not missing, f"documented rungs no scenario reaches: {missing}"


def test_the_not_yet_list_holds_only_rungs_still_unreached() -> None:
    reached = sorted(set(NOT_YET_EXERCISED) & _seen())
    assert not reached, f"now reached by a scenario, remove from NOT_YET_EXERCISED: {reached}"


def test_the_not_yet_list_names_only_documented_rungs() -> None:
    stray = sorted(set(NOT_YET_EXERCISED) - _doc_rungs())
    assert not stray, f"not in the ladder document: {stray}"
