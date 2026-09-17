"""A show with two stated hosts still gets its guests detected (#2075).

Both detector paths cut the name list to `screenplay_num_speakers` (2) with hosts FIRST, so on a
two-host show every guest was dropped before the pipeline saw it. Found on Odd Lots: the vLLM reply
carried `"guests": ["Jeffrey Schmid"]` and detection reported nobody. Measured on 56 production
episodes of 12 multi-host shows with the same model: corroborated guests 6 -> 30.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from podcast_scraper.providers.openai.openai_provider import OpenAICompatibleProvider
from podcast_scraper.speaker_detectors.detection import _build_speaker_names_list

pytestmark = pytest.mark.unit

HOSTS = {"Ada Brook", "Ben Carver"}


def _parser() -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.cfg = SimpleNamespace(screenplay_num_speakers=2)  # type: ignore[assignment]
    return p


def test_the_llm_parser_keeps_a_guest_on_a_two_host_show() -> None:
    reply = json.dumps(
        {
            "speakers": ["Ada Brook", "Ben Carver", "Cal Schmid"],
            "hosts": ["Ada Brook", "Ben Carver"],
            "guests": ["Cal Schmid"],
        }
    )
    names, hosts, ok = _parser()._parse_speakers_from_response(reply, HOSTS)
    assert ok and hosts == HOSTS
    assert "Cal Schmid" in names


def test_the_text_fallback_keeps_a_guest_on_a_two_host_show() -> None:
    reply = "Hosts: Ada Brook, Ben Carver\nGuests: Cal Schmid, Dee Lane"
    names, _hosts, _ok = _parser()._parse_speakers_from_text(reply, HOSTS)
    assert {"Cal Schmid", "Dee Lane"} <= set(names)


def test_the_ner_list_keeps_every_guest_on_a_two_host_show() -> None:
    names, ok, _defaults = _build_speaker_names_list(HOSTS, ["Cal Schmid", "Dee Lane"], 2)
    assert ok
    assert {"Cal Schmid", "Dee Lane"} <= set(names)
