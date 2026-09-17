"""A guest the model names on a two-host show reaches the roster's candidate list (#2075).

Real parser (`OpenAICompatibleProvider.detect_speakers` and `_parse_speakers_from_response`), real
pipeline step (`processing._detect_speakers_for_episode`: placeholder/organisation filters,
corroboration), with only the model call stubbed to return the shape vLLM returned for the Odd
Lots episode. Before the fix the guest never left the parser.
"""

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from podcast_scraper.providers.openai.openai_provider import OpenAICompatibleProvider
from podcast_scraper.workflow.stages import processing
from podcast_scraper.workflow.types import HostDetectionResult

pytestmark = [pytest.mark.integration]

TITLE = "Regional Fed President Cal Schmid on the First Symposium of the Era"
DESC = "As we have in the past, we speak with Cal Schmid, the president of the regional Fed."


def _detector() -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.cfg = SimpleNamespace(  # type: ignore[assignment]
        auto_speakers=True, screenplay_num_speakers=2, openai_speaker_system_prompt=None
    )
    p._speaker_detection_initialized = True  # type: ignore[attr-defined]
    p.speaker_model = "stub"  # type: ignore[attr-defined]
    p.speaker_temperature = 0.0  # type: ignore[attr-defined]
    p._token_kwarg = lambda n, *_a: {"max_tokens": n}  # type: ignore[method-assign,misc,assignment]
    p._build_speaker_detection_prompt = lambda *a, **k: "prompt"  # type: ignore[method-assign]
    reply = {
        "speakers": ["Ada Brook", "Ben Carver", "Cal Schmid"],
        "hosts": ["Ada Brook", "Ben Carver"],
        "guests": ["Cal Schmid"],
    }
    msg = SimpleNamespace(content=json.dumps(reply))
    p._chat_create = lambda **_k: SimpleNamespace(  # type: ignore[method-assign]
        choices=[SimpleNamespace(message=msg)], usage=None
    )
    p._emit_stage_cost = lambda **_k: None  # type: ignore[method-assign]
    return p


def test_the_guest_is_a_candidate(monkeypatch: Any) -> None:
    monkeypatch.setattr("podcast_scraper.prompts.store.render_prompt", lambda *a, **k: "system")
    cfg = MagicMock()
    cfg.auto_speakers = True
    cfg.dry_run = False
    cfg.screenplay_speaker_names = []
    cfg.cache_detected_hosts = False
    cfg.speaker_detector_provider = "vllm"
    cfg.known_hosts = ["Ada Brook", "Ben Carver"]
    episode = MagicMock()
    episode.idx = 1
    episode.title = TITLE
    episode.description = DESC
    item = ET.Element("item")
    ET.SubElement(item, "description").text = DESC
    episode.item = item  # the pipeline reads the description from the RSS item
    hdr = HostDetectionResult(
        cached_hosts={"Ada Brook", "Ben Carver"}, heuristics=None, speaker_detector=_detector()
    )
    result = processing._detect_speakers_for_episode(episode, cfg, hdr, None)
    assert result is not None
    assert "Cal Schmid" in (result.guests or []), result
    assert "Cal Schmid" in (result.stated or []), result
