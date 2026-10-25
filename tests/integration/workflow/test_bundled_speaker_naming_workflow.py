"""Transcript naming in the bundled modes through the real metadata-stage dispatcher.

``_generate_episode_summary`` is driven with a stub provider for each bundled mode: with transcript
naming on it must pass the speaker labels in ``params``, take the ``speakers`` answer out of the
summary, and record it in the speaker-diagnostics file beside the ladder; with it off it must
pass nothing and leave the file untouched. Synthetic fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from podcast_scraper import config
from podcast_scraper.providers.common.megabundle_parser import MegaBundleResult
from podcast_scraper.providers.ml.diarization import transcript_naming as tn

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

TRANSCRIPT = (
    "Tobias Wren: Welcome to the show, we talk about soil and water today.\n"
    "SPEAKER_01: Thanks for having me, I'm Maria Lindqvist.\n"
) * 8
DIAG = {
    "voices": [
        {
            "voice": "SPEAKER_00",
            "resolved_name": "Tobias Wren",
            "named": True,
            "role": "host",
            "source": "self_intro",
            "talk_time_s": 600,
        },
        {
            "voice": "SPEAKER_01",
            "resolved_name": "SPEAKER_01",
            "named": False,
            "role": "guest",
            "source": "raw",
            "talk_time_s": 500,
        },
    ]
}
SPEAKERS = {
    "Tobias Wren": {"name": "Tobias Wren", "role": "host", "confidence": "high", "evidence": ""},
    "SPEAKER_01": {
        "name": "Maria Lindqvist",
        "role": "guest",
        "confidence": "high",
        "evidence": "I'm Maria Lindqvist",
    },
}


def _cfg(mode: str, naming: str) -> config.Config:
    return config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "summary_provider": "anthropic",
            "anthropic_api_key": "test-api-key-123",
            "llm_pipeline_mode": mode,
            "transcript_naming": naming,
            "generate_summaries": True,
            "auto_speakers": False,
            "transcribe_missing": False,
            "generate_metadata": True,
        }
    )


def _bundle() -> MegaBundleResult:
    return MegaBundleResult(
        title="T",
        summary="S",
        bullets=["b1", "b2"],
        insights=[{"text": "ins1", "insight_type": "claim"}],
        topics=["topic1"],
        entities=[{"name": "E1", "kind": "org"}],
        raw={},
        speakers=SPEAKERS,
    )


def _provider(seen: List[Dict[str, Any]]) -> Mock:
    provider = Mock(
        spec=sorted(
            {
                "_summarization_initialized",
                "cleaning_processor",
                "summarize",
                "summarize_bundled",
                "summarize_mega_bundled",
                "summarize_extraction_bundled",
            }
        )
    )
    provider._summarization_initialized = True
    provider.cleaning_processor = None
    summary = {"title": "T", "summary": "S", "bullets": ["b1", "b2"]}

    def bundled(*a: Any, params: Any = None, **kw: Any) -> Dict[str, Any]:
        seen.append(dict(params or {}))
        body = dict(summary, speakers=SPEAKERS) if (params or {}).get(tn.PARAMS_LABELS) else summary
        return {"summary": json.dumps(body), "summary_short": None, "metadata": {"bundled": True}}

    def bundle(*a: Any, params: Any = None, **kw: Any) -> MegaBundleResult:
        seen.append(dict(params or {}))
        return _bundle()

    provider.summarize_bundled.side_effect = bundled
    provider.summarize_mega_bundled.side_effect = bundle
    provider.summarize_extraction_bundled.side_effect = bundle
    provider.summarize.side_effect = lambda *a, **kw: {
        "summary": json.dumps(summary),
        "summary_short": None,
        "metadata": {},
    }
    return provider


def _run(tmp_path: Path, mode: str, naming: str) -> Dict[str, Any]:
    from podcast_scraper.workflow import metrics as m
    from podcast_scraper.workflow.metadata_generation import _generate_episode_summary

    (tmp_path / "t.txt").write_text(TRANSCRIPT, encoding="utf-8")
    diag_path = tmp_path / "t.speakers.diagnostics.json"
    diag_path.write_text(json.dumps(DIAG), encoding="utf-8")
    seen: List[Dict[str, Any]] = []
    try:
        _generate_episode_summary(
            transcript_file_path="t.txt",
            output_dir=str(tmp_path),
            cfg=_cfg(mode, naming),
            episode_idx=0,
            summary_provider=_provider(seen),
            pipeline_metrics=m.Metrics(),
        )
    except Exception:  # noqa: BLE001 - downstream parsing of stub data may fail; not under test
        pass
    return {"seen": seen, "diag": json.loads(diag_path.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("mode", ["bundled", "mega_bundled", "extraction_bundled"])
def test_shadow_asks_inside_the_bundled_call_and_records_beside_the_ladder(
    tmp_path: Path, mode: str
) -> None:
    out = _run(tmp_path, mode, "shadow")
    asked = [p for p in out["seen"] if p.get(tn.PARAMS_LABELS)]
    assert asked and asked[0][tn.PARAMS_LABELS] == ["SPEAKER_01", "Tobias Wren"]
    side = out["diag"][tn.SIDECAR_KEY]
    assert side["call"] == mode and side["mode"] == "shadow" and side["applied"] is False
    assert side["voices"]["SPEAKER_01"]["model_result"]["name"] == "Maria Lindqvist"
    assert side["voices"]["SPEAKER_01"]["outcome"] == "adds"
    assert out["diag"]["voices"] == DIAG["voices"]


@pytest.mark.parametrize("mode", ["bundled", "mega_bundled", "extraction_bundled"])
def test_off_asks_nothing_and_leaves_the_sidecar_alone(tmp_path: Path, mode: str) -> None:
    out = _run(tmp_path, mode, "off")
    assert out["seen"] and not any(p.get(tn.PARAMS_LABELS) for p in out["seen"])
    assert out["diag"] == DIAG
