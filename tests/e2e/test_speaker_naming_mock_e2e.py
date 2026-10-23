"""Transcript naming end to end against the e2e mock LLM server, through every LLM provider.

Each provider is built from config exactly as the pipeline builds it, and its ``complete_text``
goes over HTTP to the mock (``tests/e2e/fixtures/e2e_http_server.py``, ``_naming_response_json``),
which answers from the transcript's own words: the voice that welcomes and introduces itself is
the host, the voice thanked for having it is the guest, and the ad read is an ad with no name.
Then the shadow and publish modes run through ``apply_diarization_to_result`` on a real provider.
All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization import pipeline as P, transcript_naming as tn
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]

WORDS = " ".join(["indeed"] * 45)
SEGMENTS: List[Dict[str, Any]] = [
    {"start": 0, "end": 30, "text": "Welcome to the show. I'm Tobias Wren. " + WORDS},
    {"start": 30, "end": 90, "text": "Thanks for having me. I'm Maria Lindqvist. " + WORDS},
    {"start": 90, "end": 120, "text": "This episode is brought to you by Acme. " + WORDS},
]
VOICES = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
EXPECTED = {
    "SPEAKER_00": ("Tobias Wren", "host"),
    "SPEAKER_01": ("Maria Lindqvist", "guest"),
    "SPEAKER_02": (None, "ad"),
}

# provider -> (summary_provider value, config prefix for key/base, E2EServerURLs method)
PROVIDERS = {
    "openai": ("openai", "openai", "openai_api_base"),
    "gemini": ("gemini", "gemini", "gemini_api_base"),
    "anthropic": ("anthropic", "anthropic", "anthropic_api_base"),
    "mistral": ("mistral", "mistral", "mistral_api_base"),
    "grok": ("grok", "grok", "grok_api_base"),
    "deepseek": ("deepseek", "deepseek", "deepseek_api_base"),
    "groq": ("groq", "groq", "groq_api_base"),
    "ollama": ("ollama", "ollama", "ollama_api_base"),
    "vllm": ("vllm", "vllm", "vllm_api_base"),
    "litellm": ("litellm", "litellm", "litellm_api_base"),
    "qwen": ("qwen", "qwen", "qwen_api_base"),
}


def _cfg(e2e_server: Any, provider: str, tmp_path: Path, **extra: Any) -> config.Config:
    summary_provider, prefix, url = PROVIDERS[provider]
    kwargs: Dict[str, Any] = {
        "output_dir": str(tmp_path),
        "summary_provider": summary_provider,
        f"{prefix}_api_base": getattr(e2e_server.urls, url)(),
        "speaker_resolution_llm": False,
        "preload_models": False,
    }
    if provider != "ollama":
        kwargs[f"{prefix}_api_key"] = "sk-test-e2e"
    kwargs.update(extra)
    return config.Config(**kwargs)


def _aligned():
    return [(seg, voice) for seg, voice in zip(SEGMENTS, VOICES)]


def test_the_mock_answers_only_naming_prompts() -> None:
    from tests.e2e.fixtures.e2e_http_server import _naming_response_json

    assert _naming_response_json("Summarize this transcript.") is None
    text = tn.transcript_block(_aligned(), "The Wren Hour", None, "Ep 1", None)
    answer = json.loads(_naming_response_json(tn.build_prompt(text, VOICES)) or "{}")
    got = {v: (a["name"], a["role"]) for v, a in answer["voices"].items()}
    assert got == EXPECTED
    assert answer["voices"]["SPEAKER_00"]["evidence"].startswith("[00:00] I'm Tobias Wren")


_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "maxOutputTokens")


def _find(body: Any, keys: Any) -> Any:
    """The first value under any of ``keys`` anywhere in a request body."""
    if isinstance(body, dict):
        for k, v in body.items():
            if k in keys:
                return v
            found = _find(v, keys)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_every_provider_names_the_voices_over_http(
    provider: str, e2e_server: Any, tmp_path: Path
) -> None:
    from tests.e2e.fixtures.e2e_http_server import NAMING_REQUESTS

    NAMING_REQUESTS.clear()
    cfg = _cfg(e2e_server, provider, tmp_path)
    asked = P._ask_transcript_naming(cfg, _aligned(), "Ep 1", None)
    assert len(NAMING_REQUESTS) == 1, f"{provider}: the naming prompt never reached the server"
    # The whole-episode answer needs its own output budget, not the resolver's 800.
    assert _find(NAMING_REQUESTS[0], _LIMIT_KEYS) == tn.MAX_OUTPUT_TOKENS, NAMING_REQUESTS[0]
    if provider == "ollama":
        assert _find(NAMING_REQUESTS[0], ("num_ctx",)) == cfg.ollama_num_ctx
    assert asked is not None, f"{provider}: no completion endpoint"
    assert asked.ok, f"{provider}: {asked.report}"
    assert asked.voices_asked == VOICES
    assert {v: (a["name"], a["role"]) for v, a in (asked.answer or {}).items()} == EXPECTED
    assert asked.report["provider"]


def _pipeline(cfg: config.Config) -> Dict[str, Any]:
    P._copresence_cache.clear()
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 30, "SPEAKER_00"),
            DiarizationSegment(30, 90, "SPEAKER_01"),
            DiarizationSegment(90, 120, "SPEAKER_02"),
        ],
        num_speakers=3,
    )
    result = {
        "text": " ".join(str(s["text"]) for s in SEGMENTS),
        "segments": [dict(s) for s in SEGMENTS],
    }
    out: Dict[str, Any] = P.apply_diarization_to_result(
        result,
        "",
        cfg,
        ["Maria Lindqvist"],
        precomputed_diarization=diar,
        feed_hosts=["Tobias Wren"],
        episode_title="Ep 1",
    )
    return out


def _names(out: Dict[str, Any]) -> Dict[str, Any]:
    return {v["voice"]: v["resolved_name"] for v in out["speaker_diagnostics"]["voices"]}


@pytest.mark.parametrize("provider", ["vllm", "openai", "anthropic", "gemini"])
def test_shadow_records_the_model_beside_the_ladder_through_a_real_provider(
    provider: str, e2e_server: Any, tmp_path: Path
) -> None:
    off = _pipeline(_cfg(e2e_server, provider, tmp_path, transcript_naming="off"))
    shadow = _pipeline(_cfg(e2e_server, provider, tmp_path, transcript_naming="shadow"))
    assert tn.SIDECAR_KEY not in off["speaker_diagnostics"]
    assert _names(shadow) == _names(off)
    assert shadow["segments"] == off["segments"]
    side = shadow["speaker_diagnostics"][tn.SIDECAR_KEY]
    assert side["ok"] is True and side["mode"] == "shadow"
    assert side["voices"]["SPEAKER_01"]["model_result"]["name"] == "Maria Lindqvist"
    assert side["voices"]["SPEAKER_02"]["model"]["role"] == "ad"
    assert side["report"]["clip"]["turns_dropped"] == 0


def test_publish_names_the_roster_through_a_real_provider(e2e_server: Any, tmp_path: Path):
    out = _pipeline(_cfg(e2e_server, "openai", tmp_path, transcript_naming="publish"))
    names = _names(out)
    assert names["SPEAKER_00"] == "Tobias Wren"
    assert names["SPEAKER_01"] == "Maria Lindqvist"
    assert out["speaker_diagnostics"][tn.SIDECAR_KEY]["mode"] == "publish"
