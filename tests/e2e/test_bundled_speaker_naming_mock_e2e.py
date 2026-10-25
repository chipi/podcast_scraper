"""Transcript naming inside the bundled calls, end to end against the e2e mock LLM server.

For every LLM provider, each bundled call it implements (bundled, mega_bundled,
extraction_bundled) is made over HTTP with the speaker labels in ``params``: the prompt must carry
the ``speakers`` field and the provider must hand the answer back (a MegaBundleResult's
``speakers``, or the field inside the bundled summary JSON). Without the labels the field must not
be in the prompt. Then the staged call is skipped in a bundled mode. Synthetic fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization import pipeline as P, transcript_naming as tn

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]

TRANSCRIPT = (
    "Tobias Wren: Welcome to the show, we talk about soil and water today. " * 6
    + "\nSPEAKER_01: Thanks for having me, I'm Maria Lindqvist. " * 6
)
LABELS = ["SPEAKER_01", "Tobias Wren"]
PROVIDERS = {
    "openai": ("openai", "openai_api_base"),
    "vllm": ("vllm", "vllm_api_base"),
    "litellm": ("litellm", "litellm_api_base"),
    "qwen": ("qwen", "qwen_api_base"),
    "deepseek": ("deepseek", "deepseek_api_base"),
    "groq": ("groq", "groq_api_base"),
    "gemini": ("gemini", "gemini_api_base"),
    "anthropic": ("anthropic", "anthropic_api_base"),
    "mistral": ("mistral", "mistral_api_base"),
    "grok": ("grok", "grok_api_base"),
    "ollama": ("ollama", "ollama_api_base"),
}
CALLS = ("summarize_bundled", "summarize_mega_bundled", "summarize_extraction_bundled")


def _cfg(e2e_server: Any, name: str, tmp_path: Path, **extra: Any) -> config.Config:
    prefix, url = PROVIDERS[name]
    kwargs: Dict[str, Any] = {
        "output_dir": str(tmp_path),
        "summary_provider": name,
        f"{prefix}_api_base": getattr(e2e_server.urls, url)(),
        "preload_models": False,
    }
    if name != "ollama":
        kwargs[f"{prefix}_api_key"] = "sk-test-e2e"
    if name in ("vllm", "litellm", "qwen"):
        kwargs[f"{prefix}_summary_model"] = "podcast-flash-0731"  # one the mock advertises
    kwargs.update(extra)
    return config.Config(**kwargs)


def _provider(cfg: config.Config) -> Any:
    from podcast_scraper.summarization.factory import create_summarization_provider

    provider = create_summarization_provider(cfg)
    provider.initialize()
    return provider


def _speakers(call: str, out: Any) -> Any:
    if call == "summarize_bundled":
        return json.loads(out["summary"]).get("speakers")
    return out.speakers


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_every_bundled_call_asks_and_returns_the_speakers_field(
    name: str, call: str, e2e_server: Any, tmp_path: Path
) -> None:
    from tests.e2e.fixtures.e2e_http_server import BUNDLED_NAMING_REQUESTS

    provider = _provider(_cfg(e2e_server, name, tmp_path))
    fn = getattr(provider, call, None)
    if not callable(fn):
        # Ollama has no mega or extraction bundle; its pipeline mode falls back to staged.
        assert name == "ollama" and call != "summarize_bundled", f"{name}.{call} missing"
        return
    BUNDLED_NAMING_REQUESTS.clear()
    out = fn(TRANSCRIPT, params={tn.PARAMS_LABELS: LABELS})
    assert len(BUNDLED_NAMING_REQUESTS) == 1, f"{name}.{call}: the field never reached the server"
    speakers = _speakers(call, out)
    assert speakers["Tobias Wren"]["name"] == "Tobias Wren", speakers
    assert speakers["SPEAKER_01"]["name"] is None


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
def test_without_labels_the_field_is_not_asked(
    name: str, call: str, e2e_server: Any, tmp_path: Path
) -> None:
    from tests.e2e.fixtures.e2e_http_server import BUNDLED_NAMING_REQUESTS

    provider = _provider(_cfg(e2e_server, name, tmp_path))
    BUNDLED_NAMING_REQUESTS.clear()
    try:
        getattr(provider, call)(TRANSCRIPT, params={})
    except Exception:  # noqa: BLE001 - the mock's ordinary answer may not parse as a bundle
        pass
    assert BUNDLED_NAMING_REQUESTS == []


@pytest.mark.parametrize("mode", ["bundled", "mega_bundled", "extraction_bundled"])
def test_a_bundled_mode_skips_the_staged_call(mode: str, e2e_server: Any, tmp_path: Path) -> None:
    from tests.e2e.fixtures.e2e_http_server import NAMING_REQUESTS

    cfg = _cfg(e2e_server, "openai", tmp_path, transcript_naming="shadow", llm_pipeline_mode=mode)
    NAMING_REQUESTS.clear()
    segs = [
        (
            {"start": 0, "end": 30, "text": "Welcome. I'm Tobias Wren. " + "indeed " * 50},
            "SPEAKER_00",
        )
    ]
    assert P._ask_transcript_naming(cfg, segs, "Ep 1", None) is None
    assert NAMING_REQUESTS == []
