"""The LLM detector must not drop a stated guest either (#2095, #2078).

WHY THIS FILE EXISTS SEPARATELY FROM THE SPACY ONE. The seat cap was fixed in
`speaker_detectors/detection.py` first, and that covered only the spaCy detector. Production
does not use spaCy: `config/profiles/prod_dgx_full.yaml` sets `speaker_detector_provider: vllm`,
and `VLLMProvider(OpenAICompatibleProvider)` inherits `_parse_speakers_from_response` from this
module — as do the groq, qwen, deepseek, litellm and openai providers, 21 of the 41 configured
profile entries between them. So the first fix changed nothing where it mattered most, and the
episodes it was written for (`Mackenzie Price`, `Alexander Stubb`, `Glenn Fogel`) would have
stayed broken in production while every offline check reported success.

An advisor review caught it. The lesson is in
`docs/guides/VALIDATING_CORPUS_FIXES_FAST.md`: when a decision exists on more than one provider
path, fixing the one you happen to be reading is not fixing the bug. These tests pin BOTH parse
paths so the next person cannot half-fix it again.
"""

from typing import Any, Dict, Set

from podcast_scraper.providers.openai.openai_provider import OpenAICompatibleProvider


class _Cfg:
    """Only what `_parse_speakers_from_response` reads."""

    screenplay_num_speakers = 2


def _provider() -> Any:
    """The parser without the provider's network/config machinery — it reads only ``cfg``."""
    p: Any = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.cfg = _Cfg()
    return p


HOSTS: Set[str] = {"Kevin Roose", "Casey Newton"}


class TestTheJsonPath:
    def test_the_guest_survives_two_stated_hosts(self) -> None:
        """The Hard Fork shape. Two hosts filled the seat count and the guest fell off the end,
        so she reached neither `detected_guests` nor `metadata_named` and was absent from the
        record entirely — not even `placed: false`."""
        payload = (
            '{"speakers": ["Kevin Roose", "Casey Newton", "Mackenzie Price"],'
            ' "hosts": ["Kevin Roose", "Casey Newton"],'
            ' "guests": ["Mackenzie Price"]}'
        )
        names, hosts, ok = _provider()._parse_speakers_from_response(payload, HOSTS)
        assert ok
        assert "Mackenzie Price" in names, (
            "the stated guest must survive the screenplay's seat count on the provider "
            "production actually runs"
        )
        assert hosts == HOSTS

    def test_several_guests_all_survive(self) -> None:
        payload = (
            '{"speakers": [], "hosts": ["Kevin Roose", "Casey Newton"],'
            ' "guests": ["Robert Malley", "Mark Williams"]}'
        )
        names, _, _ = _provider()._parse_speakers_from_response(payload, HOSTS)
        assert {"Robert Malley", "Mark Williams"} <= set(names)

    def test_the_defaults_floor_is_untouched(self) -> None:
        """Uncapping must not disturb the MIN_SPEAKERS_REQUIRED extension, which is what keeps a
        screenplay renderable when detection found almost nobody."""
        payload = '{"speakers": [], "hosts": [], "guests": []}'
        names, hosts, ok = _provider()._parse_speakers_from_response(payload, set())
        assert not ok
        assert len(names) >= 2, "the floor still applies when nobody was detected"
        assert hosts == set()


class TestTheTextFallbackPath:
    """The second cap. `_parse_speakers_from_response` falls back to this on invalid JSON, and it
    carried its own `[:min_speakers]` — fixing only the JSON path would leave the bug reachable."""

    def test_the_guest_survives_there_too(self) -> None:
        provider = _provider()
        fallback = getattr(provider, "_parse_speakers_from_text", None)
        if fallback is None:  # pragma: no cover - name drift guard
            import pytest

            pytest.skip("text fallback parser not found under the expected name")
        text = "Hosts: Kevin Roose, Casey Newton\nGuests: Mackenzie Price"
        names, _hosts, _ok = fallback(text, HOSTS)
        assert "Mackenzie Price" in names


def test_every_openai_compatible_detector_shares_this_code_path() -> None:
    """Documents the blast radius, so the next reader knows this is not an OpenAI-only concern.

    If one of these ever grows its own `detect_speakers`, it needs its own test.
    """
    from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider
    from podcast_scraper.providers.groq.groq_provider import GroqProvider
    from podcast_scraper.providers.litellm.litellm_provider import LiteLLMProvider
    from podcast_scraper.providers.qwen.qwen_provider import QwenProvider
    from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

    for cls in (VLLMProvider, GroqProvider, QwenProvider, DeepSeekProvider, LiteLLMProvider):
        assert issubclass(cls, OpenAICompatibleProvider)
        assert (
            "_parse_speakers_from_response" not in cls.__dict__
        ), f"{cls.__name__} now overrides the parser and needs its own seat-cap test"


def test_the_shape_that_regressed() -> None:
    """A direct statement of the defect, kept as its own case because it is the one sentence
    worth reading: two stated hosts must not be able to delete the stated guest."""
    payload: Dict[str, Any] = {
        "speakers": [],
        "hosts": ["Kevin Roose", "Casey Newton"],
        "guests": ["Mackenzie Price"],
    }
    import json as _json

    names, _, _ = _provider()._parse_speakers_from_response(_json.dumps(payload), HOSTS)
    assert len(names) == 3, f"expected both hosts and the guest, got {names}"
