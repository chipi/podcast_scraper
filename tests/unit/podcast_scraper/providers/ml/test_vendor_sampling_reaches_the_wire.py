"""The registry's researched sampling knob must reach an actual request (#2051).

``resolve_profile_to_settings`` used to write the summary stage's ``extra_settings`` into
``settings["summary_extra"]``. Nothing read that key — one producer, zero consumers — so
``presence_penalty: 1.5``, the vendor's documented mitigation for the endless repetition behind
#2053, sat declared in the registry for the life of the DGX deployment and never reached a single
request.

These tests pin the two halves that make the fix real: the value is materialized onto a governed
Config field, and the provider puts it on every chat call. The second half is the one that was
missing, so a test that only checks the resolver would have passed against the bug.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.model_registry import (
    _VENDOR_SAMPLING_HELD,
    _VENDOR_SAMPLING_PLUMBED,
    REGISTRY_GOVERNED_FIELDS,
    StageOption,
    _emit_vendor_sampling,
    resolve_profile_to_settings,
)

pytestmark = pytest.mark.unit


class TestTheRegistryValueBecomesAGovernedSetting:
    @pytest.mark.parametrize("profile", ["prod_dgx_full", "dev_dgx_full"])
    def test_the_dgx_profiles_carry_the_vendors_penalty(self, profile: str) -> None:
        assert resolve_profile_to_settings(profile)["vllm_presence_penalty"] == 1.5

    def test_it_is_governed_so_a_profile_cannot_silently_disagree(self) -> None:
        # Ungoverned means a hand-authored YAML can hold a different value and nothing notices.
        assert "vllm_presence_penalty" in REGISTRY_GOVERNED_FIELDS

    def test_the_three_held_knobs_are_still_not_emitted(self) -> None:
        # Plumbing temperature 0.3 -> 0.7 changes every summary in the corpus and no gate in this
        # repo can see a summary getting blander. It ships with its own A/B or not at all.
        settings = resolve_profile_to_settings("prod_dgx_full")
        for held in _VENDOR_SAMPLING_HELD:
            assert f"vllm_{held}" not in settings, held


class TestAnUnknownKnobFailsLoudlyInsteadOfVanishing:
    """The whole bug was a value disappearing quietly. A new one must not."""

    def _option(self, vendor: Dict[str, Any]) -> StageOption:
        return StageOption(
            stage="summary",
            option_id="test_option",
            provider="vllm",
            model="test/model",
            extra_settings={"vendor_sampling": vendor},
        )

    def test_an_unrecognised_key_raises_and_names_both_escape_hatches(self) -> None:
        settings: Dict[str, Any] = {}
        with pytest.raises(RuntimeError) as exc:
            _emit_vendor_sampling(self._option({"mirostat_tau": 5.0}), settings)
        msg = str(exc.value)
        assert "mirostat_tau" in msg
        assert "_VENDOR_SAMPLING_PLUMBED" in msg and "_VENDOR_SAMPLING_HELD" in msg

    def test_it_is_a_runtime_error_not_a_value_error(self) -> None:
        # Config._resolve_profile catches ValueError to mean "not a registry preset" and drops to
        # YAML-only — so a ValueError here would disable the registry for that profile in silence.
        with pytest.raises(RuntimeError):
            _emit_vendor_sampling(self._option({"nonsense": 1}), {})

    def test_a_non_dict_vendor_sampling_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not a dict"):
            bad = "presence_penalty=1.5"
            _emit_vendor_sampling(self._option(bad), {})  # type: ignore[arg-type]

    def test_the_held_keys_pass_the_guard_without_being_emitted(self) -> None:
        settings: Dict[str, Any] = {}
        _emit_vendor_sampling(self._option({"temperature": 0.7, "top_p": 0.8}), settings)
        assert settings == {}

    def test_the_plumbed_and_held_sets_do_not_overlap(self) -> None:
        assert not set(_VENDOR_SAMPLING_PLUMBED) & _VENDOR_SAMPLING_HELD


def _vllm_provider(**overrides: Any):
    """A REAL ``VLLMProvider`` over a mocked OpenAI client.

    Deliberately not a re-implementation of the wrapper: the whole point of #2051 is that the
    resolver half worked and the provider half did not exist, so a test that re-creates the
    provider logic would have passed against the bug it is meant to catch. This builds the real
    object through the real constructor and only replaces the network client underneath it.
    """
    from podcast_scraper.config import Config
    from podcast_scraper.providers.vllm import VLLMProvider

    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="vllm",
        speaker_detector_provider="vllm",
        generate_summaries=True,
        generate_metadata=True,
        vllm_api_base="http://dgx-llm-1:8003/v1",
        vllm_summary_model="NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4",
        vllm_speaker_model="NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4",
    )
    base.update(overrides)

    sent: List[Dict[str, Any]] = []

    def _record(**kwargs: Any) -> str:
        sent.append(kwargs)
        return "ok"

    client = MagicMock()
    client.chat.completions.create = _record
    with patch("openai.OpenAI", return_value=client):
        provider = VLLMProvider(Config(**base))
    return provider, sent


class TestTheValueActuallyReachesTheRequest:
    """The half a resolver-only test would have missed."""

    def test_every_chat_call_carries_the_registry_penalty(self) -> None:
        provider, sent = _vllm_provider(vllm_presence_penalty=1.5)
        provider.client.chat.completions.create(model="m", messages=[])
        assert sent[0]["presence_penalty"] == 1.5

    def test_it_is_absent_when_the_registry_declares_nothing(self) -> None:
        # Sending 0.0 is a real instruction to the sampler; omitting the field is not the same
        # thing, so "unset" must mean the key never appears.
        provider, sent = _vllm_provider()
        provider.client.chat.completions.create(model="m", messages=[])
        assert "presence_penalty" not in sent[0]

    def test_an_explicit_per_call_value_wins(self) -> None:
        provider, sent = _vllm_provider(vllm_presence_penalty=1.5)
        provider.client.chat.completions.create(model="m", messages=[], presence_penalty=0.2)
        assert sent[0]["presence_penalty"] == 0.2, (
            "a provider-wide default that silently beats an explicit argument is the same class "
            "of bug as the one #2051 fixes"
        )

    def test_it_composes_with_extra_body_rather_than_replacing_it(self) -> None:
        # chat_template_kwargs suppresses Qwen3 reasoning prose (#960). Losing it while adding the
        # penalty would trade one silent regression for another.
        provider, sent = _vllm_provider(
            vllm_presence_penalty=1.5,
            vllm_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        provider.client.chat.completions.create(model="m", messages=[])
        assert sent[0]["presence_penalty"] == 1.5
        assert sent[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_extra_body_alone_still_works_with_no_penalty_configured(self) -> None:
        provider, sent = _vllm_provider(
            vllm_extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        provider.client.chat.completions.create(model="m", messages=[])
        assert sent[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert "presence_penalty" not in sent[0]
