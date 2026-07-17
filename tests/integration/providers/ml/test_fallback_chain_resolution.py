"""RFC-105 (#1198): the registry owns the per-stage failover ladder, and the resolver emits it.

The bug this guards: promoting MOSS as the DGX transcription default (#1174) silently dropped the
fallback, because the ``moss`` provider has none of its own. Under RFC-105 the ladder is
registry-governed data — an ordered list of ``StageOption`` ids on the preset — and the resolver
maps it to provider values and writes ``<stage>_fallback_providers`` into the profile. These tests
prove the mapping is correct and that a cloud-primary stage is not handed a redundant fallback.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.model_registry import (
    get_diarization_option,
    get_summary_option,
    get_transcription_option,
    resolve_profile_to_settings,
)

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

# The three DGX prod presets promoted to MOSS transcription (#1174) share one ladder shape:
# on-prem tier(s) first, then the cloud_balanced tier for that stage.
_DGX_PRESETS = ["cloud_with_dgx_primary", "prod_dgx_full_with_fallback", "prod_dgx_balanced"]


@pytest.mark.parametrize("name", _DGX_PRESETS)
def test_transcription_ladder_is_moss_then_dgx_whisper_then_cloud(name: str) -> None:
    """MOSS down -> DGX faster-whisper (same box, no cloud egress) -> cloud whisper."""
    resolved = resolve_profile_to_settings(name)
    assert (
        resolved["transcription_provider"]
        == get_transcription_option("moss_transcribe_diarize").provider
    )
    assert resolved["transcription_fallback_providers"] == ["tailnet_dgx_whisper", "openai"]


@pytest.mark.parametrize("name", _DGX_PRESETS)
def test_diarization_ladder_is_dgx_pyannote_then_deepgram(name: str) -> None:
    resolved = resolve_profile_to_settings(name)
    assert resolved["diarization_fallback_providers"] == ["deepgram"]


def test_cloud_summary_stage_gets_no_fallback() -> None:
    """cloud_with_dgx_primary already summarises in the cloud (gemini) — a fallback would be a
    redundant same-tier hop, so the resolver emits none."""
    resolved = resolve_profile_to_settings("cloud_with_dgx_primary")
    assert "summary_fallback_providers" not in resolved


@pytest.mark.parametrize("name", ["prod_dgx_full_with_fallback", "prod_dgx_balanced"])
def test_dgx_llm_summary_falls_back_to_cloud(name: str) -> None:
    """A DGX-served (vLLM) summary stage degrades to the cloud_balanced summary tier when the box
    is unreachable — the one thing an all-DGX LLM stage could not do before RFC-105."""
    resolved = resolve_profile_to_settings(name)
    assert resolved["summary_fallback_providers"] == ["gemini"]


def test_a_preset_with_no_ladder_emits_no_fallback_keys() -> None:
    """cloud_balanced carries no ``*_fallback`` tuples, so the resolver must not invent chains."""
    resolved = resolve_profile_to_settings("cloud_balanced")
    for key in (
        "transcription_fallback_providers",
        "diarization_fallback_providers",
        "summary_fallback_providers",
    ):
        assert key not in resolved


def test_the_emitted_chain_is_the_stage_options_provider_value() -> None:
    """The chain in the profile is provider strings, not StageOption ids — the ids are an internal
    handle; a Config/runtime consumer sees the provider it will actually construct."""
    resolved = resolve_profile_to_settings("prod_dgx_full_with_fallback")
    assert resolved["transcription_fallback_providers"] == [
        get_transcription_option("tailnet_dgx_speaches_thread_b").provider,
        get_transcription_option("openai_whisper_1").provider,
    ]
    assert resolved["diarization_fallback_providers"] == [
        get_diarization_option("deepgram_diarization_nova3").provider,
    ]
    assert resolved["summary_fallback_providers"] == [
        get_summary_option("gemini_flash_lite").provider,
    ]
