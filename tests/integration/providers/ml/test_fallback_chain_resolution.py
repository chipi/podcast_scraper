"""RFC-106 (#1198): the registry owns the per-stage failover ladder, and the resolver emits it.

The bug this guards: promoting MOSS as the DGX transcription default (#1174) silently dropped the
fallback, because the ``moss`` provider has none of its own. Under RFC-106 the ladder is
registry-governed data — an ordered list of ``StageOption`` ids on the preset — and the resolver
maps it to provider values and writes ``<stage>_fallback_providers`` into the profile. These tests
prove the mapping is correct and that a cloud-primary stage is not handed a redundant fallback.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.model_registry import (
    _emit_fallback_chains,
    _is_cloud_option,
    _PROFILE_PRESETS,
    get_diarization_option,
    get_summary_option,
    get_transcription_option,
    ProfilePreset,
    resolve_profile_to_settings,
)

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

# The three DGX prod presets promoted to MOSS transcription (#1174) share one ladder shape:
# on-prem tier(s) first, then the cloud_balanced tier for that stage.
_DGX_PRESETS = ["cloud_with_dgx_primary", "prod_dgx_full_with_fallback", "prod_dgx_balanced"]


@pytest.mark.parametrize("name", _DGX_PRESETS)
def test_transcription_ladder_prefers_free_tiers_before_paid_cloud(name: str) -> None:
    """MOSS -> DGX faster-whisper -> local in-process whisper -> cloud whisper: the free/on-prem
    tiers are exhausted before the ladder pays for openai."""
    resolved = resolve_profile_to_settings(name)
    assert (
        resolved["transcription_provider"]
        == get_transcription_option("moss_transcribe_diarize").provider
    )
    assert resolved["transcription_fallback_providers"] == [
        "tailnet_dgx_whisper",
        "whisper",
        "openai",
    ]


@pytest.mark.parametrize("name", _DGX_PRESETS)
def test_diarization_ladder_is_dgx_then_local_pyannote_then_deepgram(name: str) -> None:
    """DGX pyannote -> local in-process pyannote (free) -> deepgram (paid) only if local can't run."""
    resolved = resolve_profile_to_settings(name)
    assert resolved["diarization_fallback_providers"] == ["local", "deepgram"]


def test_cloud_summary_stage_gets_no_fallback() -> None:
    """cloud_with_dgx_primary already summarises in the cloud (gemini) — a fallback would be a
    redundant same-tier hop, so the resolver emits none."""
    resolved = resolve_profile_to_settings("cloud_with_dgx_primary")
    assert "summary_fallback_providers" not in resolved


@pytest.mark.parametrize("name", ["prod_dgx_full_with_fallback", "prod_dgx_balanced"])
def test_dgx_llm_summary_falls_back_to_cloud(name: str) -> None:
    """A DGX-served (vLLM) summary stage degrades to the cloud_balanced summary tier when the box
    is unreachable — the one thing an all-DGX LLM stage could not do before RFC-106."""
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
        get_transcription_option("local_mps_large_v3").provider,
        get_transcription_option("openai_whisper_1").provider,
    ]
    assert resolved["diarization_fallback_providers"] == [
        get_diarization_option("pyannote_diarization_community1").provider,
        get_diarization_option("deepgram_diarization_nova3").provider,
    ]
    assert resolved["summary_fallback_providers"] == [
        get_summary_option("gemini_flash_lite").provider,
    ]


# --- allow_cloud_fallback fail-closed (RFC-106 increment 3) ---------------------------------------


def test_is_cloud_option_classification() -> None:
    """Hosted cloud vendors are cloud; DGX/local-served options (even openai-protocol vLLM) are
    not — the endpoint decides, not the vendor name."""
    assert _is_cloud_option(get_transcription_option("openai_whisper_1")) is True
    assert _is_cloud_option(get_diarization_option("deepgram_diarization_nova3")) is True
    assert _is_cloud_option(get_summary_option("gemini_flash_lite")) is True
    # On-prem tiers.
    assert _is_cloud_option(get_transcription_option("tailnet_dgx_speaches_thread_b")) is False
    assert _is_cloud_option(get_transcription_option("local_mps_large_v3")) is False
    assert _is_cloud_option(get_diarization_option("pyannote_diarization_community1")) is False


def test_fail_closed_strips_cloud_tiers_but_keeps_on_prem() -> None:
    """A no-cloud preset with a cloud-terminated ladder emits only its on-prem tiers — the chain
    ends at the last DGX/local tier and never phones out."""
    preset = ProfilePreset(
        name="synthetic_no_cloud",
        transcription="moss_transcribe_diarize",
        summary="summllama_3_2_3b_paragraph",
        kg="provider_n10_15",
        ner="spacy_sm",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        transcription_fallback=(
            "tailnet_dgx_speaches_thread_b",
            "local_mps_large_v3",
            "openai_whisper_1",
        ),
        diarization_fallback=("pyannote_diarization_community1", "deepgram_diarization_nova3"),
        allow_cloud_fallback=False,
    )
    settings: dict = {}
    _emit_fallback_chains(preset, settings)
    # openai + deepgram (cloud) dropped; dgx-whisper, local whisper, local pyannote kept.
    assert settings["transcription_fallback_providers"] == ["tailnet_dgx_whisper", "whisper"]
    assert settings["diarization_fallback_providers"] == ["local"]


def test_allow_cloud_fallback_true_by_default_keeps_cloud() -> None:
    """The same ladder with the default (True) keeps every tier, including cloud."""
    preset = ProfilePreset(
        name="synthetic_cloud_ok",
        transcription="moss_transcribe_diarize",
        summary="summllama_3_2_3b_paragraph",
        kg="provider_n10_15",
        ner="spacy_sm",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        transcription_fallback=("tailnet_dgx_speaches_thread_b", "openai_whisper_1"),
        diarization_fallback=("deepgram_diarization_nova3",),
    )
    assert preset.allow_cloud_fallback is True
    settings: dict = {}
    _emit_fallback_chains(preset, settings)
    assert settings["transcription_fallback_providers"] == ["tailnet_dgx_whisper", "openai"]
    assert settings["diarization_fallback_providers"] == ["deepgram"]


def test_airgapped_presets_declare_no_cloud() -> None:
    """The offline presets are fail-closed so any future ladder cannot reach cloud."""
    for name in ("airgapped", "airgapped_thin"):
        assert _PROFILE_PRESETS[name].allow_cloud_fallback is False
