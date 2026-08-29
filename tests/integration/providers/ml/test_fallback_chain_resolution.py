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

# DGX prod presets run turbo (tailnet_dgx_whisper) as the transcription primary since the real-GT
# bake-off (#1178/#1179) — MOSS was demoted to an accurate-but-slow fallback. Two ladder shapes now
# diverge: cloud_with_dgx_primary terminates each stage in the cloud_balanced tier, while
# prod_dgx_full is fully airgapped (ADR-147) — its ladders end at the last DGX/local
# tier and never reach cloud (asserted separately below).
_CLOUD_TERMINATED_PRESETS = ["cloud_with_dgx_primary"]


@pytest.mark.parametrize("name", _CLOUD_TERMINATED_PRESETS)
def test_transcription_ladder_prefers_free_tiers_before_paid_cloud(name: str) -> None:
    """DGX turbo -> DGX large-v3 (coverage failover) -> local in-process whisper -> Deepgram:
    the free/on-prem tiers are exhausted before the ladder pays. The paid floor is DEEPGRAM, not
    openai (2026-08-28): Deepgram is the house cloud ASR — the provider cloud_balanced already
    runs, keys, and measures — so a DGX outage degrades onto something we compare output against
    rather than a third vendor billing unattended at 03:00. Turbo replaced MOSS as the primary
    per the #1178/#1179 bake-off; MOSS is now an accurate-but-slow fallback."""
    resolved = resolve_profile_to_settings(name)
    assert resolved["transcription_provider"] == "tailnet_dgx_whisper"
    assert resolved["transcription_fallback_providers"] == [
        "tailnet_dgx_whisper",
        "whisper",
        "deepgram",
    ]
    # cost invariant: the paid cloud tier stays LAST, after the free/on-prem tiers.
    assert resolved["transcription_fallback_providers"][-1] == "deepgram"


@pytest.mark.parametrize("name", _CLOUD_TERMINATED_PRESETS)
def test_diarization_ladder_is_dgx_then_local_pyannote_then_deepgram(name: str) -> None:
    """DGX pyannote -> local in-process pyannote (free) -> deepgram (paid) only if local can't run."""
    resolved = resolve_profile_to_settings(name)
    assert resolved["diarization_fallback_providers"] == ["local", "deepgram"]


def test_cloud_summary_stage_falls_over_to_a_different_vendor() -> None:
    """cloud_with_dgx_primary summarises in the cloud and now HAS a summary ladder (#1657).

    This test previously asserted the opposite — "already summarises in the cloud (gemini), so a
    fallback would be a redundant same-tier hop". That reasoning treated the risk as latency and
    missed the one that actually bit: a VENDOR-WIDE outage or quota exhaustion takes the whole
    stage down, and a same-tier hop to a DIFFERENT vendor is exactly the recovery. #1657 measured
    the consequence — GI and speaker detection degraded silently on an OpenRouter quota outage —
    and 46a67179 configured ladders on 11 profiles instead of 2. DeepSeek is the nearest peer tier.
    """
    resolved = resolve_profile_to_settings("cloud_with_dgx_primary")
    assert resolved["summary_fallback_providers"] == ["deepseek"]


def test_airgapped_dgx_prod_ladders_never_reach_cloud() -> None:
    """prod_dgx_full is fully airgapped (ADR-147): each stage falls back only to
    DGX/local tiers — transcription to DGX-then-local whisper, diarization to local pyannote,
    summary to DGX-local ollama — and never to a cloud vendor."""
    resolved = resolve_profile_to_settings("prod_dgx_full")
    assert resolved["transcription_provider"] == "tailnet_dgx_whisper"
    assert resolved["transcription_fallback_providers"] == ["tailnet_dgx_whisper", "whisper"]
    assert resolved["diarization_fallback_providers"] == ["local"]
    assert resolved["summary_fallback_providers"] == ["ollama"]
    for stage in ("transcription", "diarization", "summary"):
        chain = resolved[f"{stage}_fallback_providers"]
        assert not ({"openai", "deepgram", "gemini"} & set(chain)), chain


def test_a_preset_with_no_ladder_emits_no_fallback_keys() -> None:
    """``airgapped`` carries no ``*_fallback`` tuples, so the resolver must not invent chains.

    The preset under test has moved twice, and each move was a real ladder being ADDED rather than
    this guard weakening: cloud_balanced gained one in RFC-111 (#1482), then cloud_qwen gained one
    in #1657 (46a67179) — a single-vendor DashScope stack loses every stage on any Qwen outage, so
    it now falls over to direct DeepSeek.

    ``airgapped`` is the durable home for this assertion: a ladder there would have to point at a
    cloud vendor, which the profile exists to forbid. Seven presets still declare no ladder at all
    and the resolver emits nothing for every one of them, so "must not invent" is still covered.
    """
    resolved = resolve_profile_to_settings("airgapped")
    for key in (
        "transcription_fallback_providers",
        "diarization_fallback_providers",
        "summary_fallback_providers",
    ):
        assert key not in resolved


def test_the_emitted_chain_is_the_stage_options_provider_value() -> None:
    """The chain in the profile is provider strings, not StageOption ids — the ids are an internal
    handle; a Config/runtime consumer sees the provider it will actually construct. For the
    airgapped prod_dgx_full the chain ends at the last DGX/local tier."""
    resolved = resolve_profile_to_settings("prod_dgx_full")
    assert resolved["transcription_fallback_providers"] == [
        get_transcription_option("tailnet_dgx_speaches_thread_b").provider,
        get_transcription_option("local_mps_large_v3").provider,
    ]
    assert resolved["diarization_fallback_providers"] == [
        get_diarization_option("pyannote_diarization_community1").provider,
    ]
    assert resolved["summary_fallback_providers"] == [
        get_summary_option("ollama_qwen35_35b").provider,
    ]


def test_cloud_balanced_summary_ladder_fails_over_to_direct_deepseek() -> None:
    """RFC-111 (#1482): a homelab:4001 LiteLLM gateway CONNECTION outage (not a model-side 503)
    must fail over to DIRECT DeepSeek (api.deepseek.com), bypassing the dead gateway entirely —
    same model (deepseek-v4-flash) as the litellm-routed primary, just a different wire path."""
    resolved = resolve_profile_to_settings("cloud_balanced")
    assert resolved["summary_fallback_providers"] == [
        get_summary_option("deepseek_native_flash").provider,
    ]
    assert get_summary_option("deepseek_native_flash").provider == "deepseek"


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
