"""Tests for the pipeline-stage expansion of the model registry.

Covers the 2026-06-12 amendment to ADR-048 that added StageOption /
ProfilePreset / _TRANSCRIPTION_OPTIONS / _SUMMARY_OPTIONS / _PROFILE_PRESETS
to `model_registry.py`. The point of these tests is **drift detection**:
profile YAMLs must match the registry presets they claim to derive from;
when they don't, the eval reports document decisions the runtime never
adopted.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.model_registry import (
    get_profile_preset,
    get_summary_option,
    get_summary_options,
    get_transcription_option,
    get_transcription_options,
    ProfilePreset,
    resolve_profile_to_settings,
)


class TestStageOptionRegistries:
    def test_transcription_options_are_populated(self) -> None:
        opts = get_transcription_options()
        assert len(opts) >= 4
        # Production-critical entries we know exist as of 2026-06-12.
        assert "openai_whisper_1" in opts
        assert "tailnet_dgx_whisper_openai" in opts
        assert "tailnet_dgx_speaches_thread_b" in opts
        assert "local_mps_large_v3" in opts

    def test_summary_options_are_populated(self) -> None:
        opts = get_summary_options()
        assert len(opts) >= 5
        assert "gemini_flash_lite" in opts
        assert "ollama_qwen35_35b" in opts
        assert "ollama_qwen35_9b" in opts

    def test_every_option_has_research_provenance(self) -> None:
        """Every StageOption must cite the eval report that justified it."""
        for opt in get_transcription_options().values():
            assert opt.research_ref, f"{opt.option_id} missing research_ref"
            assert opt.headline_metric, f"{opt.option_id} missing headline_metric"
        for opt in get_summary_options().values():
            assert opt.research_ref, f"{opt.option_id} missing research_ref"
            assert opt.headline_metric, f"{opt.option_id} missing headline_metric"

    def test_every_option_has_valid_tier(self) -> None:
        valid_tiers = {"primary", "fallback", "experimental", "deprecated"}
        for opt in get_transcription_options().values():
            assert opt.tier in valid_tiers, f"{opt.option_id}: bad tier {opt.tier!r}"
        for opt in get_summary_options().values():
            assert opt.tier in valid_tiers, f"{opt.option_id}: bad tier {opt.tier!r}"

    def test_unknown_option_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transcription option"):
            get_transcription_option("not-a-real-option")
        with pytest.raises(ValueError, match="Unknown summary option"):
            get_summary_option("not-a-real-option")


class TestProfilePresets:
    def test_known_presets_are_registered(self) -> None:
        known = {
            "cloud_balanced",
            "cloud_thin",
            "cloud_with_dgx_primary",
            "local_dgx_balanced",
            "local_dgx_full",
        }
        for name in known:
            preset = get_profile_preset(name)
            assert isinstance(preset, ProfilePreset)
            assert preset.name == name

    def test_every_preset_references_real_options(self) -> None:
        """Drift catch: every preset's stage choices must exist in the option registry."""
        all_tx = set(get_transcription_options())
        all_sm = set(get_summary_options())
        for name in [
            "cloud_balanced",
            "cloud_thin",
            "cloud_with_dgx_primary",
            "local_dgx_balanced",
            "local_dgx_full",
        ]:
            preset = get_profile_preset(name)
            assert preset.transcription in all_tx, (
                f"profile {name!r} references unknown transcription option "
                f"{preset.transcription!r}"
            )
            assert preset.summary in all_sm, (
                f"profile {name!r} references unknown summary option " f"{preset.summary!r}"
            )

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile preset"):
            get_profile_preset("not-a-real-profile")


class TestResolveProfileToSettings:
    def test_cloud_with_dgx_primary_resolves_to_whisper_openai_8002(self) -> None:
        """Post-#968 Thread B verdict: cloud_with_dgx_primary uses whisper-openai (:8002),
        not speaches (:8000). This test catches accidental regression to the old
        speaches routing."""
        settings = resolve_profile_to_settings("cloud_with_dgx_primary")
        assert settings["transcription_provider"] == "tailnet_dgx_whisper"
        assert settings["transcription_model"] == "large-v3"
        assert "8002" in settings["transcription_endpoint"]
        # NOT speaches:
        assert "Systran/faster-whisper" not in settings["transcription_model"]
        assert "8000" not in settings["transcription_endpoint"]

    def test_local_dgx_balanced_resolves_to_qwen35_35b(self) -> None:
        """Post-#928 + #958 Cell D verdict: local DGX summary is qwen3.5:35b, not 9b."""
        settings = resolve_profile_to_settings("local_dgx_balanced")
        assert settings["summary_provider"] == "ollama"
        assert settings["summary_model"] == "qwen3.5:35b"

    def test_cloud_balanced_resolves_to_gemini_flash_lite(self) -> None:
        settings = resolve_profile_to_settings("cloud_balanced")
        assert settings["summary_provider"] == "gemini"
        assert settings["summary_model"] == "gemini-2.5-flash-lite"

    def test_resolved_settings_carry_research_refs(self) -> None:
        """Every resolved profile must surface the research provenance for traceability."""
        settings = resolve_profile_to_settings("cloud_with_dgx_primary")
        assert settings["_transcription_research_ref"]
        assert settings["_summary_research_ref"]
        assert settings["_profile_preset"] == "cloud_with_dgx_primary"
