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
    get_clustering_option,
    get_clustering_options,
    get_gi_options,
    get_kg_option,
    get_kg_options,
    get_ner_option,
    get_ner_options,
    get_profile_preset,
    get_summary_option,
    get_summary_options,
    get_transcription_option,
    get_transcription_options,
    ProfilePreset,
    resolve_endpoint,
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
        for opts in (
            get_transcription_options(),
            get_summary_options(),
            get_kg_options(),
            get_ner_options(),
            get_clustering_options(),
        ):
            for opt in opts.values():
                assert opt.research_ref, f"{opt.option_id} missing research_ref"
                assert opt.headline_metric, f"{opt.option_id} missing headline_metric"

    def test_every_option_has_valid_tier(self) -> None:
        valid_tiers = {"primary", "fallback", "experimental", "deprecated"}
        for opts in (
            get_transcription_options(),
            get_summary_options(),
            get_kg_options(),
            get_ner_options(),
            get_clustering_options(),
        ):
            for opt in opts.values():
                assert opt.tier in valid_tiers, f"{opt.option_id}: bad tier {opt.tier!r}"

    def test_unknown_option_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transcription option"):
            get_transcription_option("not-a-real-option")
        with pytest.raises(ValueError, match="Unknown summary option"):
            get_summary_option("not-a-real-option")
        with pytest.raises(ValueError, match="Unknown KG option"):
            get_kg_option("not-a-real-option")
        with pytest.raises(ValueError, match="Unknown NER option"):
            get_ner_option("not-a-real-option")
        with pytest.raises(ValueError, match="Unknown clustering option"):
            get_clustering_option("not-a-real-option")


class TestPipelineStageRegistries:
    """GI / KG / NER / clustering materialized from #853 + #904 + #906 + #978 reports.

    Every pipeline stage now has at least one StageOption with research provenance.
    """

    def test_gi_registry_has_v2_winner(self) -> None:
        opts = get_gi_options()
        assert "provider_n12_grounded_bundled" in opts
        assert opts["provider_n12_grounded_bundled"].extra_settings == {
            "max_insights": 12,
            "require_grounding": True,
            "evidence_quote_mode": "bundled",
            "evidence_nli_mode": "bundled",
        }

    def test_kg_registry_has_provider_default(self) -> None:
        opts = get_kg_options()
        assert "provider_n10_15" in opts
        assert "summary_bullets_n10_15" not in opts  # removed in #1034

    def test_ner_registry_covers_cloud_and_local(self) -> None:
        opts = get_ner_options()
        assert "gemini_speaker_detector" in opts
        assert "spacy_trf" in opts
        assert "spacy_sm" in opts

    def test_clustering_registry_has_pareto_default(self) -> None:
        opts = get_clustering_options()
        assert "topic_clusters_default_0_75" in opts
        assert opts["topic_clusters_default_0_75"].extra_settings == {"threshold": 0.75}


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
        all_kg = set(get_kg_options())
        all_ner = set(get_ner_options())
        all_clustering = set(get_clustering_options())
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
            assert (
                preset.summary in all_sm
            ), f"profile {name!r} references unknown summary option {preset.summary!r}"
            assert (
                preset.kg in all_kg
            ), f"profile {name!r} references unknown KG option {preset.kg!r}"
            assert (
                preset.ner in all_ner
            ), f"profile {name!r} references unknown NER option {preset.ner!r}"
            assert (
                preset.clustering in all_clustering
            ), f"profile {name!r} references unknown clustering option {preset.clustering!r}"

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


class TestResolveEndpoint:
    """Covers the {dgx_tailnet_host} template substitution paths.

    Endpoints in StageOption are templates so the operator's specific Tailscale
    MagicDNS hostname stays out of the repo (see #975 sanitization). At read
    time the caller resolves the template via this helper.
    """

    def test_none_template_returns_none(self) -> None:
        assert resolve_endpoint(None) is None
        assert resolve_endpoint(None, dgx_tailnet_host="anything") is None

    def test_template_without_placeholder_passes_through(self) -> None:
        # Some endpoints (cloud APIs) don't need DGX host substitution.
        assert resolve_endpoint("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_explicit_host_arg_wins(self) -> None:
        url = resolve_endpoint(
            "http://{dgx_tailnet_host}:8002/v1/audio/transcriptions",
            dgx_tailnet_host="my-dgx.example.ts.net",
        )
        assert url == "http://my-dgx.example.ts.net:8002/v1/audio/transcriptions"

    def test_env_var_used_when_no_explicit_arg(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("DGX_TAILNET_HOST", "env-dgx.example.ts.net")
        url = resolve_endpoint("http://{dgx_tailnet_host}:11434/v1")
        assert url == "http://env-dgx.example.ts.net:11434/v1"

    def test_sentinel_fallback_when_nothing_set(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Clear env var if it happens to be set in the test environment.
        monkeypatch.delenv("DGX_TAILNET_HOST", raising=False)
        url = resolve_endpoint("http://{dgx_tailnet_host}:8003/v1")
        # Sentinel-laced URL fails downstream HTTP cleanly rather than silently
        # routing to a placeholder hostname.
        assert url is not None
        assert "REPLACE_ME_DGX_TAILNET_HOST" in url

    def test_explicit_arg_overrides_env_var(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("DGX_TAILNET_HOST", "env-dgx.example.ts.net")
        url = resolve_endpoint(
            "http://{dgx_tailnet_host}:8000/v1/audio/transcriptions",
            dgx_tailnet_host="explicit-dgx.example.ts.net",
        )
        assert url is not None
        assert "explicit-dgx" in url
        assert "env-dgx" not in url


class TestResolveProfileToSettingsHostThreading:
    """Covers the dgx_tailnet_host arg added to resolve_profile_to_settings()."""

    def test_explicit_host_arg_threads_through(self) -> None:
        settings = resolve_profile_to_settings(
            "cloud_with_dgx_primary",
            dgx_tailnet_host="my-dgx.example.ts.net",
        )
        # Both transcription + summary endpoints (when present) should have
        # the host substituted. Use urlparse for anchored hostname check —
        # CodeQL flags bare substring containment as py/incomplete-url-substring-sanitization.
        from urllib.parse import urlparse

        if "transcription_endpoint" in settings:
            assert urlparse(settings["transcription_endpoint"]).hostname == "my-dgx.example.ts.net"
            assert "{dgx_tailnet_host}" not in settings["transcription_endpoint"]
        if "summary_endpoint" in settings:
            assert "{dgx_tailnet_host}" not in settings["summary_endpoint"]

    def test_no_host_falls_back_to_sentinel(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("DGX_TAILNET_HOST", raising=False)
        settings = resolve_profile_to_settings("cloud_with_dgx_primary")
        # Endpoints carry the sentinel so downstream HTTP fails fast rather
        # than silently routing to a placeholder hostname.
        if "transcription_endpoint" in settings:
            assert "REPLACE_ME_DGX_TAILNET_HOST" in settings["transcription_endpoint"]

    def test_env_var_used_when_no_explicit_host(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from urllib.parse import urlparse

        monkeypatch.setenv("DGX_TAILNET_HOST", "env-dgx.example.ts.net")
        settings = resolve_profile_to_settings("cloud_with_dgx_primary")
        if "transcription_endpoint" in settings:
            # Anchored hostname check (CodeQL: py/incomplete-url-substring-sanitization
            # flags bare ``host in url`` patterns).
            assert urlparse(settings["transcription_endpoint"]).hostname == "env-dgx.example.ts.net"
