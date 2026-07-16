"""Registry-layered profile resolution on ``Config`` (#907 Option B).

Covers the 2026-06-12 amendment to ``Config._resolve_profile`` that layers
``resolve_profile_to_settings()`` UNDER the existing YAML-profile loader.

Resolution order (highest wins):
  1. Explicit fields in the caller's dict / CLI args.
  2. Profile YAML at ``config/profiles/<name>.yaml``.
  3. Registry preset (``model_registry._PROFILE_PRESETS``) — research-driven
     defaults filtered to Config field names.

These tests pin the precedence semantics and the field-filter behaviour
that prevents ``extra="forbid"`` from rejecting resolver-only keys.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.ml.model_registry import (
    get_diarization_option,
    get_diarization_options,
    get_profile_preset,
    resolve_profile_to_settings,
)


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPGRAM_API_KEY",
    ):
        monkeypatch.setenv(name, "test-dummy")


class TestRegistryLayering:
    """The registry resolver fills in routing fields when the YAML omits them."""

    def test_registry_routing_applies_under_yaml(self, _fake_keys: None) -> None:
        """Loading by profile name surfaces registry-driven routing.

        ``cloud_with_dgx_primary`` is in ``_PROFILE_PRESETS``; its
        ``transcription_provider`` should be ``tailnet_dgx_whisper`` per
        the registry, regardless of what the YAML happens to set.
        """
        cfg = Config.model_validate({"profile": "cloud_with_dgx_primary"})
        # Either the YAML or the registry must provide this; both currently
        # agree, so we assert the post-merge value (registry-driven default
        # when the YAML is empty, YAML override when explicit).
        assert cfg.transcription_provider == "tailnet_dgx_whisper"

    def test_explicit_field_overrides_registry(self, _fake_keys: None) -> None:
        """Explicit data wins over registry defaults (layer 1 > layer 3).

        Uses ``cloud_balanced`` (flat ``transcription_provider:`` in the YAML)
        rather than ``cloud_with_dgx_primary`` because the latter uses a
        nested ``transcription: {primary, fallback}`` block that gets
        flattened AFTER the explicit-data merge — masking the precedence
        signal we want to assert.
        """
        # cloud_balanced registry default for summary is gemini-2.5-flash-lite;
        # override to a different gemini model to exercise the precedence path.
        cfg = Config.model_validate(
            {
                "profile": "cloud_balanced",
                "summary_provider": "openai",  # override registry's 'gemini'
            }
        )
        assert cfg.summary_provider == "openai"

    def test_unknown_profile_name_is_warned_not_errored(
        self, _fake_keys: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A profile name absent from BOTH registry and config/profiles/
        logs a warning and falls back to Pydantic field defaults."""
        with caplog.at_level("WARNING"):
            cfg = Config.model_validate({"profile": "definitely-not-a-real-profile"})
        assert "not found" in caplog.text.lower()
        # Field defaults still apply; Config still constructs cleanly.
        assert cfg is not None

    def test_resolver_only_keys_are_dropped_not_rejected(self, _fake_keys: None) -> None:
        """Resolver emits keys like ``transcription_endpoint`` that Config
        doesn't have. Those must be filtered out before Pydantic sees them
        (Config has ``extra="forbid"``)."""
        # Construction must not raise.
        cfg = Config.model_validate({"profile": "cloud_with_dgx_primary"})
        assert not hasattr(cfg, "transcription_endpoint")


class TestStageOptionRegistries:
    """Diarization registry — provenance + tier coverage missing from integration suite."""

    def test_every_option_has_research_provenance(self) -> None:
        """Every diarization StageOption must cite the eval report that justified it."""
        for opts in (get_diarization_options(),):
            for opt in opts.values():
                assert opt.research_ref, f"{opt.option_id} missing research_ref"
                assert opt.headline_metric, f"{opt.option_id} missing headline_metric"

    def test_every_option_has_valid_tier(self) -> None:
        valid_tiers = {"primary", "fallback", "experimental", "deprecated"}
        for opts in (get_diarization_options(),):
            for opt in opts.values():
                assert opt.tier in valid_tiers, f"{opt.option_id}: bad tier {opt.tier!r}"

    def test_unknown_option_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown diarization option"):
            get_diarization_option("not-a-real-option")


class TestProfilePresets:
    """Drift catch: preset diarization fields must reference registered diarization options."""

    def test_every_preset_references_real_options(self) -> None:
        all_dia = set(get_diarization_options())
        for name in [
            "cloud_balanced",
            "cloud_thin",
            "cloud_with_dgx_primary",
            "local_dgx_balanced",
            "local_dgx_full",
        ]:
            preset = get_profile_preset(name)
            assert preset.diarization in all_dia, (
                f"profile {name!r} references unknown diarization option " f"{preset.diarization!r}"
            )

    def test_resolver_routes_diarization_model_by_backend(self) -> None:
        """resolve_profile_to_settings routes the diarization model to the config field
        for the option's backend (pyannote->diarization_model, tailnet_dgx->
        dgx_diarize_model, deepgram->deepgram_diarization_model) + a research ref."""
        # local -> in-process pyannote
        s = resolve_profile_to_settings("local", dgx_tailnet_host="h")
        assert s["diarization_model"] == "pyannote/speaker-diarization-community-1"
        assert "dgx_diarize_model" not in s and "deepgram_diarization_model" not in s
        assert s.get("_diarization_research_ref")
        # eval_default -> DGX diarize service
        s = resolve_profile_to_settings("eval_default", dgx_tailnet_host="h")
        assert s["dgx_diarize_model"] == "pyannote/speaker-diarization-community-1"
        assert "diarization_model" not in s and "deepgram_diarization_model" not in s
        # cloud_balanced -> standalone Deepgram pass
        s = resolve_profile_to_settings("cloud_balanced", dgx_tailnet_host="h")
        assert s["deepgram_diarization_model"] == "nova-3-general"
        assert "diarization_model" not in s and "dgx_diarize_model" not in s


class TestARegisteredParamMustReachProduction:
    """A param the registry records but never plumbs is a setting production silently does not run.

    Found by re-breaking the fix: the resolver raises ``RuntimeError`` on an unmapped GI setting,
    and that exception TYPE is load-bearing — ``Config._resolve_profile`` catches ``ValueError`` to
    mean "not a registry preset" and quietly drops to YAML-only. Downgrade the raise to
    ``ValueError`` and a single typo disables the whole registry for that profile without a word,
    which is the exact silent-fallback class of bug the registry exists to prevent. Nothing caught
    that downgrade, so nothing stopped it from happening again.
    """

    def test_an_unmapped_gi_setting_raises_RUNTIME_error_not_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        from podcast_scraper.providers.ml import model_registry as reg

        preset = reg.get_profile_preset("experiment_dgx_only")
        original = reg.get_gi_option(preset.gi)
        poisoned = dataclasses.replace(
            original,
            extra_settings={**(original.extra_settings or {}), "a_knob_nobody_plumbed": 1},
        )
        monkeypatch.setitem(reg._GI_OPTIONS, preset.gi, poisoned)

        # ValueError would be swallowed by Config._resolve_profile and the profile would silently
        # run on YAML defaults. RuntimeError is what makes it impossible to miss.
        with pytest.raises(RuntimeError, match="a_knob_nobody_plumbed"):
            resolve_profile_to_settings("experiment_dgx_only")

    def test_every_registered_gi_param_is_actually_plumbed(self) -> None:
        """The completeness half: no shipped option may carry a param the resolver cannot map.

        The test above proves an unmapped param fails LOUDLY. This proves none exists today — so
        the registry's recorded params and the settings production runs cannot drift apart.
        """
        from podcast_scraper.providers.ml.model_registry import (
            _PROFILE_PRESETS,
            get_gi_options,
        )

        for name, preset in _PROFILE_PRESETS.items():
            if not preset.gi:
                continue
            option = get_gi_options()[preset.gi]
            for key in option.extra_settings or {}:
                resolved = resolve_profile_to_settings(name)
                assert resolved, f"{name}: resolved to nothing"
                # If `key` were unmapped, resolve_profile_to_settings would have raised above.
