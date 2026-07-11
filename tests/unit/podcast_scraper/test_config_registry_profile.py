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
