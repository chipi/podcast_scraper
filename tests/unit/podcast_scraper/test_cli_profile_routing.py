"""CLI profile-routing regression tests (#646).

Covers two pre-existing bugs exposed by the real-episode validation:

- Bug A: ``--profile NAME`` had a separate (broken) code path that did NOT
  go through ``_load_and_merge_config``. argparse defaults (e.g.
  ``--summary-provider`` default ``"transformers"``) silently overrode
  profile values. Users of ``cloud_balanced`` / ``cloud_quality`` got
  13 wrong fields.
- Bug B: ``_build_config``'s payload dict did not forward fields that
  only existed in the profile YAML (no argparse flag), e.g.
  ``llm_pipeline_mode``, ``cloud_llm_structured_min_output_tokens``,
  ``audio_preprocessing_profile``. ``--config <profile.yaml>`` worked
  for argparse-known fields but silently dropped these.

Both paths now route through ``_load_and_merge_config`` and include the
missing fields conditionally in payload.
"""

from __future__ import annotations

import pytest

from podcast_scraper.cli import _build_config, parse_args


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROK_API_KEY",
    ):
        monkeypatch.setenv(name, "test-" + name.lower().replace("_", "-") + "-dummy-key")


CLOUD_BALANCED_EXPECTED = {
    "transcription_provider": "openai",
    "openai_transcription_model": "whisper-1",
    "audio_preprocessing_profile": "speech_optimal_v1",
    "speaker_detector_provider": "spacy",
    "ner_model": "en_core_web_trf",
    "auto_speakers": True,
    "summary_provider": "gemini",
    "gemini_summary_model": "gemini-2.5-flash-lite",
    "preprocessing_enabled": True,
    "llm_pipeline_mode": "mega_bundled",
    "cloud_llm_structured_min_output_tokens": 4096,
    "generate_gi": True,
    "gi_insight_source": "provider",
    "gi_max_insights": 12,
    "gi_require_grounding": True,
    "generate_kg": True,
    "kg_extraction_source": "provider",
    "kg_max_topics": 10,
    "kg_max_entities": 15,
    "vector_search": True,
    "vector_backend": "faiss",
    "generate_metadata": True,
    "generate_summaries": True,
    "preprocessing_silence_threshold": "-30dB",
    "preprocessing_silence_duration": 0.5,
    "preprocessing_mp3_bitrate_kbps": 32,
    "preprocessing_sample_rate": 16000,
    "preprocessing_target_loudness": -16,
}

CLOUD_QUALITY_OVERRIDES = {
    "summary_provider": "anthropic",
    "anthropic_summary_model": "claude-haiku-4-5",
    "llm_pipeline_mode": "mega_bundled",
    "cloud_llm_structured_min_output_tokens": 4096,
    "audio_preprocessing_profile": "speech_optimal_v1",
}


def _build_via_profile(name: str) -> object:
    args = parse_args(
        ["--profile", name, "https://example.com/feed.xml", "--output-dir", "/tmp/_t"]
    )
    return _build_config(args)


def _build_via_config(path: str) -> object:
    args = parse_args(["--config", path, "https://example.com/feed.xml", "--output-dir", "/tmp/_t"])
    return _build_config(args)


class TestProfileFlagRouting:
    """Bug A regression: --profile NAME must produce every field from the YAML."""

    def test_profile_cloud_balanced_matches_yaml(self, _fake_keys: None) -> None:
        cfg = _build_via_profile("cloud_balanced")
        for key, expected in CLOUD_BALANCED_EXPECTED.items():
            actual = getattr(cfg, key, None)
            assert actual == expected, f"{key}: expected={expected!r} actual={actual!r}"

    def test_profile_cloud_quality_matches_yaml(self, _fake_keys: None) -> None:
        cfg = _build_via_profile("cloud_quality")
        for key, expected in CLOUD_QUALITY_OVERRIDES.items():
            actual = getattr(cfg, key, None)
            assert actual == expected, f"{key}: expected={expected!r} actual={actual!r}"

    def test_explicit_cli_flag_overrides_profile(self, _fake_keys: None) -> None:
        """User's explicit --summary-provider beats profile setting."""
        args = parse_args(
            [
                "--profile",
                "cloud_balanced",
                "--summary-provider",
                "deepseek",
                "https://example.com/feed.xml",
                "--output-dir",
                "/tmp/_t",
            ]
        )
        cfg = _build_config(args)
        assert cfg.summary_provider == "deepseek"


class TestConfigFlagRouting:
    """Bug B regression: --config YAML must forward all profile fields."""

    def test_config_cloud_balanced_matches_yaml(self, _fake_keys: None) -> None:
        cfg = _build_via_config("config/profiles/cloud_balanced.yaml")
        for key, expected in CLOUD_BALANCED_EXPECTED.items():
            actual = getattr(cfg, key, None)
            assert actual == expected, f"{key}: expected={expected!r} actual={actual!r}"

    def test_config_preserves_llm_pipeline_mode(self, _fake_keys: None) -> None:
        """Previously dropped because _build_config payload omitted it."""
        cfg = _build_via_config("config/profiles/cloud_balanced.yaml")
        assert cfg.llm_pipeline_mode == "mega_bundled"

    def test_config_preserves_audio_preprocessing_profile(self, _fake_keys: None) -> None:
        cfg = _build_via_config("config/profiles/cloud_balanced.yaml")
        assert cfg.audio_preprocessing_profile == "speech_optimal_v1"

    def test_config_preserves_cloud_llm_structured_min_output_tokens(
        self, _fake_keys: None
    ) -> None:
        cfg = _build_via_config("config/profiles/cloud_balanced.yaml")
        assert cfg.cloud_llm_structured_min_output_tokens == 4096


class TestBothPathsAgree:
    """--profile NAME and --config config/profiles/NAME.yaml must produce
    identical Config objects (they routed through different code paths before
    #646; now both go through _load_and_merge_config)."""

    def test_profile_and_config_produce_same_config(self, _fake_keys: None) -> None:
        via_profile = _build_via_profile("cloud_balanced")
        via_config = _build_via_config("config/profiles/cloud_balanced.yaml")
        # Compare every key present in the expected table.
        for key in CLOUD_BALANCED_EXPECTED:
            assert getattr(via_profile, key, None) == getattr(via_config, key, None), key
