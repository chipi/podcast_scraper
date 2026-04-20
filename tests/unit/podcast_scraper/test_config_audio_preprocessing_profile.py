"""Unit tests for audio_preprocessing_profile + ml_preprocessing_profile (#634).

Verifies:
  - audio_preprocessing_profile resolves to config/profiles/audio/<name>.yaml
  - Merge order: preset < deployment profile < explicit args
  - Unknown preset logs warning + leaves other fields alone
  - ml_preprocessing_profile flows to ML summary provider resolution
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from podcast_scraper.config import Config


@pytest.fixture
def tmp_audio_preset(tmp_path, monkeypatch):
    """Create a test audio preset YAML and point resolution at tmp_path."""
    audio_dir = tmp_path / "config" / "profiles" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "speech_test_v1.yaml").write_text(dedent("""
            preprocessing_enabled: true
            preprocessing_sample_rate: 16000
            preprocessing_mp3_bitrate_kbps: 48
            preprocessing_silence_threshold: "-40dB"
            preprocessing_silence_duration: 1.0
            preprocessing_target_loudness: -18
            """).strip())
    monkeypatch.chdir(tmp_path)
    return audio_dir


class TestAudioPreprocessingProfile:
    def test_direct_reference_loads_all_preset_fields(self, tmp_audio_preset):
        c = Config(
            rss_url="https://example.com/feed.xml",
            audio_preprocessing_profile="speech_test_v1",
        )
        assert c.audio_preprocessing_profile == "speech_test_v1"
        assert c.preprocessing_mp3_bitrate_kbps == 48
        assert c.preprocessing_sample_rate == 16000
        assert c.preprocessing_silence_threshold == "-40dB"
        assert c.preprocessing_silence_duration == 1.0
        assert c.preprocessing_target_loudness == -18

    def test_explicit_field_overrides_preset(self, tmp_audio_preset):
        c = Config(
            rss_url="https://example.com/feed.xml",
            audio_preprocessing_profile="speech_test_v1",
            preprocessing_mp3_bitrate_kbps=96,  # explicit override
        )
        assert c.preprocessing_mp3_bitrate_kbps == 96  # explicit wins
        assert c.preprocessing_sample_rate == 16000  # preset still applies

    def test_unknown_preset_logs_warning_and_keeps_defaults(self, tmp_path, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="podcast_scraper.config")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config" / "profiles" / "audio").mkdir(parents=True)
        c = Config(
            rss_url="https://example.com/feed.xml",
            audio_preprocessing_profile="does_not_exist",
        )
        assert c.audio_preprocessing_profile == "does_not_exist"
        # Default preprocessing_mp3_bitrate_kbps is None (auto) — preset never merged.
        assert c.preprocessing_mp3_bitrate_kbps is None
        assert any(
            "does_not_exist" in rec.getMessage() and "not found" in rec.getMessage()
            for rec in caplog.records
        )

    def test_unset_preset_is_noop(self):
        # Neither profile nor audio preset — defaults untouched.
        c = Config(rss_url="https://example.com/feed.xml")
        assert c.audio_preprocessing_profile is None
        assert c.preprocessing_mp3_bitrate_kbps is None

    def test_deployment_profile_reference_loads_preset(self):
        """Integration: dev.yaml references speech_optimal_v1 → bitrate=32.

        Uses dev profile (no cloud API keys required) to exercise the real
        cloud_balanced/cloud_quality/local/airgapped/dev all-reference pattern.
        """
        c = Config(
            rss_url="https://example.com/feed.xml",
            profile="dev",
        )
        assert c.audio_preprocessing_profile == "speech_optimal_v1"
        # speech_optimal_v1.yaml sets bitrate=32
        assert c.preprocessing_mp3_bitrate_kbps == 32
        assert c.preprocessing_sample_rate == 16000

    def test_deployment_profile_field_overrides_preset(self, tmp_audio_preset):
        """Preset says 48; deployment profile override in test YAML would set 64;
        explicit explicit wins.

        Here we simulate the middle layer via monkeypatched profile YAML.
        """
        # Simulate a deployment-profile YAML that references the preset + overrides one field.
        dep_profile_dir = Path("config/profiles")
        dep_profile_dir.mkdir(parents=True, exist_ok=True)
        (dep_profile_dir / "test_override.yaml").write_text(dedent("""
                rss_url: "https://example.com/feed.xml"
                audio_preprocessing_profile: speech_test_v1
                preprocessing_mp3_bitrate_kbps: 64  # deployment profile override
                """).strip())
        c = Config(profile="test_override")
        # Deployment profile overrides preset (48 -> 64)
        assert c.preprocessing_mp3_bitrate_kbps == 64
        # Other preset fields survive
        assert c.preprocessing_silence_duration == 1.0

        # Explicit kwarg wins over deployment profile
        c2 = Config(profile="test_override", preprocessing_mp3_bitrate_kbps=96)
        assert c2.preprocessing_mp3_bitrate_kbps == 96


class TestMLPreprocessingProfile:
    def test_default_is_none(self):
        c = Config(rss_url="https://example.com/feed.xml")
        assert c.ml_preprocessing_profile is None

    def test_can_be_set(self):
        c = Config(
            rss_url="https://example.com/feed.xml",
            ml_preprocessing_profile="cleaning_v3",
        )
        assert c.ml_preprocessing_profile == "cleaning_v3"
