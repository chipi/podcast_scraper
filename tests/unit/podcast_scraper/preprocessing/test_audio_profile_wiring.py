"""Wiring test: ``audio_preprocessing_profile`` YAML value flows through
Config → factory → FFmpegAudioPreprocessor → ffmpeg cmd (#646).

Motivated by #644's Phase-3C class of bug: unit tests that construct Config
programmatically may miss silent drops on the profile → factory → subprocess
path. This test covers the full wiring end-to-end with a mocked subprocess so
it runs cheap and deterministic but exercises the real factory + ffmpeg
command builder.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from podcast_scraper import config as cfg_mod
from podcast_scraper.preprocessing.audio.factory import create_audio_preprocessor


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """cloud_balanced wires summary_provider=gemini → Config validates an api key."""
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROK_API_KEY",
    ):
        monkeypatch.setenv(name, "test-" + name.lower().replace("_", "-") + "-dummy-key")


class TestAudioPreprocessingProfileWiring:
    def test_cloud_balanced_resolves_speech_optimal_v1_to_ffmpeg_args(
        self, _fake_keys: None, tmp_path
    ) -> None:
        data = cfg_mod.load_config_file("config/profiles/cloud_balanced.yaml")
        cfg = cfg_mod.Config.model_validate({**data, "rss_url": "https://example.com/feed.xml"})

        # Confirm profile resolution happens.
        assert cfg.audio_preprocessing_profile == "speech_optimal_v1"
        assert cfg.preprocessing_silence_threshold == "-30dB"
        assert cfg.preprocessing_silence_duration == 0.5
        assert cfg.preprocessing_mp3_bitrate_kbps == 32
        assert cfg.preprocessing_sample_rate == 16000
        assert cfg.preprocessing_target_loudness == -16

        prep = create_audio_preprocessor(cfg)
        assert prep is not None
        assert prep.silence_threshold == "-30dB"
        assert prep.silence_duration == 0.5
        assert prep.mp3_bitrate_kbps == 32

        # Capture the actual ffmpeg cmd.
        with (
            patch(
                "podcast_scraper.preprocessing.audio.ffmpeg_processor._run_text_subprocess"
            ) as mock_sp,
            patch(
                "podcast_scraper.preprocessing.audio.ffmpeg_processor._check_ffmpeg_available",
                return_value=True,
            ),
        ):
            prep.preprocess(
                str(tmp_path / "in.mp3"),
                str(tmp_path / "out.mp3"),
            )

        cmd = mock_sp.call_args[0][0]
        joined = " ".join(cmd)
        assert "-b:a" in cmd and "32k" in cmd, joined
        assert "-ar" in cmd and "16000" in cmd, joined
        assert "start_threshold=-30dB" in joined, joined
        assert "stop_threshold=-30dB" in joined, joined
        assert "start_duration=0.5" in joined, joined
        assert "stop_duration=0.5" in joined, joined
        assert "loudnorm=I=-16" in joined, joined
