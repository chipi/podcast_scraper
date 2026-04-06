"""Unit tests for scripts/acceptance/run_acceptance_tests.py helpers."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from podcast_scraper.rss.feed_cache import ENV_RSS_CACHE_DIR

# Add scripts/acceptance to path so we can import run_acceptance_tests
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_ACCEPTANCE = _PROJECT_ROOT / "scripts" / "acceptance"
if str(_SCRIPTS_ACCEPTANCE) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ACCEPTANCE))

from run_acceptance_tests import (  # noqa: E402
    _extract_provider_info,
    _line_is_debug_for_console_filter,
    apply_session_rss_cache_env,
    filter_fast_configs,
)


@pytest.mark.unit
class TestExtractProviderInfo:
    """Tests for _extract_provider_info."""

    def test_whisper_provider_missing_whisper_model_defaults_to_base_en(self):
        """When transcription_provider is whisper and whisper_model is missing, default is base.en.

        This matches Config default in config.py so acceptance reports reflect what the service
        actually used. Prevents reporting 'base' when the app used base.en.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("transcription_provider: whisper\n")
            config_path = Path(f.name)

        try:
            info = _extract_provider_info(config_path)
            assert info.get("transcription_provider") == "whisper"
            assert info.get("transcription_model") == "base.en"
        finally:
            config_path.unlink(missing_ok=True)

    def test_whisper_provider_explicit_whisper_model_preserved(self):
        """Explicit whisper_model in config is preserved."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("transcription_provider: whisper\nwhisper_model: tiny.en\n")
            config_path = Path(f.name)

        try:
            info = _extract_provider_info(config_path)
            assert info.get("transcription_model") == "tiny.en"
        finally:
            config_path.unlink(missing_ok=True)

    def test_hybrid_ml_provider_extracts_map_reduce_backend(self):
        """hybrid_ml config extracts map_model, reduce_model, reduce_backend."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "summary_provider: hybrid_ml\n"
                "hybrid_map_model: longt5-base\n"
                "hybrid_reduce_model: qwen2.5:7b\n"
                "hybrid_reduce_backend: ollama\n"
            )
            config_path = Path(f.name)

        try:
            info = _extract_provider_info(config_path)
            assert info.get("summary_provider") == "hybrid_ml"
            assert info.get("summary_map_model") == "longt5-base"
            assert info.get("summary_reduce_model") == "qwen2.5:7b"
            assert info.get("summary_reduce_backend") == "ollama"
        finally:
            config_path.unlink(missing_ok=True)


@pytest.mark.unit
class TestLineIsDebugForConsoleFilter:
    """Tests for DEBUG line filtering when streaming to console."""

    def test_detects_levelname_debug(self):
        assert _line_is_debug_for_console_filter(
            "2026-01-01 12:00:00,000 - podcast_scraper - DEBUG - msg"
        )

    def test_detects_debug_prefix(self):
        assert _line_is_debug_for_console_filter("DEBUG root: something")

    def test_not_debug_for_info(self):
        assert not _line_is_debug_for_console_filter(
            "2026-01-01 12:00:00,000 INFO podcast_scraper.workflow: ok"
        )

    def test_not_debug_for_warning(self):
        assert not _line_is_debug_for_console_filter("WARNING: disk almost full")

    def test_empty_line_not_debug(self):
        assert not _line_is_debug_for_console_filter("")


@pytest.mark.unit
class TestFilterFastConfigs:
    """Tests for filter_fast_configs (--fast-only)."""

    def test_filter_keeps_only_stems_in_set(self):
        """Only configs whose stem is in fast_stems are kept."""
        fast_stems = {"acceptance_planet_money_ml_prod", "acceptance_planet_money_ml_dev"}
        configs = [
            Path("config/acceptance/full/acceptance_planet_money_ml_prod.yaml"),
            Path("config/acceptance/full/acceptance_planet_money_openai.yaml"),
            Path("config/acceptance/full/acceptance_planet_money_ml_dev.yaml"),
        ]
        out = filter_fast_configs(configs, fast_stems)
        assert len(out) == 2
        assert out[0].stem == "acceptance_planet_money_ml_prod"
        assert out[1].stem == "acceptance_planet_money_ml_dev"

    def test_filter_empty_stems_returns_all(self):
        """When fast_stems is empty, all configs are returned."""
        configs = [Path("a/acceptance_planet_money_ml.yaml")]
        assert filter_fast_configs(configs, set()) == configs


@pytest.mark.unit
class TestApplySessionRssCacheEnv:
    """Tests for apply_session_rss_cache_env (acceptance session RSS feed cache)."""

    def test_creates_rss_cache_dir_and_sets_env(self, tmp_path, monkeypatch):
        """Session rss_cache exists and PODCAST_SCRAPER_RSS_CACHE_DIR points to it."""
        monkeypatch.delenv(ENV_RSS_CACHE_DIR, raising=False)
        session_dir = tmp_path / "sessions" / "session_abc"
        session_dir.mkdir(parents=True)

        rss_cache = apply_session_rss_cache_env(session_dir)

        expected = (session_dir / "rss_cache").resolve()
        assert rss_cache == expected
        assert rss_cache.is_dir()
        assert os.environ[ENV_RSS_CACHE_DIR] == str(rss_cache)
