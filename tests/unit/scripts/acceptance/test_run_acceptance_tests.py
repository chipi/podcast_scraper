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
    _assess_vector_index_run,
    _estimate_llm_cost_usd_from_stdout_log,
    _extract_provider_info,
    _line_is_debug_for_console_filter,
    _strict_vector_index_requested,
    apply_session_rss_cache_env,
    collect_logs_from_output,
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


@pytest.mark.unit
class TestEstimateLlmCostAcceptance:
    """Regression: multi-feed cost from stdout when metrics are nested."""

    def test_stdout_log_sums_total_estimated_cost_lines(self, tmp_path):
        """Parses every 'Total estimated cost: $X' line (e.g. per-feed summaries)."""
        p = tmp_path / "stdout.log"
        p.write_text(
            "x\nTotal estimated cost: $0.01\ny\nTotal estimated cost: $0.02\n",
            encoding="utf-8",
        )
        assert _estimate_llm_cost_usd_from_stdout_log(p) == pytest.approx(0.03)

    def test_nested_metrics_summed_with_monkeypatch(self, tmp_path, monkeypatch):
        """rglob metrics.json under run dir each contribute to the total."""
        import run_acceptance_tests as rat

        (tmp_path / "config.original.yaml").write_text(
            "rss: https://example.com/feed.xml\noutput_dir: .\nmax_episodes: 1\n",
            encoding="utf-8",
        )
        (tmp_path / "feeds" / "a" / "run_x").mkdir(parents=True)
        (tmp_path / "feeds" / "b" / "run_y").mkdir(parents=True)
        (tmp_path / "feeds" / "a" / "run_x" / "metrics.json").write_text("{}", encoding="utf-8")
        (tmp_path / "feeds" / "b" / "run_y" / "metrics.json").write_text("{}", encoding="utf-8")

        def fake_est(_cfg, _d):
            return 0.04

        monkeypatch.setattr(rat, "estimated_llm_cost_usd_from_metrics_dict", fake_est)
        total = rat._estimate_llm_cost_usd_for_run_dir(tmp_path)
        assert total == pytest.approx(0.08)


@pytest.mark.unit
class TestAssessVectorIndexRun:
    """FAISS health flags for acceptance run_data / strict mode."""

    def test_empty_metadata_with_episodes_is_fail(self, tmp_path):
        from podcast_scraper.config import Config

        (tmp_path / "search").mkdir()
        (tmp_path / "search" / "metadata.json").write_text("{}", encoding="utf-8")
        cfg = Config(rss="https://example.com/f.xml", output_dir=str(tmp_path), max_episodes=1)
        cfg = cfg.model_copy(update={"vector_search": True, "vector_backend": "faiss"})
        ok, note = _assess_vector_index_run(tmp_path, cfg, episodes_processed=3, is_dry_run=False)
        assert ok is False
        assert "empty" in note

    def test_vector_search_off_skips(self, tmp_path):
        from podcast_scraper.config import Config

        cfg = Config(rss="https://example.com/f.xml", output_dir=str(tmp_path), max_episodes=1)
        cfg = cfg.model_copy(update={"vector_search": False})
        ok, note = _assess_vector_index_run(tmp_path, cfg, episodes_processed=3, is_dry_run=False)
        assert ok is True
        assert note == "vector_search_off"


@pytest.mark.unit
class TestCollectLogsLineTotals:
    """Repeated WARNING lines contribute to warning_lines_total once per line."""

    def test_repeated_warnings_count_lines_and_distinct(self, tmp_path):
        p = tmp_path / "stderr.log"
        line = "2026-01-01 12:00:00,000 WARNING sentence_transformers.SentenceTransformer: spam\n"
        p.write_text(line * 4, encoding="utf-8")
        out = collect_logs_from_output(tmp_path)
        assert out["warning_lines_total"] == 4
        assert out["warning_count_distinct"] == 1
        assert len(out["warnings"]) == 1


@pytest.mark.unit
class TestStrictVectorIndexEnv:
    def test_strict_vector_index_env_truthy(self, monkeypatch):
        monkeypatch.setenv("STRICT_VECTOR_INDEX", "1")
        assert _strict_vector_index_requested() is True
        monkeypatch.setenv("STRICT_VECTOR_INDEX", "0")
        assert _strict_vector_index_requested() is False
