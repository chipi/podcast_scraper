"""Unit tests for scripts/acceptance/run_acceptance_tests.py helpers."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts/acceptance to path so we can import run_acceptance_tests
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_ACCEPTANCE = _PROJECT_ROOT / "scripts" / "acceptance"
if str(_SCRIPTS_ACCEPTANCE) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ACCEPTANCE))

from run_acceptance_tests import _extract_provider_info, filter_fast_configs  # noqa: E402


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
class TestFilterFastConfigs:
    """Tests for filter_fast_configs (--fast-only)."""

    def test_filter_keeps_only_stems_in_set(self):
        """Only configs whose stem is in fast_stems are kept."""
        fast_stems = {"acceptance_planet_money_ml", "acceptance_planet_money_gi_ml"}
        configs = [
            Path("config/acceptance/summarization/acceptance_planet_money_ml.yaml"),
            Path("config/acceptance/summarization/acceptance_planet_money_openai.yaml"),
            Path("config/acceptance/gi/acceptance_planet_money_gi_ml.yaml"),
        ]
        out = filter_fast_configs(configs, fast_stems)
        assert len(out) == 2
        assert out[0].stem == "acceptance_planet_money_ml"
        assert out[1].stem == "acceptance_planet_money_gi_ml"

    def test_filter_empty_stems_returns_all(self):
        """When fast_stems is empty, all configs are returned."""
        configs = [Path("a/acceptance_planet_money_ml.yaml")]
        assert filter_fast_configs(configs, set()) == configs
