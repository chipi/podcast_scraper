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

from run_acceptance_tests import _extract_provider_info  # noqa: E402


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
