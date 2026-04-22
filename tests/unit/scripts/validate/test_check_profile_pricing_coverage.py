"""Unit tests for the #651 profile-pricing coverage guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "validate" / "check_profile_pricing_coverage.py"

_spec = importlib.util.spec_from_file_location("check_pricing_coverage", SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["check_pricing_coverage"] = _mod
_spec.loader.exec_module(_mod)

extract_model_refs = _mod._extract_model_refs  # noqa: SLF001
pricing_has_model = _mod._pricing_has_model  # noqa: SLF001

pytestmark = [pytest.mark.unit]


class TestExtractModelRefs:
    def test_billable_providers_captured(self) -> None:
        doc = {
            "openai_transcription_model": "whisper-1",
            "gemini_summary_model": "gemini-2.5-flash-lite",
            "anthropic_speaker_model": "claude-haiku-4-5",
        }
        refs = extract_model_refs(doc)
        assert ("openai", "transcription", "whisper-1") in refs
        assert ("gemini", "text", "gemini-2.5-flash-lite") in refs
        assert ("anthropic", "text", "claude-haiku-4-5") in refs

    def test_local_stack_ignored(self) -> None:
        doc = {
            "whisper_model": "medium.en",
            "ner_model": "en_core_web_trf",
            "summary_provider": "transformers",
        }
        assert extract_model_refs(doc) == []

    def test_ollama_ignored_as_non_billable(self) -> None:
        doc = {"ollama_summary_model": "llama3.1:8b"}
        assert extract_model_refs(doc) == []

    def test_cleaning_insight_kg_extraction_map_to_text(self) -> None:
        doc = {
            "openai_cleaning_model": "gpt-4o-mini",
            "gemini_insight_model": "gemini-2.5-flash-lite",
            "anthropic_kg_extraction_model": "claude-haiku-4-5",
        }
        refs = extract_model_refs(doc)
        sections = {(p, s) for p, s, _ in refs}
        assert ("openai", "text") in sections
        assert ("gemini", "text") in sections
        assert ("anthropic", "text") in sections

    def test_unknown_field_ignored(self) -> None:
        assert extract_model_refs({"unrelated_field": "foo"}) == []
        assert extract_model_refs({"openai_model": "gpt-4o"}) == []  # missing capability token


class TestPricingHasModel:
    def _pricing_fixture(self) -> dict:
        return {
            "providers": {
                "openai": {
                    "transcription": {"whisper-1": {"cost_per_minute": 0.006}},
                    "text": {
                        "gpt-4o": {"input_cost_per_1m_tokens": 2.5},
                        "default": {"input_cost_per_1m_tokens": 2.5},
                    },
                },
                "gemini": {
                    "text": {
                        "gemini-2.5-flash-lite": {"input_cost_per_1m_tokens": 0.10},
                    }
                },
            }
        }

    def test_exact_match(self) -> None:
        p = self._pricing_fixture()
        assert pricing_has_model(p, "openai", "transcription", "whisper-1")
        assert pricing_has_model(p, "gemini", "text", "gemini-2.5-flash-lite")

    def test_prefix_match(self) -> None:
        p = self._pricing_fixture()
        assert pricing_has_model(p, "gemini", "text", "gemini-2.5-flash-lite-latest")
        assert pricing_has_model(p, "openai", "text", "gpt-4o-2024-08-06")

    def test_fallback_to_default(self) -> None:
        # "gpt-5-turbo" has no exact row but openai.text.default exists.
        p = self._pricing_fixture()
        assert pricing_has_model(p, "openai", "text", "gpt-5-turbo")

    def test_missing_provider(self) -> None:
        p = self._pricing_fixture()
        assert not pricing_has_model(p, "grok", "text", "grok-4")

    def test_missing_section(self) -> None:
        p = self._pricing_fixture()
        # gemini has no transcription section
        assert not pricing_has_model(p, "gemini", "transcription", "gemini-2.5-flash-lite")

    def test_missing_model_no_default(self) -> None:
        # Gemini.text has no default row in the fixture.
        p = self._pricing_fixture()
        assert not pricing_has_model(p, "gemini", "text", "gemini-unknown")


def test_script_passes_on_current_repo_state() -> None:
    """Smoke test: the CI guard succeeds on the committed config/."""
    # Run the main() function directly rather than via subprocess — cleaner
    # import + lets us assert on exit code.
    assert _mod.main() == 0


class TestGuardCatchesDrift:
    """End-to-end negative tests: prove main() returns 1 when a profile
    references a model that has no YAML rate row. Guards against the
    failure-mode #651 exists to prevent — a profile landing with a new
    model and no accompanying pricing entry, silently dropping its
    cost to $0.
    """

    def _write_pricing(self, path: Path, extra_openai_text: dict | None = None) -> None:
        text_models = {"gpt-4o": {"input_cost_per_1m_tokens": 2.5}}
        if extra_openai_text:
            text_models.update(extra_openai_text)
        payload = {
            "providers": {
                "openai": {
                    "transcription": {"whisper-1": {"cost_per_minute": 0.006}},
                    "text": text_models,
                }
            }
        }
        import yaml as _yaml

        path.write_text(_yaml.safe_dump(payload), encoding="utf-8")

    def _write_profile(self, path: Path, **fields: str) -> None:
        import yaml as _yaml

        path.write_text(_yaml.safe_dump(fields), encoding="utf-8")

    def test_main_returns_1_when_profile_references_missing_model(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        pricing = tmp_path / "pricing.yaml"
        self._write_pricing(pricing)
        # Profile references openai.text.gpt-5-mega which is NOT in pricing
        # and openai.text has no `default` fallback row.
        self._write_profile(
            profiles_dir / "ships_new_model.yaml",
            openai_summary_model="gpt-5-mega",
        )

        assert _mod.main(profiles_dir=profiles_dir, pricing_yaml=pricing) == 1

    def test_main_returns_1_when_transcription_model_missing(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        pricing = tmp_path / "pricing.yaml"
        self._write_pricing(pricing)
        # Profile uses an unpriced Whisper variant; transcription has no
        # default row so this must FAIL.
        self._write_profile(
            profiles_dir / "unpriced_whisper.yaml",
            openai_transcription_model="whisper-3-ultra",
        )

        assert _mod.main(profiles_dir=profiles_dir, pricing_yaml=pricing) == 1

    def test_main_returns_0_when_model_falls_back_to_default(self, tmp_path: Path) -> None:
        """Sanity-inverse: if the provider+section has a `default` row, an
        unlisted model is tolerated. Proves the negative tests above fail
        for the *right* reason (truly missing), not a stray yaml issue."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        pricing = tmp_path / "pricing.yaml"
        self._write_pricing(
            pricing,
            extra_openai_text={"default": {"input_cost_per_1m_tokens": 2.5}},
        )
        self._write_profile(
            profiles_dir / "new_model_but_default_covers.yaml",
            openai_summary_model="gpt-5-mega",
        )

        assert _mod.main(profiles_dir=profiles_dir, pricing_yaml=pricing) == 0
