"""Integration tests for ModelRegistry (RFC-044).

Verifies the full alias resolution → capabilities lookup chain that all ML
loaders depend on.  No optional dependencies required.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.model_registry import (
    ModeConfiguration,
    ModelRegistry,
)

pytestmark = [pytest.mark.integration]


class TestEvidenceAliasResolution:
    """resolve_evidence_model_id: alias → full HF ID for every evidence model."""

    @pytest.mark.parametrize(
        "alias, expected_prefix",
        [
            ("minilm-l6", "sentence-transformers/"),
            ("minilm-l12", "sentence-transformers/"),
            ("mpnet-base", "sentence-transformers/"),
            ("roberta-squad2", "deepset/"),
            ("deberta-squad2", "deepset/"),
            ("nli-deberta-base", "cross-encoder/"),
            ("nli-deberta-small", "cross-encoder/"),
        ],
    )
    def test_alias_resolves_to_known_prefix(self, alias, expected_prefix):
        resolved = ModelRegistry.resolve_evidence_model_id(alias)
        assert resolved.startswith(expected_prefix), f"{alias} → {resolved}"

    def test_full_hf_id_passes_through(self):
        hf_id = "sentence-transformers/all-MiniLM-L6-v2"
        assert ModelRegistry.resolve_evidence_model_id(hf_id) == hf_id

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown evidence model id"):
            ModelRegistry.resolve_evidence_model_id("nonexistent-model")


class TestCapabilitiesLookup:
    """get_capabilities: registry → dynamic → pattern → safe default."""

    @pytest.mark.parametrize(
        "model_id, expected_family",
        [
            ("facebook/bart-large-cnn", "map"),
            ("allenai/led-base-16384", "map"),
            ("google/flan-t5-base", "reduce"),
            ("sentence-transformers/all-MiniLM-L6-v2", "embedding"),
            ("deepset/roberta-base-squad2", "extractive_qa"),
            ("cross-encoder/nli-deberta-v3-base", "nli"),
        ],
    )
    def test_known_model_returns_correct_family(self, model_id, expected_family):
        caps = ModelRegistry.get_capabilities(model_id)
        assert caps.model_family == expected_family

    def test_bart_has_1024_max_tokens(self):
        caps = ModelRegistry.get_capabilities("facebook/bart-large-cnn")
        assert caps.max_input_tokens == 1024
        assert not caps.supports_long_context

    def test_led_has_16384_max_tokens(self):
        caps = ModelRegistry.get_capabilities("allenai/led-base-16384")
        assert caps.max_input_tokens == 16384
        assert caps.supports_long_context

    def test_embedding_model_has_dimension(self):
        caps = ModelRegistry.get_capabilities("sentence-transformers/all-MiniLM-L6-v2")
        assert caps.embedding_dim == 384

    def test_unknown_model_returns_safe_default(self):
        caps = ModelRegistry.get_capabilities("totally-unknown/model-xyz")
        assert caps.max_input_tokens == 1024
        assert caps.model_type == "unknown"

    def test_pattern_fallback_for_bart_like_name(self):
        caps = ModelRegistry.get_capabilities("custom-org/my-bart-model")
        assert caps.model_type == "bart"

    def test_pattern_fallback_for_led_like_name(self):
        caps = ModelRegistry.get_capabilities("custom-org/my-led-model")
        assert caps.model_type == "led"
        assert caps.supports_long_context


class TestModeConfigurations:
    """get_mode_configuration: promoted mode registry."""

    def test_ml_small_authority_exists(self):
        mode = ModelRegistry.get_mode_configuration("ml_small_authority")
        assert isinstance(mode, ModeConfiguration)
        assert mode.map_model == "bart-small"
        assert mode.reduce_model == "long-fast"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="not found in registry"):
            ModelRegistry.get_mode_configuration("nonexistent_mode")

    def test_hybrid_mode_has_ollama_reduce(self):
        mode = ModelRegistry.get_mode_configuration("ml_hybrid_bart_llama32_3b_autoresearch_v1")
        assert mode.reduce_backend == "ollama"
        assert mode.ollama_reduce_params is not None
        assert "max_tokens" in mode.ollama_reduce_params

    def test_deprecated_mode_has_deprecation_info(self):
        mode = ModelRegistry.get_mode_configuration("ml_prod_authority_v1")
        assert mode.deprecated_at is not None
        assert mode.deprecation_reason is not None
        assert "Pegasus" in mode.deprecation_reason
