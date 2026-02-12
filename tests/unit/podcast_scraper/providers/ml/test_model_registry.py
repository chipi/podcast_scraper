"""Unit tests for Model Registry (RFC-044).

Verifies registry lookup, dynamic detection fallback, pattern-based fallbacks,
safe default, extensibility (register_model), and completeness vs DEFAULT_SUMMARY_MODELS.
"""

import unittest

import pytest

from podcast_scraper.providers.ml import summarizer
from podcast_scraper.providers.ml.model_registry import ModelCapabilities, ModelRegistry


@pytest.mark.unit
class TestModelRegistryLookup(unittest.TestCase):
    """Registry lookup for known model IDs and aliases."""

    def test_bart_aliases_and_ids(self):
        """All BART aliases and full IDs return 1024 max_input_tokens."""
        for model_id in (
            "bart-large",
            "bart-small",
            "fast",
            "facebook/bart-large-cnn",
            "facebook/bart-base",
            "sshleifer/distilbart-cnn-12-6",
        ):
            caps = ModelRegistry.get_capabilities(model_id)
            self.assertEqual(caps.max_input_tokens, 1024, msg=model_id)
            self.assertEqual(caps.model_type, "bart", msg=model_id)
            self.assertEqual(caps.model_family, "map", msg=model_id)
            self.assertFalse(caps.supports_long_context, msg=model_id)

    def test_pegasus_aliases_and_ids(self):
        """All PEGASUS aliases and full IDs return 1024 max_input_tokens."""
        for model_id in (
            "pegasus",
            "pegasus-cnn",
            "pegasus-xsum",
            "google/pegasus-large",
            "google/pegasus-cnn_dailymail",
            "google/pegasus-xsum",
        ):
            caps = ModelRegistry.get_capabilities(model_id)
            self.assertEqual(caps.max_input_tokens, 1024, msg=model_id)
            self.assertEqual(caps.model_type, "pegasus", msg=model_id)
            self.assertEqual(caps.model_family, "map", msg=model_id)

    def test_led_aliases_and_ids(self):
        """All LED aliases and full IDs return 16384 max_input_tokens."""
        for model_id in (
            "long",
            "long-large",
            "long-fast",
            "allenai/led-large-16384",
            "allenai/led-base-16384",
        ):
            caps = ModelRegistry.get_capabilities(model_id)
            self.assertEqual(caps.max_input_tokens, 16384, msg=model_id)
            self.assertEqual(caps.model_type, "led", msg=model_id)
            self.assertTrue(caps.supports_long_context, msg=model_id)

    def test_default_chunk_size_for_bart(self):
        """BART/PEGASUS have default_chunk_size 600, default_overlap 60."""
        caps = ModelRegistry.get_capabilities("bart-small")
        self.assertEqual(caps.default_chunk_size, 600)
        self.assertEqual(caps.default_overlap, 60)

    def test_default_chunk_size_for_led(self):
        """LED has default_chunk_size 16384, default_overlap 1638."""
        caps = ModelRegistry.get_capabilities("long-fast")
        self.assertEqual(caps.default_chunk_size, 16384)
        self.assertEqual(caps.default_overlap, 1638)


@pytest.mark.unit
class TestModelRegistryCompleteness(unittest.TestCase):
    """All DEFAULT_SUMMARY_MODELS keys and values are in registry."""

    def test_all_default_summary_model_aliases_registered(self):
        """Every key in DEFAULT_SUMMARY_MODELS has a registry entry."""
        for alias in summarizer.DEFAULT_SUMMARY_MODELS:
            caps = ModelRegistry.get_capabilities(alias)
            self.assertNotEqual(
                caps.model_type,
                "unknown",
                msg=f"Alias {alias} should be in registry",
            )

    def test_all_default_summary_model_full_ids_registered(self):
        """Every value (full ID) in DEFAULT_SUMMARY_MODELS has a registry entry."""
        for full_id in summarizer.DEFAULT_SUMMARY_MODELS.values():
            caps = ModelRegistry.get_capabilities(full_id)
            self.assertNotEqual(
                caps.model_type,
                "unknown",
                msg=f"Full ID {full_id} should be in registry",
            )


@pytest.mark.unit
class TestModelRegistryPatternFallback(unittest.TestCase):
    """Pattern-based fallback for unknown model IDs."""

    def test_led_pattern_fallback(self):
        """Unknown model ID containing 'led' gets LED capabilities."""
        caps = ModelRegistry.get_capabilities("some/led-custom-16384")
        self.assertEqual(caps.max_input_tokens, 16384)
        self.assertEqual(caps.model_type, "led")
        self.assertTrue(caps.supports_long_context)

    def test_bart_pattern_fallback(self):
        """Unknown model ID containing 'bart' gets BART capabilities."""
        caps = ModelRegistry.get_capabilities("org/custom-bart-v2")
        self.assertEqual(caps.max_input_tokens, 1024)
        self.assertEqual(caps.model_type, "bart")

    def test_pegasus_pattern_fallback(self):
        """Unknown model ID containing 'pegasus' gets PEGASUS capabilities."""
        caps = ModelRegistry.get_capabilities("org/pegasus-xyz")
        self.assertEqual(caps.max_input_tokens, 1024)
        self.assertEqual(caps.model_type, "pegasus")

    def test_flan_t5_pattern_fallback(self):
        """Unknown model ID containing 'flan-t5' gets FLAN-T5 capabilities."""
        caps = ModelRegistry.get_capabilities("google/custom-flan-t5")
        self.assertEqual(caps.max_input_tokens, 512)
        self.assertEqual(caps.model_type, "flan-t5")
        self.assertEqual(caps.model_family, "reduce")

    def test_safe_default_unknown_model(self):
        """Completely unknown model gets safe default 1024."""
        caps = ModelRegistry.get_capabilities("unknown/strange-model-xyz")
        self.assertEqual(caps.max_input_tokens, 1024)
        self.assertEqual(caps.model_type, "unknown")
        self.assertEqual(caps.model_family, "unknown")
        self.assertFalse(caps.supports_long_context)


@pytest.mark.unit
class TestModelRegistryDynamicDetection(unittest.TestCase):
    """Dynamic detection from model_instance when not in registry."""

    def test_dynamic_detection_from_model_config(self):
        """When model_instance has .config.max_position_embeddings, use it."""
        config = type("Config", (), {"max_position_embeddings": 8192, "model_type": "led"})()
        model_instance = type("Model", (), {"config": config})()
        caps = ModelRegistry.get_capabilities("custom/led-8k", model_instance=model_instance)
        self.assertEqual(caps.max_input_tokens, 8192)
        self.assertTrue(caps.supports_long_context)  # >= 4096

    def test_dynamic_detection_from_model_model_config(self):
        """When model_instance has .model.config (pipeline wrapper), use it."""
        config = type("Config", (), {"max_position_embeddings": 512, "model_type": "t5"})()
        inner = type("Inner", (), {"config": config})()
        model_instance = type("Model", (), {"model": inner})()
        caps = ModelRegistry.get_capabilities("custom/t5", model_instance=model_instance)
        self.assertEqual(caps.max_input_tokens, 512)

    def test_dynamic_detection_fallback_when_no_max_position(self):
        """When config has no max_position_embeddings, fall through to pattern/default."""
        config = type("Config", (), {"model_type": "other"})()
        model_instance = type("Model", (), {"config": config})()
        caps = ModelRegistry.get_capabilities("unknown/other-model", model_instance=model_instance)
        self.assertEqual(caps.max_input_tokens, 1024)


@pytest.mark.unit
class TestModelRegistryExtensibility(unittest.TestCase):
    """register_model adds new entries."""

    def test_register_model_adds_lookup(self):
        """After register_model, get_capabilities returns the registered capabilities."""
        custom = ModelCapabilities(
            max_input_tokens=4096,
            model_type="custom",
            model_family="map",
            supports_long_context=True,
        )
        try:
            ModelRegistry.register_model("custom/my-model", custom)
            caps = ModelRegistry.get_capabilities("custom/my-model")
            self.assertEqual(caps.max_input_tokens, 4096)
            self.assertEqual(caps.model_type, "custom")
        finally:
            # Teardown: remove so other tests are not affected (shared _registry)
            del ModelRegistry._registry["custom/my-model"]
