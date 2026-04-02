"""Unit tests for provider_metadata helpers."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.utils import provider_metadata

pytestmark = [pytest.mark.unit]


class TestLogProviderMetadata(unittest.TestCase):
    """log_provider_metadata."""

    def test_logs_debug_when_metadata_present(self) -> None:
        with self.assertLogs(provider_metadata.logger, level="DEBUG") as cm:
            provider_metadata.log_provider_metadata(
                "OpenAI",
                account="org-123",
                region="us-east-1",
            )
        self.assertTrue(any("OpenAI provider metadata" in r.getMessage() for r in cm.records))
        self.assertTrue(any("org-123" in r.getMessage() for r in cm.records))

    @patch.object(provider_metadata.logger, "debug")
    def test_no_debug_when_all_empty(self, mock_debug: MagicMock) -> None:
        provider_metadata.log_provider_metadata("Gemini")
        mock_debug.assert_not_called()


class TestExtractRegionFromEndpoint(unittest.TestCase):
    """extract_region_from_endpoint."""

    def test_none_and_empty(self) -> None:
        self.assertIsNone(provider_metadata.extract_region_from_endpoint(None))
        self.assertIsNone(provider_metadata.extract_region_from_endpoint(""))

    def test_openai_style_host(self) -> None:
        self.assertEqual(
            provider_metadata.extract_region_from_endpoint("https://us-east-1.api.openai.com/v1"),
            "us-east-1",
        )

    def test_anthropic_style_host(self) -> None:
        self.assertEqual(
            provider_metadata.extract_region_from_endpoint("https://api.eu-west-1.anthropic.com"),
            "eu-west-1",
        )

    def test_no_region_in_url(self) -> None:
        self.assertIsNone(
            provider_metadata.extract_region_from_endpoint("https://gemini.googleapis.com/v1"),
        )


class TestValidateApiKeyFormat(unittest.TestCase):
    """validate_api_key_format."""

    def test_missing_key(self) -> None:
        ok, err = provider_metadata.validate_api_key_format(None, "OpenAI")
        self.assertFalse(ok)
        self.assertIn("missing", (err or "").lower())

    def test_empty_key(self) -> None:
        ok, err = provider_metadata.validate_api_key_format("", "Mistral")
        self.assertFalse(ok)
        self.assertIn("missing", (err or "").lower())

    def test_too_short(self) -> None:
        ok, err = provider_metadata.validate_api_key_format("short", "X")
        self.assertFalse(ok)
        self.assertIn("short", (err or "").lower())

    def test_prefix_mismatch(self) -> None:
        ok, err = provider_metadata.validate_api_key_format(
            "notsk-xxxxxxxxxx",
            "OpenAI",
            expected_prefixes=["sk-"],
        )
        self.assertFalse(ok)
        self.assertIn("prefix", (err or "").lower())

    def test_valid_with_prefix(self) -> None:
        ok, err = provider_metadata.validate_api_key_format(
            "sk-1234567890",
            "OpenAI",
            expected_prefixes=["sk-"],
        )
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_valid_without_prefix_constraint(self) -> None:
        ok, err = provider_metadata.validate_api_key_format(
            "any-long-enough-key",
            "Ollama",
        )
        self.assertTrue(ok)
        self.assertIsNone(err)
