"""Tests for provider fingerprinting functionality."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from podcast_scraper.evaluation.fingerprint import (
    generate_provider_fingerprint,
    ProviderFingerprint,
)


class TestProviderFingerprint(unittest.TestCase):
    """Test ProviderFingerprint dataclass."""

    def test_fingerprint_creation(self):
        """Test creating a fingerprint."""
        fingerprint = ProviderFingerprint(
            model_name="facebook/bart-large-cnn",
            device="mps",
            package_version="2.5.0",
        )
        self.assertEqual(fingerprint.model_name, "facebook/bart-large-cnn")
        self.assertEqual(fingerprint.device, "mps")
        # Compute hash manually (normally done by generate_provider_fingerprint)
        fingerprint.fingerprint_hash = fingerprint.compute_hash()
        self.assertIsNotNone(fingerprint.fingerprint_hash)

    def test_fingerprint_to_dict(self):
        """Test converting fingerprint to dictionary."""
        fingerprint = ProviderFingerprint(
            model_name="test-model",
            device="cpu",
            package_version="2.5.0",
        )
        result = fingerprint.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["model_name"], "test-model")
        self.assertEqual(result["device"], "cpu")
        self.assertIn("fingerprint_hash", result)

    def test_fingerprint_hash_computation(self):
        """Test fingerprint hash is computed correctly."""
        fingerprint1 = ProviderFingerprint(
            model_name="test-model",
            device="cpu",
            package_version="2.5.0",
        )
        hash1 = fingerprint1.compute_hash()

        # Same fingerprint should produce same hash
        fingerprint2 = ProviderFingerprint(
            model_name="test-model",
            device="cpu",
            package_version="2.5.0",
        )
        hash2 = fingerprint2.compute_hash()
        self.assertEqual(hash1, hash2)

        # Different model should produce different hash
        fingerprint3 = ProviderFingerprint(
            model_name="different-model",
            device="cpu",
            package_version="2.5.0",
        )
        hash3 = fingerprint3.compute_hash()
        self.assertNotEqual(hash1, hash3)


class TestGenerateProviderFingerprint(unittest.TestCase):
    """Test generate_provider_fingerprint function."""

    @patch("podcast_scraper.evaluation.fingerprint._get_git_commit")
    @patch("podcast_scraper.evaluation.fingerprint._get_package_version")
    @patch("podcast_scraper.evaluation.fingerprint._get_library_versions")
    def test_generate_fingerprint_basic(self, mock_lib_versions, mock_pkg_version, mock_git):
        """Test generating a basic fingerprint."""
        mock_git.return_value = ("abc1234", False)
        mock_pkg_version.return_value = "2.5.0"
        mock_lib_versions.return_value = {"torch": "2.0.0"}

        fingerprint = generate_provider_fingerprint(
            model_name="test-model",
            device="cpu",
            preprocessing_profile="cleaning_v3",
        )

        self.assertEqual(fingerprint.model_name, "test-model")
        self.assertEqual(fingerprint.device, "cpu")
        self.assertEqual(fingerprint.preprocessing_profile, "cleaning_v3")
        self.assertEqual(fingerprint.git_commit, "abc1234")
        self.assertFalse(fingerprint.git_dirty)
        self.assertIsNotNone(fingerprint.fingerprint_hash)

    @patch("podcast_scraper.evaluation.fingerprint._get_git_commit")
    def test_generate_fingerprint_no_git(self, mock_git):
        """Test generating fingerprint when not in git repo."""
        mock_git.return_value = (None, False)

        fingerprint = generate_provider_fingerprint(
            model_name="test-model",
            device="cpu",
        )

        self.assertIsNone(fingerprint.git_commit)
        self.assertFalse(fingerprint.git_dirty)
