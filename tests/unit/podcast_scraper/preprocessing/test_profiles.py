"""Tests for preprocessing profile registry."""

from __future__ import annotations

import unittest

from podcast_scraper.preprocessing.profiles import (
    apply_profile,
    DEFAULT_PROFILE,
    get_profile,
    list_profiles,
    register_profile,
)


class TestPreprocessingProfiles(unittest.TestCase):
    """Test preprocessing profile registry."""

    def test_list_profiles(self):
        """Test listing available profiles."""
        profiles = list_profiles()
        self.assertIsInstance(profiles, list)
        self.assertIn("cleaning_v1", profiles)
        self.assertIn("cleaning_v2", profiles)
        self.assertIn("cleaning_v3", profiles)
        self.assertIn("cleaning_none", profiles)

    def test_get_profile(self):
        """Test getting a profile by ID."""
        profile = get_profile("cleaning_v3")
        self.assertIsNotNone(profile)
        self.assertTrue(callable(profile))

    def test_get_profile_not_found(self):
        """Test error when profile not found."""
        with self.assertRaises(ValueError):
            get_profile("nonexistent_profile")

    def test_apply_profile(self):
        """Test applying a profile to text."""
        text = "  Test text with   extra spaces  "
        cleaned = apply_profile(text, "cleaning_none")
        self.assertEqual(cleaned, text)  # No-op profile

    def test_register_custom_profile(self):
        """Test registering a custom profile."""

        def custom_cleaner(text: str) -> str:
            return text.upper()

        register_profile("custom_v1", custom_cleaner)
        self.assertIn("custom_v1", list_profiles())

        result = apply_profile("test", "custom_v1")
        self.assertEqual(result, "TEST")

    def test_default_profile(self):
        """Test default profile constant."""
        self.assertEqual(DEFAULT_PROFILE, "cleaning_v3")
