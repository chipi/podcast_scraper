"""Tests for config_constants (revision helpers, etc.)."""

import pytest

from podcast_scraper import config_constants

pytestmark = [pytest.mark.unit]


class TestIsShaRevision:
    """Tests for is_sha_revision (Issue #428)."""

    def test_accepts_40_lowercase_hex(self):
        """40 lowercase hex chars are accepted as SHA."""
        sha = "38335783885b338d93791936c54bb4be46bebed9"
        assert config_constants.is_sha_revision(sha) is True

    def test_accepts_40_uppercase_hex(self):
        """40 uppercase hex chars are accepted (normalized to lower)."""
        sha = "38335783885B338D93791936C54BB4BE46BEBED9"
        assert config_constants.is_sha_revision(sha) is True

    def test_rejects_main(self):
        """Branch name 'main' is not a SHA."""
        assert config_constants.is_sha_revision("main") is False

    def test_rejects_short_string(self):
        """Short string is not a SHA."""
        assert config_constants.is_sha_revision("abc") is False

    def test_rejects_long_string(self):
        """41+ chars are not a SHA."""
        assert config_constants.is_sha_revision("0" * 41) is False

    def test_rejects_empty(self):
        """Empty string is not a SHA."""
        assert config_constants.is_sha_revision("") is False

    def test_rejects_non_hex(self):
        """Non-hex character (e.g. 'g') is rejected."""
        assert config_constants.is_sha_revision("g" + "0" * 39) is False
