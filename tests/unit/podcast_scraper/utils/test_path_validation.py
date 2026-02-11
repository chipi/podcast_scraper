"""Unit tests for podcast_scraper.utils.path_validation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from podcast_scraper.utils.path_validation import (
    sanitize_model_name,
    validate_cache_path,
    validate_path_is_safe,
)


@pytest.mark.unit
class TestValidateCachePath:
    """Tests for validate_cache_path."""

    def test_resolves_path(self):
        """Valid path is resolved and returned."""
        with tempfile.TemporaryDirectory() as tmp:
            p = validate_cache_path(tmp)
            assert p == Path(tmp).resolve()

    def test_path_traversal_raises(self):
        """Path containing .. raises ValueError."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_cache_path("/some/../etc/passwd")

    def test_path_outside_base_dir_raises(self):
        """Path outside base_dir raises ValueError."""
        with tempfile.TemporaryDirectory() as base:
            with pytest.raises(ValueError, match="outside base directory"):
                validate_cache_path("/tmp", base_dir=base)

    def test_path_inside_base_dir_ok(self):
        """Path inside base_dir is accepted."""
        with tempfile.TemporaryDirectory() as base:
            sub = Path(base) / "sub" / "cache"
            sub.mkdir(parents=True)
            p = validate_cache_path(sub, base_dir=base)
            assert p == sub.resolve()


@pytest.mark.unit
class TestSanitizeModelName:
    """Tests for sanitize_model_name."""

    def test_valid_model_name_unchanged(self):
        """Valid model name is returned unchanged."""
        assert sanitize_model_name("facebook/bart-base") == "facebook/bart-base"
        assert sanitize_model_name("model-name_v1") == "model-name_v1"
        assert sanitize_model_name("model_name") == "model_name"

    def test_invalid_chars_raise(self):
        """Invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            sanitize_model_name("model with spaces")
        with pytest.raises(ValueError, match="Invalid model name"):
            sanitize_model_name("model@symbol")

    def test_path_traversal_in_name_raises(self):
        """.. in model name raises ValueError (invalid chars or explicit .. check)."""
        with pytest.raises(ValueError, match="Invalid model name"):
            sanitize_model_name("aa..bb")

    def test_leading_slash_raises(self):
        """Model name starting with / raises ValueError."""
        with pytest.raises(ValueError, match="cannot.*start with"):
            sanitize_model_name("/absolute/path")


@pytest.mark.unit
class TestValidatePathIsSafe:
    """Tests for validate_path_is_safe."""

    def test_path_under_trusted_root_returns_true(self):
        """Path under a trusted root returns True."""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root).resolve()
            sub = root_path / "sub" / "file.txt"
            sub.parent.mkdir(parents=True, exist_ok=True)
            sub.touch()
            assert validate_path_is_safe(str(sub), [root_path]) is True

    def test_path_outside_trusted_roots_returns_false(self):
        """Path outside all trusted roots returns False."""
        with tempfile.TemporaryDirectory() as root:
            other = Path("/tmp/other")
            assert validate_path_is_safe(str(other), [Path(root)]) is False

    def test_allow_absolute_allows_absolute_path(self):
        """When allow_absolute=True, absolute path returns True."""
        assert validate_path_is_safe("/tmp/foo", [], allow_absolute=True) is True

    def test_path_not_under_any_root_returns_false(self):
        """Path not under any trusted root and allow_absolute False returns False."""
        result = validate_path_is_safe("/tmp/somewhere", [], allow_absolute=False)
        assert result is False
