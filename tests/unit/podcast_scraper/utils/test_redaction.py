"""Unit tests for podcast_scraper.utils.redaction module."""

from __future__ import annotations

import pytest

from podcast_scraper.utils import redaction


@pytest.mark.unit
class TestIsSecretKey:
    """Tests for _is_secret_key."""

    def test_denylist_key_matches(self):
        """Key in denylist returns True."""
        assert redaction._is_secret_key("api_key") is True
        assert redaction._is_secret_key("token") is True
        assert redaction._is_secret_key("password") is True

    def test_key_contains_pattern(self):
        """Key containing pattern returns True."""
        assert redaction._is_secret_key("my_api_key") is True
        assert redaction._is_secret_key("auth_token_here") is True

    def test_safe_key_returns_false(self):
        """Safe key returns False."""
        assert redaction._is_secret_key("title") is False
        assert redaction._is_secret_key("url") is False


@pytest.mark.unit
class TestLooksLikeSecret:
    """Tests for _looks_like_secret."""

    def test_sk_prefix_matches(self):
        """sk-* string matches API key pattern."""
        assert redaction._looks_like_secret("sk-" + "a" * 20) is True

    def test_bearer_token_matches(self):
        """Bearer token matches."""
        assert redaction._looks_like_secret("Bearer " + "x" * 25) is True

    def test_long_alphanumeric_matches(self):
        """Long alphanumeric string matches."""
        assert redaction._looks_like_secret("a" * 32) is True

    def test_normal_string_returns_false(self):
        """Normal string returns False."""
        assert redaction._looks_like_secret("hello world") is False

    def test_non_string_returns_false(self):
        """Non-string returns False."""
        assert redaction._looks_like_secret(123) is False
        assert redaction._looks_like_secret(None) is False


@pytest.mark.unit
class TestRedactSecrets:
    """Tests for redact_secrets."""

    def test_dict_with_secret_key_redacted(self):
        """Dict with secret key has value redacted."""
        data = {"api_key": "sk-secret123", "title": "Podcast"}
        out = redaction.redact_secrets(data)
        assert out["api_key"] == "__redacted__"
        assert out["title"] == "Podcast"

    def test_dict_with_secret_value_by_pattern(self):
        """Dict with value matching secret pattern is redacted when redact_patterns=True."""
        data = {"x": "sk-" + "a" * 20}
        out = redaction.redact_secrets(data, redact_patterns=True)
        assert out["x"] == "__redacted__"

    def test_redact_patterns_false_does_not_redact_by_value(self):
        """When redact_patterns=False, only key-based redaction applies."""
        data = {"other_key": "sk-" + "a" * 20}
        out = redaction.redact_secrets(data, redact_patterns=False)
        assert out["other_key"] == "sk-" + "a" * 20

    def test_nested_dict_redacted(self):
        """Nested dict is recursively redacted."""
        data = {"level1": {"api_key": "secret", "nested": {"token": "xyz"}}}
        out = redaction.redact_secrets(data)
        assert out["level1"]["api_key"] == "__redacted__"
        assert out["level1"]["nested"]["token"] == "__redacted__"

    def test_list_recursively_redacted(self):
        """List items are recursively redacted."""
        data = [{"password": "pwd"}, "normal"]
        out = redaction.redact_secrets(data)
        assert out[0]["password"] == "__redacted__"
        assert out[1] == "normal"

    def test_primitive_looks_like_secret_redacted(self):
        """Top-level primitive that looks like secret is redacted when redact_patterns=True."""
        out = redaction.redact_secrets("sk-" + "a" * 20, redact_patterns=True)
        assert out == "__redacted__"

    def test_primitive_safe_unchanged(self):
        """Top-level safe primitive is unchanged."""
        assert redaction.redact_secrets(42) == 42
        assert redaction.redact_secrets("hello") == "hello"
