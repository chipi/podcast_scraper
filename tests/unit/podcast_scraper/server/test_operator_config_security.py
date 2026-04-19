"""Unit tests for operator YAML secret denylist (shallow)."""

from __future__ import annotations

import pytest

from podcast_scraper.server.operator_config_security import (
    assert_operator_yaml_safe_for_persist,
    forbidden_operator_top_level_keys,
)


def test_forbidden_top_level_openai_key() -> None:
    assert forbidden_operator_top_level_keys({"openai_api_key": "x"}) == ["openai_api_key"]


def test_forbidden_suffix_api_key() -> None:
    assert forbidden_operator_top_level_keys({"foo_api_key": "z"}) == ["foo_api_key"]


def test_allowed_benign_keys() -> None:
    assert forbidden_operator_top_level_keys({"output_dir": "./x", "rss_urls": []}) == []


def test_assert_safe_accepts_mapping_yaml() -> None:
    assert_operator_yaml_safe_for_persist("a: 1\nb: two\n")


def test_assert_safe_rejects_forbidden() -> None:
    with pytest.raises(Exception) as excinfo:
        assert_operator_yaml_safe_for_persist("gemini_api_key: secret\n")
    exc = excinfo.value
    assert getattr(exc, "status_code", None) == 400
    detail = getattr(exc, "detail", None)
    assert isinstance(detail, dict)
    assert detail.get("error") == "forbidden_operator_keys"
