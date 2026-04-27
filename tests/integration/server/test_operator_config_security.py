"""Unit tests for operator YAML secret denylist (shallow)."""

from __future__ import annotations

import pytest

from podcast_scraper.server.operator_config_security import (

    assert_operator_yaml_safe_for_persist,
    forbidden_operator_top_level_keys,
    OperatorYamlUnsafeError,
)

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def test_forbidden_top_level_openai_key() -> None:
    assert forbidden_operator_top_level_keys({"openai_api_key": "x"}) == ["openai_api_key"]


def test_forbidden_suffix_api_key() -> None:
    assert forbidden_operator_top_level_keys({"foo_api_key": "z"}) == ["foo_api_key"]


def test_allowed_benign_keys() -> None:
    assert forbidden_operator_top_level_keys({"output_dir": "./x", "max_episodes": 3}) == []


def test_assert_safe_accepts_mapping_yaml() -> None:
    assert_operator_yaml_safe_for_persist("a: 1\nb: two\n")


def test_assert_safe_rejects_forbidden() -> None:
    with pytest.raises(OperatorYamlUnsafeError) as excinfo:
        assert_operator_yaml_safe_for_persist("gemini_api_key: secret\n")
    exc = excinfo.value
    assert exc.status_code == 400
    assert isinstance(exc.detail, dict)
    assert exc.detail.get("error") == "forbidden_operator_keys"
