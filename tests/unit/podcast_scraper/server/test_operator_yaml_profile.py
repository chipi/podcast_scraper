"""Unit tests for ``operator_yaml_profile`` (viewer operator merge helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.operator_yaml_profile import (
    expand_profile_only_with_packaged_example,
    format_profile_scalar,
    merge_operator_yaml_profile,
    parse_pipeline_install_extras,
    split_operator_yaml_profile,
)

pytestmark = pytest.mark.unit


def test_split_merge_roundtrip() -> None:
    assert split_operator_yaml_profile("profile: cloud_balanced\nmax_episodes: 2\n") == (
        "cloud_balanced",
        "max_episodes: 2",
    )
    assert merge_operator_yaml_profile("cloud_balanced", "max_episodes: 2") == (
        "profile: cloud_balanced\nmax_episodes: 2\n"
    )


def test_expand_appends_example_when_profile_only(tmp_path: Path) -> None:
    ex = tmp_path / "ex.yaml"
    ex.write_text("max_episodes: 1\ndelay_ms: 100\n", encoding="utf-8")
    out = expand_profile_only_with_packaged_example(
        "profile: local\n",
        example_path=ex,
    )
    assert "profile: local" in out
    assert "max_episodes: 1" in out
    assert "delay_ms: 100" in out


def test_expand_noop_when_body_has_keys(tmp_path: Path) -> None:
    ex = tmp_path / "ex.yaml"
    ex.write_text("max_episodes: 99\n", encoding="utf-8")
    raw = "profile: local\nmax_episodes: 3\n"
    assert expand_profile_only_with_packaged_example(raw, example_path=ex) == raw


def test_expand_empty_when_no_example(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert expand_profile_only_with_packaged_example("profile: x\n", example_path=missing) == (
        "profile: x\n"
    )


def test_expand_returns_content_when_example_path_is_none() -> None:
    raw = "profile: thin\n"
    assert expand_profile_only_with_packaged_example(raw, example_path=None) is raw


def test_format_profile_scalar_quotes_unsafe_tokens() -> None:
    assert format_profile_scalar("") == '""'
    assert format_profile_scalar("cloud_balanced") == "cloud_balanced"
    assert format_profile_scalar("has space") == '"has space"'


def test_parse_pipeline_install_extras_non_ml_llm_returns_token() -> None:
    assert parse_pipeline_install_extras("pipeline_install_extras: cuda\n") == "cuda"


def test_split_profile_strips_inline_comment_and_quoted_value() -> None:
    name, body = split_operator_yaml_profile('profile: "p1" # comment\nk: v\n')
    assert name == "p1"
    assert body == "k: v"


def test_merge_operator_yaml_profile_empty_profile_returns_body_only() -> None:
    assert merge_operator_yaml_profile("", "k: 1") == "k: 1\n"
    assert merge_operator_yaml_profile("   ", "") == ""


def test_merge_operator_yaml_profile_body_empty_emits_profile_only() -> None:
    assert merge_operator_yaml_profile("cloud_thin", "") == "profile: cloud_thin\n"


def test_expand_comment_only_operator_merges_example_without_profile_line(tmp_path: Path) -> None:
    ex = tmp_path / "ex.yaml"
    ex.write_text("max_episodes: 3\nbatch_size: 1\n", encoding="utf-8")
    out = expand_profile_only_with_packaged_example("# header only\n", example_path=ex)
    assert "profile:" not in out
    assert "max_episodes: 3" in out


def test_expand_noop_when_example_body_is_empty(tmp_path: Path) -> None:
    ex = tmp_path / "empty_body.yaml"
    ex.write_text("profile: only\n", encoding="utf-8")
    raw = "profile: local\n"
    assert expand_profile_only_with_packaged_example(raw, example_path=ex) == raw
