"""Unit tests for ``operator_yaml_profile`` (viewer operator merge helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.operator_yaml_profile import (
    expand_profile_only_with_packaged_example,
    merge_operator_yaml_profile,
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
