"""Unit tests for packaged profile name listing (operator-config API)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_list_packaged_profile_names_includes_repo_profiles() -> None:
    from podcast_scraper.server import profile_presets

    names = profile_presets.list_packaged_profile_names()
    assert len(names) >= 3
    assert names == sorted(names)
    assert "cloud_balanced" in names
    assert "profile_freeze" not in names  # profile_freeze.example stem ends with .example


def test_profile_directories_prefers_cwd_then_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.server import profile_presets

    cwd_prof = tmp_path / "config" / "profiles"
    cwd_prof.mkdir(parents=True)
    (cwd_prof / "from_cwd_only.yaml").write_text("x: 1\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    names = profile_presets.list_packaged_profile_names()
    assert "from_cwd_only" in names
