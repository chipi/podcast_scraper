"""`PODCAST_DEV_OBS_ENV=0` keeps the dev observability file out of a process.

Test servers (both Playwright e2e APIs) run `podcast_scraper.cli serve` from a checkout that has
the gitignored `.env.obs.dev`, which points at the homelab and production telemetry. They must never
load it (operator, 2026-10-06): the viewer's Ops tab was reading live production gateway spend
during e2e runs, and every error was being reported to the homelab GlitchTip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import podcast_scraper.cache as cache_mod
from podcast_scraper import config

KEY = "PODCAST_DEV_OBS_SWITCH_TEST_KEY"


@pytest.fixture
def project_with_obs_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".env.obs.dev").write_text(f"{KEY}=from-the-dev-obs-file\n")
    monkeypatch.setattr(cache_mod, "get_project_root", lambda: tmp_path)
    monkeypatch.delenv(KEY, raising=False)
    return tmp_path


def test_the_dev_obs_file_loads_by_default(
    project_with_obs_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PODCAST_DEV_OBS_ENV", raising=False)
    try:
        assert config._load_dev_obs_env() is True
        assert config.os.environ.get(KEY) == "from-the-dev-obs-file"
    finally:
        monkeypatch.delenv(KEY, raising=False)


def test_the_switch_keeps_it_out(
    project_with_obs_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_DEV_OBS_ENV", "0")
    assert config._load_dev_obs_env() is False
    assert KEY not in config.os.environ
