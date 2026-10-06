"""`PODCAST_DEV_OBS_ENV=0` keeps dev observability out of `podcast_obs` too (operator, 2026-10-06).

The e2e test servers set it. Twin of `tests/unit/podcast_scraper/test_dev_obs_env_switch.py`, which
covers `podcast_scraper.config`; this side covers the `.env.obs.dev` loader AND the auto-discovered
`config/observability.homelab.yaml` the Ops routes read their targets from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_obs import config as obs_config

KEY = "PODCAST_DEV_OBS_SWITCH_TEST_KEY"


def test_podcast_obs_honours_the_same_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env.obs.dev").write_text(f"{KEY}=from-the-dev-obs-file\n")
    monkeypatch.chdir(tmp_path)
    # The loader also skips under pytest; lift that so this test exercises the switch itself.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(KEY, raising=False)

    monkeypatch.setenv("PODCAST_DEV_OBS_ENV", "0")
    obs_config._load_obs_dev_env()
    assert KEY not in obs_config.os.environ

    monkeypatch.delenv("PODCAST_DEV_OBS_ENV")
    try:
        obs_config._load_obs_dev_env()
        assert obs_config.os.environ.get(KEY) == "from-the-dev-obs-file"
    finally:
        monkeypatch.delenv(KEY, raising=False)


def test_the_switch_also_skips_the_committed_homelab_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ops routes read their targets from `ObservabilityConfig.load()`, which auto-discovers the
    TRACKED `config/observability.homelab.yaml`; with the env file alone switched off, a test server
    still showed `"target": "homelab"` and live production gateway spend."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PODCAST_OBS_CONFIG", raising=False)
    monkeypatch.delenv("PODCAST_OBS_TARGET", raising=False)
    assert obs_config._discover_default_config() is not None  # the YAML is there to be found

    monkeypatch.setenv("PODCAST_DEV_OBS_ENV", "0")
    cfg = obs_config.ObservabilityConfig.load()
    assert "homelab" not in cfg.targets
    assert cfg.default_target == "default"
