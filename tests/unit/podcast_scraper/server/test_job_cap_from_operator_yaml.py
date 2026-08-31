"""Job concurrency cap resolves from the operator YAML before the env var (#1888).

Container env is fixed at creation, so an env-only cap cannot be changed without a
redeploy — which is what made the concurrency-headroom question untestable without a
90-minute image cycle. These lock the resolution order and, more importantly, the
blast-radius guard: the new key must be *stripped* by ``Config``, never rejected.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config
from podcast_scraper.server.jobs import max_concurrent_jobs


@pytest.fixture
def op_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", raising=False)
    path = tmp_path / "viewer_operator.yaml"
    path.write_text("profile: prod_dgx_full\n", encoding="utf-8")
    return path


def test_defaults_to_one_when_neither_layer_states_it(op_yaml):
    assert max_concurrent_jobs(op_yaml) == 1


def test_yaml_value_is_used(op_yaml):
    op_yaml.write_text("max_concurrent_pipeline_jobs: 2\n", encoding="utf-8")
    assert max_concurrent_jobs(op_yaml) == 2


def test_yaml_wins_over_env(op_yaml, monkeypatch):
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "4")
    op_yaml.write_text("max_concurrent_pipeline_jobs: 2\n", encoding="utf-8")
    assert max_concurrent_jobs(op_yaml) == 2


def test_env_still_applies_when_yaml_is_silent(op_yaml, monkeypatch):
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "4")
    assert max_concurrent_jobs(op_yaml) == 4


def test_unparsable_yaml_value_falls_back_to_env_and_warns(op_yaml, monkeypatch, caplog):
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "4")
    op_yaml.write_text("max_concurrent_pipeline_jobs: banana\n", encoding="utf-8")
    assert max_concurrent_jobs(op_yaml) == 4
    assert "not an int" in caplog.text


@pytest.mark.parametrize("raw,expected", [("0", 1), ("-3", 1), ("3  # comment", 3)])
def test_value_parsing(op_yaml, raw, expected):
    op_yaml.write_text(f"max_concurrent_pipeline_jobs: {raw}\n", encoding="utf-8")
    assert max_concurrent_jobs(op_yaml) == expected


def test_missing_file_does_not_wedge_admission(tmp_path, monkeypatch):
    """An absent operator file must fall through, not raise — it would stop the queue."""
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "3")
    assert max_concurrent_jobs(tmp_path / "absent.yaml") == 3


def test_none_keeps_legacy_env_only_behaviour(monkeypatch):
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "5")
    assert max_concurrent_jobs(None) == 5
    assert max_concurrent_jobs() == 5


def test_config_strips_the_key_rather_than_rejecting_it():
    """The blast radius: ``Config`` forbids extras, so a rejected key fails EVERY run.

    ``viewer_operator.yaml`` is passed straight to the pipeline CLI, so the cap key
    lands in front of ``Config``. It must be tolerated and dropped.
    """
    assert "max_concurrent_pipeline_jobs" in config.OPERATOR_ONLY_TOP_LEVEL_KEYS
    cfg = config.Config.model_validate(
        {"rss_url": "https://example.com/f.rss", "max_concurrent_pipeline_jobs": 2}
    )
    assert not hasattr(cfg, "max_concurrent_pipeline_jobs")
