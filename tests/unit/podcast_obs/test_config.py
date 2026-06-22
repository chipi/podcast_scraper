"""Config loading: env single-target, YAML multi-target, secret-env indirection."""

from __future__ import annotations

import textwrap

import pytest

from podcast_obs.config import (
    DEFAULT_GITHUB_REPO,
    ObservabilityConfig,
    ObservabilityConfigError,
    TargetConfig,
)


def _clear_obs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("PODCAST_OBS_"):
            monkeypatch.delenv(key, raising=False)


def test_from_env_single_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_TARGET", "local")
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://localhost:8080")
    cfg = ObservabilityConfig.load()  # no PODCAST_OBS_CONFIG -> env path
    target = cfg.target()
    assert target.name == "local"
    assert target.api_base == "http://localhost:8080"
    assert target.github_repo == DEFAULT_GITHUB_REPO


def test_unknown_target_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    cfg = ObservabilityConfig.from_env()
    with pytest.raises(ObservabilityConfigError):
        cfg.target("does-not-exist")


def test_require_missing_field() -> None:
    target = TargetConfig(name="t")
    with pytest.raises(ObservabilityConfigError):
        target.require("sentry_token", "set a Sentry token")


def test_from_yaml_multitarget_and_secret_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_GH_TOKEN", "gh-secret-value")
    config_path = tmp_path / "obs.yaml"
    config_path.write_text(
        textwrap.dedent("""
            default_target: prod
            targets:
              local:
                api_base: http://localhost:8080
              prod:
                api_base: https://prod-podcast.example.ts.net
                github:
                  repo: chipi/podcast_scraper
                  token_env: MY_GH_TOKEN
                sentry:
                  org: acme
                  projects: [api, pipeline]
                  environment: prod
            """),
        encoding="utf-8",
    )
    cfg = ObservabilityConfig.from_yaml(config_path)
    assert cfg.default_target == "prod"
    assert set(cfg.targets) == {"local", "prod"}
    prod = cfg.target("prod")
    assert prod.github_token == "gh-secret-value"  # resolved via token_env indirection
    assert prod.sentry_projects == ("api", "pipeline")
    local_base = cfg.target("local").api_base
    assert local_base is not None and local_base.endswith(":8080")


def test_from_yaml_without_targets_raises(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("unrelated: true\n", encoding="utf-8")
    with pytest.raises(ObservabilityConfigError):
        ObservabilityConfig.from_yaml(config_path)


def test_from_yaml_default_target_not_in_targets_raises(tmp_path) -> None:
    config_path = tmp_path / "obs.yaml"
    config_path.write_text(
        textwrap.dedent("""
            default_target: ghost
            targets:
              local:
                api_base: http://localhost:8080
            """),
        encoding="utf-8",
    )
    with pytest.raises(ObservabilityConfigError):
        ObservabilityConfig.from_yaml(config_path)


def test_from_env_external_source_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_TIMEOUT", "2.5")
    monkeypatch.setenv("PODCAST_OBS_LOKI_TOKEN", "lt")
    monkeypatch.setenv("PODCAST_OBS_SENTRY_PROJECTS", "a, b ,c")
    monkeypatch.setenv("PODCAST_OBS_ENV_LABEL", "drill")
    target = ObservabilityConfig.from_env().target()
    assert target.timeout == 2.5
    assert target.loki_token == "lt"
    assert target.sentry_projects == ("a", "b", "c")  # CSV split + trimmed
    assert target.env_label == "drill"


def test_from_env_bad_timeout_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_TIMEOUT", "notanumber")
    assert ObservabilityConfig.from_env().target().timeout == 10.0  # DEFAULT_TIMEOUT


def test_from_yaml_inline_token_and_csv_projects(tmp_path) -> None:
    config_path = tmp_path / "obs.yaml"
    config_path.write_text(
        textwrap.dedent("""
            default_target: prod
            targets:
              prod:
                api_base: https://prod.example
                github:
                  token: inline-gh-token
                sentry:
                  org: acme
                  projects: "x,y,z"
            """),
        encoding="utf-8",
    )
    target = ObservabilityConfig.from_yaml(config_path).target("prod")
    assert target.github_token == "inline-gh-token"  # literal (not _env) path
    assert target.sentry_projects == ("x", "y", "z")  # string-form projects split
