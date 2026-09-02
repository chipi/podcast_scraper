"""Config loading: env single-target, YAML multi-target, secret-env indirection."""

from __future__ import annotations

import textwrap

import pytest

from podcast_obs import config as obs_config
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


def test_discover_default_config_finds_cwd_homelab_yaml(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero-config dev default: a committed ``config/observability.homelab.yaml`` under the cwd is
    auto-discovered so ``podcast_obs`` needs no ``PODCAST_OBS_CONFIG`` on a developer box."""
    (tmp_path / "config").mkdir()
    yaml = tmp_path / "config" / "observability.homelab.yaml"
    yaml.write_text("default_target: homelab\ntargets:\n  homelab: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert obs_config._discover_default_config() == str(yaml)


def test_discover_default_config_ignores_example_yaml(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery is exact (``observability.homelab.yaml``) — it must NOT latch onto the shipped
    ``observability.example.yaml``. With only the example under cwd, cwd yields nothing and it falls
    through to the real repo-root default (which is the homelab file, never the example)."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "observability.example.yaml").write_text("x: 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    found = obs_config._discover_default_config()
    assert found is None or found.endswith("observability.homelab.yaml")
    assert found is None or not found.endswith("observability.example.yaml")


def test_from_yaml_langfuse_keys_fall_back_to_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config-target probe must pick up the langfuse SDK-native env keys when the YAML omits them
    (secrets never live in the file) — else `podcast_obs traces` is blind while traces flow."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-env")
    config_path = tmp_path / "obs.yaml"
    config_path.write_text(
        textwrap.dedent("""
            default_target: homelab
            targets:
              homelab:
                langfuse:
                  base_url: http://homelab:4000
            """),
        encoding="utf-8",
    )
    t = ObservabilityConfig.from_yaml(config_path).target("homelab")
    assert t.langfuse_public_key == "pk-env"
    assert t.langfuse_secret_key == "sk-env"
    assert t.langfuse_base_url == "http://homelab:4000"  # explicit YAML base_url still wins


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
    monkeypatch.setenv("PODCAST_OBS_GRAFANA_TOKEN", "gt")
    monkeypatch.setenv("PODCAST_OBS_SENTRY_PROJECTS", "a, b ,c")
    monkeypatch.setenv("PODCAST_OBS_ENV_LABEL", "drill")
    target = ObservabilityConfig.from_env().target()
    assert target.timeout == 2.5
    assert target.grafana_token == "gt"
    assert target.sentry_projects == ("a", "b", "c")  # CSV split + trimmed
    assert target.env_label == "drill"


def test_sentry_token_falls_back_to_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # The GlitchTip issue-link pivot reuses the platform's existing SENTRY_AUTH_TOKEN when the
    # PODCAST_OBS_ one isn't set; an explicit PODCAST_OBS_SENTRY_TOKEN still wins.
    _clear_obs_env(monkeypatch)
    monkeypatch.delenv("PODCAST_OBS_SENTRY_TOKEN", raising=False)
    monkeypatch.setenv("SENTRY_AUTH_TOKEN", "gh-secret-tok")
    assert ObservabilityConfig.from_env().target().sentry_token == "gh-secret-tok"
    monkeypatch.setenv("PODCAST_OBS_SENTRY_TOKEN", "explicit")
    assert ObservabilityConfig.from_env().target().sentry_token == "explicit"


def test_obs_dev_env_skipped_under_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    # PYTEST_CURRENT_TEST is set by pytest during a test → the auto-load is a hermetic no-op, so a
    # dev's .env.obs.dev can never leak real backend URLs into the test env.
    import dotenv

    calls: list = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append(a))
    assert "PYTEST_CURRENT_TEST" in __import__("os").environ
    obs_config._load_obs_dev_env()
    assert calls == []


def test_obs_dev_env_loads_from_cwd_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Outside pytest, `podcast_obs serve` in a worktree auto-loads that dir's .env.obs.dev — this
    # is what makes the spawned MCP server zero-config (an MCP client hands it a clean env).
    import dotenv

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.obs.dev").write_text("PODCAST_OBS_UMAMI_URL=http://x\n")
    loaded: list = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda p, **k: loaded.append(str(p)))
    obs_config._load_obs_dev_env()
    assert any(".env.obs.dev" in p for p in loaded)


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


def test_operator_key_from_env(monkeypatch) -> None:
    """The gated probes need a key; ``from_env`` must pick it up under either name."""
    from podcast_obs.config import ObservabilityConfig

    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://x")
    monkeypatch.setenv("PODCAST_OBS_OPERATOR_KEY", "prefixed")
    assert ObservabilityConfig.from_env().target().operator_key == "prefixed"

    monkeypatch.delenv("PODCAST_OBS_OPERATOR_KEY")
    monkeypatch.setenv("APP_OPERATOR_API_KEY", "bare")
    assert ObservabilityConfig.from_env().target().operator_key == "bare"


def test_operator_key_absent_is_none(monkeypatch) -> None:
    from podcast_obs.config import ObservabilityConfig

    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://x")
    monkeypatch.delenv("PODCAST_OBS_OPERATOR_KEY", raising=False)
    monkeypatch.delenv("APP_OPERATOR_API_KEY", raising=False)
    assert ObservabilityConfig.from_env().target().operator_key is None
