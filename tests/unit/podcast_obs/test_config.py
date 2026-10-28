"""Config loading: env single-target, YAML multi-target, secret-env indirection."""

from __future__ import annotations

import textwrap
from pathlib import Path

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
    """Zero-config dev default: a placed ``config/observability.homelab.yaml`` under the cwd is
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


def test_operator_key_from_yaml(tmp_path, monkeypatch) -> None:
    """The YAML path must carry the key too.

    ``load()`` auto-discovers ``config/observability.homelab.yaml`` BEFORE falling back to
    ``from_env``, so a key wired only into ``from_env`` is inert on any box that has the
    committed YAML — which is where the 403s were observed in the first place.
    """
    from podcast_obs.config import ObservabilityConfig

    monkeypatch.delenv("APP_OPERATOR_API_KEY", raising=False)
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(
        "default_target: prod\n"
        "targets:\n"
        "  prod:\n"
        "    api_base: http://x\n"
        "    operator_key: literal-key\n",
        encoding="utf-8",
    )
    assert ObservabilityConfig.load(cfg_path).target("prod").operator_key == "literal-key"


def test_operator_key_from_yaml_env_indirection(tmp_path, monkeypatch) -> None:
    """Secrets stay out of the file: ``operator_key_env`` names the variable."""
    from podcast_obs.config import ObservabilityConfig

    monkeypatch.setenv("MY_OPERATOR_KEY", "from-env-var")
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(
        "default_target: prod\n"
        "targets:\n"
        "  prod:\n"
        "    api_base: http://x\n"
        "    operator_key_env: MY_OPERATOR_KEY\n",
        encoding="utf-8",
    )
    assert ObservabilityConfig.load(cfg_path).target("prod").operator_key == "from-env-var"


def test_operator_key_from_yaml_falls_back_to_bare_name(tmp_path, monkeypatch) -> None:
    from podcast_obs.config import ObservabilityConfig

    monkeypatch.setenv("APP_OPERATOR_API_KEY", "platform-key")
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(
        "default_target: prod\ntargets:\n  prod:\n    api_base: http://x\n", encoding="utf-8"
    )
    assert ObservabilityConfig.load(cfg_path).target("prod").operator_key == "platform-key"


# -- backend read URLs come from deploy-rendered env, not the YAML (#2188) -----------------------
#
# Measured 2026-09-29 inside prod `player-obs-1`: metrics, logs, traces and errors all failed while
# the backends held live data. The YAML path — the one `load()` takes whenever PODCAST_OBS_CONFIG is
# set, i.e. always on prod — read backend URLs from YAML literals ONLY, so the settings the deploy
# renders for every other container never reached obs; the literals it did carry pointed at
# `homelab:9428` / `homelab:3000`, which the tailnet ACL drops for prod.

_PLATFORM_VARS = (
    "PODCAST_OBS_VICTORIALOGS_URL",
    "PODCAST_OBS_VICTORIAMETRICS_URL",
    "PODCAST_OBS_VICTORIATRACES_URL",
    "PODCAST_OBS_GRAFANA_URL",
    "PODCAST_OBS_SENTRY_URL",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "PODCAST_LOGS_PUSH_URL",
    "PODCAST_METRICS_PUSH_URL",
    "PODCAST_SENTRY_DSN_PIPELINE",
    "PODCAST_SENTRY_DSN_API",
)

_NO_URL_YAML = (
    "default_target: prod\ntargets:\n  prod:\n    api_base: http://api:8000\n"
    "    grafana:\n      token_env: PODCAST_OBS_GRAFANA_TOKEN\n"
    "    sentry:\n      org: homelab\n"
)


def _clean_platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _PLATFORM_VARS:
        monkeypatch.delenv(k, raising=False)


def _render_like_the_prod_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The values deploy-player.yml renders (tailnet suffix replaced by example.ts.net)."""
    monkeypatch.setenv("PODCAST_OBS_VICTORIALOGS_URL", "https://vlogs.example.ts.net")
    monkeypatch.setenv("PODCAST_OBS_VICTORIAMETRICS_URL", "https://vm.example.ts.net")
    monkeypatch.setenv("PODCAST_OBS_GRAFANA_URL", "https://grafana.example.ts.net")
    monkeypatch.setenv("PODCAST_OBS_SENTRY_URL", "https://glitchtip.example.ts.net")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://homelab:10428/insert/opentelemetry/v1/traces"
    )


def test_a_yaml_target_without_urls_takes_the_deploy_rendered_ones(tmp_path, monkeypatch) -> None:
    """THE REGRESSION: every one of these came back None, so every source was dead."""
    _clean_platform_env(monkeypatch)
    _render_like_the_prod_deploy(monkeypatch)
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(_NO_URL_YAML, encoding="utf-8")
    t = ObservabilityConfig.load(cfg_path).target("prod")
    assert t.victorialogs_url == "https://vlogs.example.ts.net"
    assert t.victoriametrics_url == "https://vm.example.ts.net"
    assert t.grafana_url == "https://grafana.example.ts.net"
    assert t.sentry_url == "https://glitchtip.example.ts.net"
    # traces: the SAME endpoint the api exports to, reduced to its origin for reading
    assert t.victoriatraces_url == "http://homelab:10428"


def test_both_config_paths_derive_urls_identically(tmp_path, monkeypatch) -> None:
    """One derivation. If the YAML path ever drifts from from_env, this catches it."""
    _clean_platform_env(monkeypatch)
    _render_like_the_prod_deploy(monkeypatch)
    monkeypatch.setattr(obs_config, "_load_obs_dev_env", lambda: None)
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(_NO_URL_YAML, encoding="utf-8")
    y = ObservabilityConfig.load(cfg_path).target("prod")
    e = ObservabilityConfig.from_env().target()
    for field in (
        "victorialogs_url",
        "victoriametrics_url",
        "victoriatraces_url",
        "grafana_url",
        "sentry_url",
    ):
        assert getattr(y, field) == getattr(e, field), field


def test_a_literal_url_in_the_yaml_still_wins(tmp_path, monkeypatch) -> None:
    """observability.local.yaml hardcodes localhost / homelab URLs; they must keep working."""
    _clean_platform_env(monkeypatch)
    _render_like_the_prod_deploy(monkeypatch)
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(
        "default_target: local\ntargets:\n  local:\n    api_base: http://localhost:8000\n"
        "    victoria:\n      logs_url: http://homelab:9428\n",
        encoding="utf-8",
    )
    t = ObservabilityConfig.load(cfg_path).target("local")
    assert t.victorialogs_url == "http://homelab:9428"
    assert t.victoriametrics_url == "https://vm.example.ts.net"  # the omitted one still falls back


def test_nothing_rendered_degrades_to_not_configured_not_a_crash(tmp_path, monkeypatch) -> None:
    _clean_platform_env(monkeypatch)
    cfg_path = tmp_path / "obs.yaml"
    cfg_path.write_text(_NO_URL_YAML, encoding="utf-8")
    t = ObservabilityConfig.load(cfg_path).target("prod")
    assert t.victorialogs_url is None
    assert t.victoriametrics_url is None
    assert t.victoriatraces_url is None


# -- the shipped files that must agree with each other -------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (_repo_root() / rel).read_text(encoding="utf-8")


def test_the_obs_service_env_contract_matches_compose() -> None:
    """THE SILENT SEAM: compose passes a container only what its `environment:` block names.

    The deployment renders the obs read URLs and tokens, and that side is not in this repository.
    So the names the obs service accepts are published in ``config/deploy_contract.json`` and kept
    EXACTLY equal to compose here; the deployment tests that everything it renders is in the list.
    """
    import json

    import yaml

    svc = yaml.safe_load(_read("compose/docker-compose.player-public.yml"))["services"]["obs"]
    declared = set(svc.get("environment") or {})
    contract = set(json.loads(_read("config/deploy_contract.json"))["obs_service_env"])
    assert declared == contract, (
        "config/deploy_contract.json obs_service_env has drifted from the obs service in "
        "compose/docker-compose.player-public.yml — "
        f"in compose only: {sorted(declared - contract)}; in the contract only: "
        f"{sorted(contract - declared)}. Update the contract to match compose."
    )


def test_operator_key_from_a_mounted_secret_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prod obs reads its GET-only operator key from a tmpfs secret file (the obs image has no
    secrets shim), so the key never sits in an env file on the box's disk (2026-10-03)."""
    monkeypatch.delenv("APP_OPERATOR_API_KEY", raising=False)
    secret = tmp_path / "app_operator_read_key"
    secret.write_text("read-key-value\n", encoding="utf-8")
    config_path = tmp_path / "obs.yaml"
    config_path.write_text(
        textwrap.dedent(f"""
            default_target: prod
            targets:
              prod:
                api_base: http://api:8000
                operator_key_file: {secret}
              missing:
                api_base: http://api:8000
                operator_key_file: {tmp_path / "absent"}
            """),
        encoding="utf-8",
    )
    cfg = ObservabilityConfig.from_yaml(config_path)
    assert cfg.target("prod").operator_key == "read-key-value"
    assert cfg.target("missing").operator_key is None
