"""Configuration for the observability control plane.

Target-agnostic by design: you point the control plane at *any* deploy (your local stack,
prod over Tailscale, a drill, another box) and it observes whatever's reachable. Two ways
to configure:

- **Env** (single target) — ``PODCAST_OBS_*`` vars. Ideal for a one-target container.
- **YAML** (multi-target) — ``PODCAST_OBS_CONFIG=/path/to/config.yaml`` with a ``targets``
  map, so ``--target local`` / ``--target prod`` switch between deploys on a dev box.

Secrets in YAML use ``<field>_env: ENV_VAR_NAME`` indirection so tokens stay out of the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ENV_PREFIX = "PODCAST_OBS_"
DEFAULT_GITHUB_REPO = "chipi/podcast_scraper"
DEFAULT_TIMEOUT = 10.0


class ObservabilityConfigError(RuntimeError):
    """Raised on a missing/unknown target or a required-but-unset field."""


@dataclass(frozen=True)
class TargetConfig:
    """One deploy the control plane can observe. Only ``api_base`` is needed for the
    credential-free probes (health/version/runs); the rest wire the external sources."""

    name: str
    api_base: Optional[str] = None
    github_repo: Optional[str] = DEFAULT_GITHUB_REPO
    github_token: Optional[str] = None
    sentry_org: Optional[str] = None
    sentry_projects: tuple[str, ...] = ()
    sentry_token: Optional[str] = None
    sentry_environment: str = "prod"
    grafana_url: Optional[str] = None
    grafana_token: Optional[str] = None  # Grafana service-account token (alerting API)
    loki_url: Optional[str] = None
    loki_user: Optional[str] = None
    loki_token: Optional[str] = (
        None  # Loki access-policy token (logs:read); falls back to grafana_token
    )
    # Langfuse public API (#1052) — same key pair the pipeline traces with
    # (SDK-native LANGFUSE_*); the probe only *reads* recent traces (Basic auth).
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None  # unset → Langfuse Cloud
    env_label: str = "prod"  # the deploy's Loki/metrics ``env`` label (PODCAST_ENV)
    timeout: float = DEFAULT_TIMEOUT

    def require(self, attr: str, hint: str) -> Any:
        """Return ``attr``'s value or raise with a clear "not configured" message."""
        value = getattr(self, attr, None)
        if not value:
            raise ObservabilityConfigError(f"target {self.name!r}: {attr} not configured ({hint})")
        return value


@dataclass(frozen=True)
class ObservabilityConfig:
    """A set of named targets plus the default one to use when ``--target`` is omitted."""

    targets: dict[str, TargetConfig]
    default_target: str

    def target(self, name: Optional[str] = None) -> TargetConfig:
        """The named target (or the default); raises if the name isn't configured."""
        key = name or self.default_target
        if key not in self.targets:
            have = ", ".join(sorted(self.targets)) or "none"
            raise ObservabilityConfigError(f"unknown target {key!r} (configured: {have})")
        return self.targets[key]

    # --- loaders -------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str | os.PathLike[str]] = None) -> "ObservabilityConfig":
        """YAML if ``path``/``PODCAST_OBS_CONFIG`` is set, else a single target from env."""
        path = path or os.environ.get(f"{ENV_PREFIX}CONFIG")
        if path:
            return cls.from_yaml(path)
        return cls.from_env()

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Build a single-target config from ``PODCAST_OBS_*`` (+ bare ``LANGFUSE_*``) env vars."""
        name = os.environ.get(f"{ENV_PREFIX}TARGET", "default")
        projects = _split_csv(_env("SENTRY_PROJECTS"))
        target = TargetConfig(
            name=name,
            api_base=_env("API_BASE"),
            github_repo=_env("GITHUB_REPO") or DEFAULT_GITHUB_REPO,
            github_token=_env("GITHUB_TOKEN"),
            sentry_org=_env("SENTRY_ORG"),
            sentry_projects=projects,
            sentry_token=_env("SENTRY_TOKEN"),
            sentry_environment=_env("SENTRY_ENV") or "prod",
            grafana_url=_env("GRAFANA_URL"),
            grafana_token=_env("GRAFANA_TOKEN"),
            loki_url=_env("LOKI_URL"),
            loki_user=_env("LOKI_USER"),
            loki_token=_env("LOKI_TOKEN"),
            # Langfuse uses its SDK-native bare names (not the PODCAST_OBS_ prefix) so the
            # same keys the pipeline traces with drive the probe — no duplicate config.
            langfuse_public_key=_bare("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=_bare("LANGFUSE_SECRET_KEY"),
            langfuse_base_url=_bare("LANGFUSE_BASE_URL") or _bare("LANGFUSE_HOST"),
            env_label=_env("ENV_LABEL") or "prod",
            timeout=_as_float(_env("TIMEOUT"), DEFAULT_TIMEOUT),
        )
        return cls(targets={name: target}, default_target=name)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "ObservabilityConfig":
        """Build a multi-target config from a YAML file with a ``targets`` mapping."""
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        targets_raw = raw.get("targets")
        if not isinstance(targets_raw, dict) or not targets_raw:
            raise ObservabilityConfigError(f"{path}: no 'targets' mapping found")
        targets = {name: _target_from_yaml(name, spec or {}) for name, spec in targets_raw.items()}
        default = raw.get("default_target") or next(iter(targets))
        if default not in targets:
            raise ObservabilityConfigError(f"{path}: default_target {default!r} not in targets")
        return cls(targets=targets, default_target=default)


# --- helpers -----------------------------------------------------------------------


def _env(suffix: str) -> Optional[str]:
    value = os.environ.get(f"{ENV_PREFIX}{suffix}")
    return value if value else None


def _bare(name: str) -> Optional[str]:
    """Read an un-prefixed env var (for third-party SDK-native names like LANGFUSE_*)."""
    value = os.environ.get(name)
    return value if value else None


def _split_csv(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value else default
    except (TypeError, ValueError):
        return default


def _secret(spec: dict, key: str) -> Optional[str]:
    """Resolve ``key`` from a literal value or ``<key>_env`` env-var indirection."""
    if not isinstance(spec, dict):
        return None
    if spec.get(key):
        return str(spec[key])
    env_name = spec.get(f"{key}_env")
    return os.environ.get(env_name) if env_name else None


def _target_from_yaml(name: str, spec: dict) -> TargetConfig:
    github = spec.get("github") or {}
    sentry = spec.get("sentry") or {}
    grafana = spec.get("grafana") or {}
    langfuse = spec.get("langfuse") or {}
    projects = sentry.get("projects") or []
    if isinstance(projects, str):
        projects = _split_csv(projects)
    raw_timeout = spec.get("timeout")
    timeout = _as_float(str(raw_timeout) if raw_timeout else None, DEFAULT_TIMEOUT)
    return TargetConfig(
        name=name,
        api_base=spec.get("api_base"),
        github_repo=github.get("repo") or DEFAULT_GITHUB_REPO,
        github_token=_secret(github, "token"),
        sentry_org=sentry.get("org"),
        sentry_projects=tuple(projects),
        sentry_token=_secret(sentry, "token"),
        sentry_environment=sentry.get("environment") or "prod",
        grafana_url=grafana.get("url"),
        grafana_token=_secret(grafana, "token"),
        loki_url=grafana.get("loki_url"),
        loki_user=grafana.get("loki_user"),
        loki_token=_secret(grafana, "loki_token"),
        langfuse_public_key=_secret(langfuse, "public_key"),
        langfuse_secret_key=_secret(langfuse, "secret_key"),
        langfuse_base_url=langfuse.get("base_url"),
        env_label=spec.get("env_label") or "prod",
        timeout=timeout,
    )
