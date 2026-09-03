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
    """One deploy the control plane can observe. ``api_base`` alone reaches the unauthenticated
    probes (health/version); ``operator_key`` is additionally required for the operator-gated
    ones, and the rest wire the external sources.

    ``runs`` used to be credential-free and this docstring used to say so. It stopped being true
    when ``/api/jobs`` came under the operator gate (``app_operator_guard``, #1071/#1128) and the
    probe was never updated, so ``ops summary`` reported ``runs``/``cache_stats``/``enrichment_*``
    as *failed* against a perfectly healthy deploy — a monitor that cries wolf during exactly the
    long pipeline runs it exists to watch."""

    name: str
    api_base: Optional[str] = None
    #: Operator API key for the gated probes. Same value the API validates as ``X-Operator-Key``
    #: (``APP_OPERATOR_API_KEY``). Set per-target in YAML (``operator_key`` /
    #: ``operator_key_env``) or via ``PODCAST_OBS_OPERATOR_KEY``; both fall back to the bare
    #: ``APP_OPERATOR_API_KEY``.
    #:
    #: Trap the bare fallback carries: with ``.env.obs.dev`` auto-loaded, a LOCAL server's key is
    #: sent to whatever ``api_base`` the target points at. Against a remote deploy that is a 403
    #: (harmless) plus a credential leaving the box it belongs to (less so). Prefer an explicit
    #: per-target ``operator_key_env`` whenever more than one deploy is configured.
    operator_key: Optional[str] = None
    github_repo: Optional[str] = DEFAULT_GITHUB_REPO
    github_token: Optional[str] = None
    sentry_org: Optional[str] = None
    sentry_projects: tuple[str, ...] = ()
    sentry_token: Optional[str] = None
    sentry_environment: str = "prod"
    # Errors backend base URL. Default = Sentry SaaS; set to a self-hosted GlitchTip
    # (e.g. http://homelab:8090) — Sentry-API-compatible, so only the base URL changes.
    sentry_url: Optional[str] = None
    grafana_url: Optional[str] = None
    grafana_token: Optional[str] = None  # Grafana service-account token (alerting API)
    # Current self-hosted stack (homelab): VictoriaLogs (LogsQL), VictoriaMetrics (PromQL),
    # VictoriaTraces (Jaeger API). Tailnet-reachable; auth via optional bearer token.
    victorialogs_url: Optional[str] = None
    victoriametrics_url: Optional[str] = None
    victoriatraces_url: Optional[str] = None
    victoria_token: Optional[str] = None  # optional bearer for all three (if fronted by auth)
    # Langfuse public API (#1052) — same key pair the pipeline traces with
    # (SDK-native LANGFUSE_*); the probe only *reads* recent traces (Basic auth).
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None  # unset → Langfuse Cloud
    # Umami (ADR-126) — the user-action lens for the operator surface. Cookieless, self-hosted. The
    # website_id is per-environment (operator-dev vs operator-prod) exactly like the player's
    # VITE_UMAMI_WEBSITE_ID; reading needs admin auth (token, or username+password → login token).
    umami_url: Optional[str] = None
    umami_website_id: Optional[str] = None
    umami_token: Optional[str] = None
    umami_username: Optional[str] = None
    umami_password: Optional[str] = None
    env_label: str = "prod"  # the deploy's metrics ``env`` label (PODCAST_ENV)
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
        """Resolve config in precedence order, then build:

        1. explicit ``path`` arg (a caller passed it),
        2. ``PODCAST_OBS_CONFIG`` env var,
        3. the committed dev default ``config/observability.homelab.yaml`` if present,
        4. a single target from ``PODCAST_OBS_*`` / platform env (``from_env``).

        Step 3 is what makes ``podcast_obs`` zero-config on a dev box: the multi-target homelab
        YAML (correct org/projects/URLs) is auto-discovered so nobody has to remember to export
        ``PODCAST_OBS_CONFIG``. The old failure was skipping straight to ``from_env``, which then
        read stale/wrong ``PODCAST_OBS_SENTRY_*`` from ``.env`` and 401'd against a nonexistent org.
        """
        _load_obs_dev_env()  # zero-config: pick up the worktree's .env.obs.dev if present
        path = path or os.environ.get(f"{ENV_PREFIX}CONFIG")
        # Auto-discovery is a dev-machine convenience; skip under pytest so the env-path tests stay
        # hermetic (they run from the repo cwd, where the committed YAML would otherwise be found) —
        # same rationale as ``_load_obs_dev_env``; ``_discover_default_config`` stays testable.
        if not path and not os.environ.get("PYTEST_CURRENT_TEST"):
            path = _discover_default_config()
        if path:
            return cls.from_yaml(path)
        return cls.from_env()

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Build a single-target config from ``PODCAST_OBS_*`` (+ bare ``LANGFUSE_*``) env vars.

        The read endpoints also fall back to the SAME env vars the platform SHIPS telemetry with
        (``PODCAST_LOGS_PUSH_URL`` / ``PODCAST_METRICS_PUSH_URL`` / ``OTEL_EXPORTER_OTLP_TRACES_
        ENDPOINT`` / ``PODCAST_SENTRY_DSN_*``): a control-plane process that inherits the app's
        observability env (dev via ``.env.obs.dev``, prod via the injected env) can observe the same
        backends with zero extra ``PODCAST_OBS_*`` config — no separate URLs to keep in sync.
        """
        _load_obs_dev_env()  # zero-config: pick up the worktree's .env.obs.dev if present
        name = os.environ.get(f"{ENV_PREFIX}TARGET", "default")
        projects = _split_csv(_env("SENTRY_PROJECTS"))
        _dsn = _bare("PODCAST_SENTRY_DSN_PIPELINE") or _bare("PODCAST_SENTRY_DSN_API")
        target = TargetConfig(
            name=name,
            api_base=_env("API_BASE"),
            operator_key=_env("OPERATOR_KEY") or _bare("APP_OPERATOR_API_KEY"),
            github_repo=_env("GITHUB_REPO") or DEFAULT_GITHUB_REPO,
            github_token=_env("GITHUB_TOKEN"),
            sentry_org=_env("SENTRY_ORG"),
            sentry_projects=projects,
            # Falls back to the platform's own SENTRY_AUTH_TOKEN (the existing GH secret) so the
            # GlitchTip issue-link (permalink) pivot works from the same env, no PODCAST_OBS_ dup.
            sentry_token=_env("SENTRY_TOKEN") or _bare("SENTRY_AUTH_TOKEN"),
            sentry_environment=_env("SENTRY_ENV") or "prod",
            sentry_url=_env("SENTRY_URL") or _origin(_dsn),
            grafana_url=_env("GRAFANA_URL"),
            grafana_token=_env("GRAFANA_TOKEN"),
            victorialogs_url=_env("VICTORIALOGS_URL") or _origin(_bare("PODCAST_LOGS_PUSH_URL")),
            victoriametrics_url=(
                _env("VICTORIAMETRICS_URL") or _origin(_bare("PODCAST_METRICS_PUSH_URL"))
            ),
            victoriatraces_url=(
                _env("VICTORIATRACES_URL") or _origin(_bare("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
            ),
            victoria_token=_env("VICTORIA_TOKEN"),
            # Langfuse uses its SDK-native bare names (not the PODCAST_OBS_ prefix) so the
            # same keys the pipeline traces with drive the probe — no duplicate config.
            langfuse_public_key=_bare("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=_bare("LANGFUSE_SECRET_KEY"),
            langfuse_base_url=_bare("LANGFUSE_BASE_URL") or _bare("LANGFUSE_HOST"),
            # Umami — env-driven per environment (dev→operator-dev site, prod→operator-prod), the
            # same shape as the player's VITE_UMAMI_*; url falls back to the ingest script's origin.
            umami_url=_env("UMAMI_URL") or _origin(_bare("VITE_UMAMI_SRC")),
            umami_website_id=_env("UMAMI_WEBSITE_ID"),
            umami_token=_env("UMAMI_TOKEN"),
            umami_username=_env("UMAMI_USERNAME"),
            umami_password=_env("UMAMI_PASSWORD"),
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


def _origin(url: Optional[str]) -> Optional[str]:
    """``scheme://host[:port]`` from a platform ship URL/DSN — drops path AND userinfo.

    The app's push endpoint (``…:9428/insert/jsonline``) and a Sentry/GlitchTip DSN
    (``http://<key>@host:8090/1``) both carry the host the read sources need but wrapped in a path
    or credentials. Reducing to the origin lets ``from_env`` reuse the exact vars the platform ships
    telemetry with, instead of a parallel ``PODCAST_OBS_*`` URL set that can drift out of sync.
    """
    if not url:
        return None
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def _discover_default_config() -> Optional[str]:
    """Return the committed dev-default multi-target YAML if it exists, else ``None``.

    Looks for ``config/observability.homelab.yaml`` under the cwd (worktree root when an agent
    launches the tool there) then the editable repo root (two levels up from this file). This is
    the zero-config default for a developer machine: the file is TRACKED (correct homelab
    org/projects/URLs, secrets via ``*_env`` indirection), so discovering it needs no per-machine
    setup. Deliberately exact — never globs ``observability.*.yaml`` so the shipped
    ``observability.example.yaml`` can't be picked up by accident.
    """
    repo_root = Path(__file__).resolve().parents[2]
    for base in (Path.cwd(), repo_root):
        candidate = base / "config" / "observability.homelab.yaml"
        try:
            if candidate.is_file():
                return str(candidate)
        except Exception:  # pragma: no cover — dev convenience only, never fail config load
            continue
    return None


def _load_obs_dev_env() -> None:
    """Dev convenience: auto-load ``.env.obs.dev`` so ``podcast_obs serve`` in a worktree is
    zero-config. An agent's MCP client spawns the server with a CLEAN env, so without this every
    source reads unconfigured. Mirrors ``podcast_scraper.config`` (same gitignored file); the prod
    image has no such file so it no-ops there. Skipped under pytest; ``override=False`` so an
    explicit shell env still wins. Self-contained — podcast_obs stays light-dep (no app import).
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover — dotenv is optional for the light-dep core
        return
    # Repo root is two levels up from src/podcast_obs/config.py (editable/dev layout); also try cwd,
    # which is the worktree root when an agent launches `podcast_obs serve` there.
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (Path.cwd() / ".env.obs.dev", repo_root / ".env.obs.dev"):
        try:
            if candidate.exists():
                load_dotenv(candidate, override=False)
                return
        except Exception:  # pragma: no cover — dev convenience only, never fail config load
            continue


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
    victoria = spec.get("victoria") or {}
    umami = spec.get("umami") or {}
    projects = sentry.get("projects") or []
    if isinstance(projects, str):
        projects = _split_csv(projects)
    raw_timeout = spec.get("timeout")
    timeout = _as_float(str(raw_timeout) if raw_timeout else None, DEFAULT_TIMEOUT)
    return TargetConfig(
        name=name,
        api_base=spec.get("api_base"),
        # Operator key for the gated probes (jobs / ops / enrichment). Same ``<key>_env``
        # indirection as every other secret here, plus the SDK-native bare name as a fallback so
        # a target that omits it still works on a box that exports the platform's own key.
        #
        # This mapping was missing when ``operator_key`` was first added and the omission made the
        # whole fix INERT in the common path: ``load()`` auto-discovers
        # ``config/observability.homelab.yaml`` at precedence step 3, BEFORE ``from_env`` at step
        # 4, so on any dev box the YAML branch is what builds the target — and it hard-coded the
        # key to None. The 403s the fix was written for went right on happening.
        operator_key=_secret(spec, "operator_key") or _bare("APP_OPERATOR_API_KEY"),
        github_repo=github.get("repo") or DEFAULT_GITHUB_REPO,
        github_token=_secret(github, "token"),
        sentry_org=sentry.get("org"),
        sentry_projects=tuple(projects),
        sentry_token=_secret(sentry, "token"),
        sentry_environment=sentry.get("environment") or "prod",
        sentry_url=sentry.get("url"),
        grafana_url=grafana.get("url"),
        grafana_token=_secret(grafana, "token"),
        victorialogs_url=victoria.get("logs_url"),
        victoriametrics_url=victoria.get("metrics_url"),
        victoriatraces_url=victoria.get("traces_url"),
        victoria_token=_secret(victoria, "token"),
        # Fall back to the langfuse SDK-native env vars (the keys the pipeline traces with) when the
        # YAML omits them — secrets never live in the config file, so a config-target probe still
        # picks up LANGFUSE_PUBLIC_KEY / SECRET_KEY, matching the default (env) target.
        langfuse_public_key=_secret(langfuse, "public_key") or _bare("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_secret(langfuse, "secret_key") or _bare("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=(
            langfuse.get("base_url") or _bare("LANGFUSE_BASE_URL") or _bare("LANGFUSE_HOST")
        ),
        umami_url=umami.get("url"),
        umami_website_id=umami.get("website_id"),
        umami_token=_secret(umami, "token"),
        umami_username=umami.get("username"),
        umami_password=_secret(umami, "password"),
        env_label=spec.get("env_label") or "prod",
        timeout=timeout,
    )
