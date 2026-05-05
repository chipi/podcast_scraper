"""Docker Compose pipeline job factory (#660 Phase 2).

When ``PODCAST_PIPELINE_EXEC_MODE=docker``, ``create_app`` attaches
``app.state.jobs_subprocess_factory`` so ``POST /api/jobs`` runs the CLI inside the
``pipeline`` or ``pipeline-llm`` service instead of ``sys.executable`` in the API container.

Requires:
  - Docker CLI + Compose v2 plugin in the API image (see ``docker/api/Dockerfile``).
  - Host Docker socket mounted into the API container (e.g. ``/var/run/docker.sock``).
  - ``PODCAST_DOCKER_PROJECT_DIR``: absolute path to the repository root **as seen by the API
    process** (with ``compose/docker-compose.jobs-docker.yml``, the stack bind-mounts the host
    repo at ``/podcast_repo`` and sets this env to ``/podcast_repo`` inside ``api``).
  - Operator YAML must include ``pipeline_install_extras: ml`` or ``pipeline_install_extras: llm``
    when this mode is enabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from podcast_scraper.server.operator_yaml_profile import parse_pipeline_install_extras

logger = logging.getLogger(__name__)


def _project_dir() -> Path:
    raw = os.environ.get("PODCAST_DOCKER_PROJECT_DIR", "").strip()
    if not raw:
        raise RuntimeError(
            "PODCAST_PIPELINE_EXEC_MODE=docker requires PODCAST_DOCKER_PROJECT_DIR "
            "(absolute repo root path visible to the Docker daemon)."
        )
    return Path(raw).expanduser().resolve()


def _compose_files() -> list[str]:
    """Parse ``PODCAST_DOCKER_COMPOSE_FILES`` into a list of repo-relative paths.

    Each entry MUST be a repository-relative path without ``..`` segments and
    must not be absolute. This guards against an env-var attacker (or a
    mis-configured operator) pointing the factory at ``/etc/hosts`` or at a
    compose file outside the project root. Resolution against
    ``PODCAST_DOCKER_PROJECT_DIR`` happens in the caller.
    """
    raw = os.environ.get("PODCAST_DOCKER_COMPOSE_FILES", "compose/docker-compose.stack.yml").strip()
    entries = [p.strip() for p in raw.split(",") if p.strip()]
    for entry in entries:
        candidate = Path(entry)
        if candidate.is_absolute():
            raise RuntimeError(
                f"PODCAST_DOCKER_COMPOSE_FILES entry must be repo-relative, got absolute: {entry}"
            )
        # Normalise and reject any ``..`` traversal. Running ``Path.resolve``
        # here would evaluate against the API container's cwd, not the project
        # root, so stay textual instead.
        parts = candidate.parts
        if any(part == ".." for part in parts):
            raise RuntimeError(f"PODCAST_DOCKER_COMPOSE_FILES entry contains '..': {entry}")
    return entries


def _resolve_compose_path(project: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``project`` and assert it stays inside the tree.

    Pure belt-and-suspenders against symlink or path-construction shenanigans
    that slipped past :func:`_compose_files`. ``project`` is already resolved
    by :func:`_project_dir`, so any escape here is a real bug, not a footgun.
    """
    resolved = (project / rel).resolve()
    try:
        resolved.relative_to(project)
    except ValueError as exc:
        raise RuntimeError(
            f"Compose file {rel!r} escapes PODCAST_DOCKER_PROJECT_DIR ({project})."
        ) from exc
    return resolved


def _service_for_extras(extras: str) -> tuple[str, str]:
    """Return (compose_service_name, compose_profile_name)."""
    if extras == "llm":
        return "pipeline-llm", "pipeline-llm"
    return "pipeline", "pipeline"


def _cli_argv_tail(argv: Sequence[str]) -> list[str]:
    """Strip interpreter ``… -m podcast_scraper.cli`` prefix from *argv*."""
    seq = list(argv)
    for i in range(max(0, len(seq) - 2)):
        if seq[i + 1] == "-m" and i + 2 < len(seq) and seq[i + 2] == "podcast_scraper.cli":
            return seq[i + 3 :]
    return seq[1:] if len(seq) > 1 else []


def _env_default_pipeline_install_extras() -> str | None:
    """Operator-set fallback for ``pipeline_install_extras`` from env.

    When the corpus's operator YAML doesn't declare ``pipeline_install_extras``
    (e.g. fresh corpus, first-run prod), fall back to
    ``PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS`` if the host has set it. The
    prod overlay (``compose/docker-compose.prod.yml``) sets it to ``llm`` so
    first-run UX on the published image set works without operator
    YAML editing. Returns ``None`` if env unset or set to anything other
    than ``ml`` / ``llm`` (caller falls back to the strict YAML-required
    error path so operators see a clear failure rather than silently
    picking a wrong service).
    """
    raw = os.environ.get("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "").strip()
    if raw in ("ml", "llm"):
        return raw
    return None


def assert_operator_pipeline_extras(operator_yaml: Path) -> str:
    """Ensure operator YAML declares ``pipeline_install_extras: ml`` or ``llm``; return value.

    Falls back to ``PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS`` env var when
    the YAML omits the field (so first-run prod doesn't require operator
    YAML editing — see ``_env_default_pipeline_install_extras``).
    """
    # codeql[py/path-injection] -- Docker enqueue: path from viewer_operator_extras_source
    # only (Type 1).
    text = operator_yaml.read_text(encoding="utf-8", errors="replace")
    extras = parse_pipeline_install_extras(text)
    if extras is None:
        env_default = _env_default_pipeline_install_extras()
        if env_default is not None:
            return env_default
    if extras not in ("ml", "llm"):
        raise ValueError(
            "Docker pipeline jobs require top-level pipeline_install_extras: ml "
            "or pipeline_install_extras: llm in the operator YAML "
            f"({operator_yaml}); set PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS "
            "on the api container env to provide a host-wide default."
        )
    return extras


def validate_operator_pipeline_extras(operator_yaml: Path, pipe_mode: str) -> str | None:
    """Mode-aware validator for ``pipeline_install_extras`` (#666 review #13).

    * The *value* is validated symmetrically across modes: if the operator
      YAML sets ``pipeline_install_extras`` to anything other than ``ml``
      or ``llm``, reject in both Docker and subprocess mode.
    * The *presence* requirement stays mode-specific:
        - ``pipe_mode == "docker"``: the field is mandatory because the
          API must pick a compose service (``pipeline`` vs
          ``pipeline-llm``) whose image matches.
        - subprocess mode: the field is optional; the job runs inside the
          API container using whatever extras were installed at build
          time.
    Returns the declared value or ``None`` when absent + mode permits.
    """
    # codeql[py/path-injection] -- operator_yaml from viewer_operator_extras_source
    # (safe_resolve_directory + safe_fixed_file_under_root): Type 1.
    # Missing file is equivalent to "no extras declared". Subprocess mode
    # tolerates that (handled by the ``extras is None`` branch below);
    # Docker mode falls through to the same explicit error a few lines
    # down. Catching here keeps a brand-new corpus (``viewer_operator.yaml``
    # not yet written) from 500'ing on ``POST /api/jobs``.
    try:
        text = operator_yaml.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    extras = parse_pipeline_install_extras(text)
    if extras is not None and extras not in ("ml", "llm"):
        raise ValueError(
            "pipeline_install_extras must be 'ml' or 'llm' "
            f"(got {extras!r}) in operator YAML ({operator_yaml})."
        )
    if extras is None and pipe_mode == "docker":
        # Env-var fallback so first-run prod doesn't require operator
        # YAML editing — prod overlay sets PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS=llm.
        env_default = _env_default_pipeline_install_extras()
        if env_default is not None:
            return env_default
        raise ValueError(
            "Docker pipeline jobs require top-level pipeline_install_extras: ml "
            "or pipeline_install_extras: llm in the operator YAML "
            f"({operator_yaml}); set PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS "
            "on the api container env to provide a host-wide default."
        )
    return extras


async def _docker_jobs_factory(
    argv: Sequence[str],
    corpus_root: Path,
    log_abs: Path,
    *,
    operator_yaml: Path,
) -> asyncio.subprocess.Process:
    extras = assert_operator_pipeline_extras(operator_yaml)
    service, profile = _service_for_extras(extras)
    project = _project_dir()
    compose_files = _compose_files()
    tail = _cli_argv_tail(argv)
    if not shutil.which("docker"):
        raise RuntimeError("docker CLI not found in PATH (install docker-ce-cli in the API image).")

    cmd: list[str] = ["docker", "compose"]
    # Pass the project's ``.env`` so nested compose can resolve env vars
    # used in the YAML (e.g. ``PODCAST_CORPUS_HOST_PATH`` on a VPS where
    # the operator stages ``.env`` per PROD_RUNBOOK). Compose's default
    # ``.env`` auto-load looks in the project dir = ``dirname(first -f)``
    # = ``<project>/compose``, NOT ``<project>``, so without this flag
    # YAML refs like ``${VAR:?...}`` fail at parse time even when the
    # operator's ``.env`` has the value. Same root cause as the
    # direct-shell ``docker compose`` deploy gotcha. Guarded on file
    # existence so codespace + stack-test (no project-root ``.env``;
    # vars come from the api process env) keep their existing behaviour.
    env_file = project / ".env"
    if env_file.is_file():
        cmd.extend(["--env-file", str(env_file)])
    for f in compose_files:
        cmd.extend(["-f", str(_resolve_compose_path(project, f))])
    # Intentionally NOT passing ``--project-directory``. Compose v2 resolves
    # bind-mount sources and build contexts relative to the project
    # directory. When ``--project-directory`` is set explicitly to the repo
    # root, ``../config/ci/foo.yaml`` in ``compose/docker-compose.*.yml``
    # resolves to one level *above* the repo (silent host-side mkdir, then
    # the pipeline entrypoint fails ``[ ! -f /app/config.yaml ]`` because
    # the bind source is an empty directory). Letting Compose default the
    # project directory to ``dirname(first -f)`` = ``<repo>/compose`` makes
    # the existing relative paths (``../<...>``) resolve correctly to
    # ``<repo>/<...>``. ``COMPOSE_PROJECT_NAME`` (set in the API
    # ``environment:``) keeps the spawned container in the running stack's
    # project regardless.
    cmd.extend(
        [
            "--profile",
            profile,
            "run",
            "-T",
            "--rm",
            "--no-deps",
            service,
            "python",
            "-m",
            "podcast_scraper.cli",
            *tail,
        ]
    )

    log_abs.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_abs, "wb")
    logger.info("docker job spawn service=%s profile=%s cwd=%s", service, profile, project)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=log_f,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(project),
        start_new_session=os.name != "nt",
    )
    setattr(proc, "_ps_log_fp", log_f)
    return proc


def attach_docker_jobs_factory(app: Any) -> None:
    """Set ``app.state.jobs_subprocess_factory`` when Docker exec mode is enabled."""

    async def factory(
        argv: Sequence[str], corpus_root: Path, log_abs: Path
    ) -> asyncio.subprocess.Process:
        """Spawn pipeline via Compose ``pipeline`` / ``pipeline-llm`` (not ``sys.executable``)."""
        from podcast_scraper.server.operator_paths import viewer_operator_extras_source

        op = viewer_operator_extras_source(app, corpus_root)
        return await _docker_jobs_factory(argv, corpus_root, log_abs, operator_yaml=op)

    app.state.jobs_subprocess_factory = factory
