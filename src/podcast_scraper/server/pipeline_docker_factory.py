"""Docker Compose pipeline job factory (#660 / RFC-079 Phase 2).

When ``PODCAST_PIPELINE_EXEC_MODE=docker``, ``create_app`` attaches
``app.state.jobs_subprocess_factory`` so ``POST /api/jobs`` runs the CLI inside the
``pipeline`` or ``pipeline-llm`` service instead of ``sys.executable`` in the API container.

Requires:
  - Docker CLI + Compose v2 plugin in the API image (see ``docker/api/Dockerfile``).
  - Host Docker socket mounted into the API container (e.g. ``/var/run/docker.sock``).
  - ``PODCAST_DOCKER_PROJECT_DIR``: absolute path to the repository root on the **host** (same
    path visible to the Docker daemon), typically bind-mounted into the API container.
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
    raw = os.environ.get("PODCAST_DOCKER_COMPOSE_FILES", "compose/docker-compose.stack.yml").strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


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


def assert_operator_pipeline_extras(operator_yaml: Path) -> str:
    """Ensure operator YAML declares ``pipeline_install_extras: ml`` or ``llm``; return value."""
    text = operator_yaml.read_text(encoding="utf-8", errors="replace")
    extras = parse_pipeline_install_extras(text)
    if extras not in ("ml", "llm"):
        raise ValueError(
            "Docker pipeline jobs require top-level pipeline_install_extras: ml "
            "or pipeline_install_extras: llm in the operator YAML "
            f"({operator_yaml})."
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
    for f in compose_files:
        cmd.extend(["-f", str(project / f)])
    cmd.extend(
        [
            "--project-directory",
            str(project),
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
        from podcast_scraper.server.operator_paths import viewer_operator_yaml_path

        op = viewer_operator_yaml_path(app, corpus_root)
        return await _docker_jobs_factory(argv, corpus_root, log_abs, operator_yaml=op)

    app.state.jobs_subprocess_factory = factory
