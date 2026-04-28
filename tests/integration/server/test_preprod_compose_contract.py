#!/usr/bin/env python3
"""Compose-config contract tests for the codespace pre-prod overlay (#693).

Asserts structural properties of ``stack.yml + prod.yml`` so the kind of
defects RFC-081 Phase 1 first shipped cannot regress silently:

* ``corpus_data`` volume is a host bind mount (not the Docker-managed default
  inherited from stack.yml). Without this, ``backup-corpus.yml`` tarballs an
  empty directory and the operator can't edit feeds.spec.yaml from the
  codespace shell.
* ``PODCAST_PIPELINE_EXEC_MODE=docker`` on api so the Docker job factory
  attaches at startup. Without this, ``POST /api/jobs`` would try to run
  pipeline code in-process inside the published api image (which only ships
  ``[server]`` extras) and crash on missing ``[llm]`` deps.
* api mounts ``/var/run/docker.sock`` and the project dir so the nested
  ``docker compose run --rm pipeline-llm`` works.
* ``PODCAST_DOCKER_COMPOSE_FILES`` references both overlays so the spawned
  pipeline container inherits the same prod-overlay ``corpus_data`` bind
  mount.
* ``PODCAST_AVAILABLE_PROFILES`` is set so the operator UI dropdown is
  filtered to published-image profiles only (#692, RFC-081 §Layer 1).
* Every ``${PODCAST_*}`` env var has a sensible default when unset (no
  surprise empty strings where a value is required).

Test discovers compose files relative to the repo root (parent of
``tests/``) so it works whether pytest is run from the repo root, the
package dir, or via ``make`` targets.

These assertions encode the structural contract — both Phase 1 first-ship
defects (corpus volume mismatch, exec mode unset) would have failed at
least one of these before reaching the operator. Regression-test the
test by reverting either ``corpus_data`` override or
``PODCAST_PIPELINE_EXEC_MODE`` and re-running this module: at least one
test must fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# ``critical_path`` so this runs in ``test-integration-fast`` on PRs. Without
# it, the compose-contract test would only run in main's full integration
# job — defeating the purpose (catching defects before merge to main).
pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


REPO_ROOT = Path(__file__).resolve().parents[3]
STACK_YML = REPO_ROOT / "compose" / "docker-compose.stack.yml"
PROD_YML = REPO_ROOT / "compose" / "docker-compose.prod.yml"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


pytestmark.append(
    pytest.mark.skipif(
        not _docker_available(),
        reason="docker CLI not on PATH; compose-config contract test requires docker",
    )
)


def _run_compose_config(extra_profile: str | None = None) -> Dict[str, Any]:
    env = {**os.environ, "PODCAST_DOCKER_PROJECT_DIR": "/workspaces/podcast_scraper"}
    cmd = [
        "docker",
        "compose",
        "-f",
        str(STACK_YML),
        "-f",
        str(PROD_YML),
    ]
    if extra_profile:
        cmd.extend(["--profile", extra_profile])
    cmd.extend(["config", "--format", "yaml"])
    proc = subprocess.run(  # noqa: S603 - cmd is hardcoded above
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"`docker compose config` exited {proc.returncode}.\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout[:2000]}"
        )
    parsed: Dict[str, Any] = yaml.safe_load(proc.stdout)
    return parsed


@pytest.fixture(scope="module")
def resolved_compose() -> Dict[str, Any]:
    """Resolved ``stack.yml + prod.yml`` config — default profile set.

    Module-scoped because the call is deterministic for a given repo state and
    invoking docker is the slow part (~1-2s on the first run, cached afterwards).
    Sets ``PODCAST_DOCKER_PROJECT_DIR`` to a stable codespace-shaped value so
    ``${PODCAST_DOCKER_PROJECT_DIR:-…}`` substitution is predictable in CI.
    Profile-gated services (``pipeline-llm``) are excluded from this fixture by
    design — they're spawned on-demand by the api job factory, not by
    ``compose up`` at boot. Use ``resolved_compose_with_pipeline`` for assertions
    that depend on the on-demand service definition.
    """
    return _run_compose_config()


@pytest.fixture(scope="module")
def resolved_compose_with_pipeline() -> Dict[str, Any]:
    """Same as ``resolved_compose`` but with ``--profile pipeline-llm`` so the
    on-demand pipeline service appears in the resolved config (lets us assert
    on its ``image:`` ref without spawning the container)."""
    return _run_compose_config(extra_profile="pipeline-llm")


def _api_service(resolved: Dict[str, Any]) -> Dict[str, Any]:
    services = resolved.get("services") or {}
    api = services.get("api")
    assert api is not None, "resolved compose config has no ``api`` service"
    out: Dict[str, Any] = api
    return out


# ---------------------------------------------------------------------------
# corpus_data volume topology — host bind mount, not Docker-managed default
# ---------------------------------------------------------------------------


def test_corpus_data_volume_is_local_bind_mount(resolved_compose: Dict[str, Any]) -> None:
    """``corpus_data`` is a local-driver bind mount, not Docker-managed.

    Failure mode this catches: stack.yml's default ``corpus_data: {}`` carrying
    over into prod, leaving the corpus inside ``/var/lib/docker/volumes/...``
    where backup-corpus.yml can't reach it and the operator can't edit
    feeds.spec.yaml from the codespace shell.
    """
    volumes = resolved_compose.get("volumes") or {}
    corpus = volumes.get("corpus_data")
    assert corpus is not None, "``corpus_data`` volume missing from resolved config"
    assert (
        corpus.get("driver") == "local"
    ), f"``corpus_data`` driver should be ``local`` for bind mount, got {corpus.get('driver')!r}"
    opts = corpus.get("driver_opts") or {}
    assert (
        opts.get("type") == "none"
    ), "``corpus_data`` driver_opts.type should be ``none`` for bind"
    assert opts.get("o") == "bind", "``corpus_data`` driver_opts.o should be ``bind``"
    device = opts.get("device") or ""
    assert device, "``corpus_data`` driver_opts.device must be set (host bind path)"
    # Codespace default; VPS deploys override via PODCAST_CORPUS_HOST_PATH.
    # When unset, we expect the codespace path so backup-corpus.yml's
    # tarball-from-host-path workflow lines up.
    assert device.endswith(".codespace_corpus") or device.startswith(
        "/"
    ), f"``corpus_data`` device should resolve to an absolute path, got {device!r}"


# ---------------------------------------------------------------------------
# api Docker job-mode wiring (PODCAST_PIPELINE_EXEC_MODE + sock + project dir)
# ---------------------------------------------------------------------------


def test_api_pipeline_exec_mode_is_docker(resolved_compose: Dict[str, Any]) -> None:
    """``PODCAST_PIPELINE_EXEC_MODE=docker`` so the api attaches Docker job factory.

    Failure mode this catches: api silently falls back to in-process subprocess
    mode and crashes on first pipeline run because ``[llm]`` SDKs aren't in
    the published api image.
    """
    env = _api_service(resolved_compose).get("environment") or {}
    assert env.get("PODCAST_PIPELINE_EXEC_MODE") == "docker", (
        "api must run in Docker exec mode in the prod overlay; got "
        f"{env.get('PODCAST_PIPELINE_EXEC_MODE')!r}"
    )


def test_api_mounts_docker_socket(resolved_compose: Dict[str, Any]) -> None:
    """api volumes include ``/var/run/docker.sock:/var/run/docker.sock``."""
    volumes = _api_service(resolved_compose).get("volumes") or []
    has_sock = any(
        (
            isinstance(v, dict)
            and v.get("source") == "/var/run/docker.sock"
            and v.get("target") == "/var/run/docker.sock"
        )
        or (isinstance(v, str) and v.startswith("/var/run/docker.sock:/var/run/docker.sock"))
        for v in volumes
    )
    assert has_sock, (
        "api must mount /var/run/docker.sock for the nested ``docker compose run`` "
        f"to reach the host daemon. Got volumes:\n{json.dumps(volumes, indent=2)}"
    )


def test_api_mounts_project_dir_readonly(resolved_compose: Dict[str, Any]) -> None:
    """api bind-mounts the project dir so ``compose/*.yml`` is reachable from inside.

    Pinned to ``PODCAST_DOCKER_PROJECT_DIR`` from the fixture (codespace-shaped).
    """
    volumes = _api_service(resolved_compose).get("volumes") or []
    has_project = False
    for v in volumes:
        if isinstance(v, dict):
            src = str(v.get("source") or "")
            if src == "/workspaces/podcast_scraper" and v.get("read_only", False):
                has_project = True
                break
        elif isinstance(v, str) and "/workspaces/podcast_scraper" in v and v.endswith(":ro"):
            has_project = True
            break
    assert has_project, (
        "api must bind-mount the project dir read-only so the nested compose "
        f"call can find compose/*.yml. Got volumes:\n{json.dumps(volumes, indent=2)}"
    )


def test_api_compose_files_env_references_both_overlays(
    resolved_compose: Dict[str, Any],
) -> None:
    """``PODCAST_DOCKER_COMPOSE_FILES`` references both stack + prod overlays.

    Without prod in the list, the spawned pipeline container would resolve a
    Docker-managed corpus_data again (stack.yml's default) — defeating the
    bind-mount fix.
    """
    env = _api_service(resolved_compose).get("environment") or {}
    files = env.get("PODCAST_DOCKER_COMPOSE_FILES") or ""
    assert (
        "docker-compose.stack.yml" in files
    ), f"PODCAST_DOCKER_COMPOSE_FILES missing stack.yml: {files!r}"
    assert (
        "docker-compose.prod.yml" in files
    ), f"PODCAST_DOCKER_COMPOSE_FILES missing prod.yml: {files!r}"


def test_api_compose_project_name_pinned(resolved_compose: Dict[str, Any]) -> None:
    """``COMPOSE_PROJECT_NAME`` is pinned so nested ``docker compose run`` joins
    the running stack's network + volumes (not a fresh project)."""
    env = _api_service(resolved_compose).get("environment") or {}
    project = env.get("COMPOSE_PROJECT_NAME")
    assert project, "COMPOSE_PROJECT_NAME must be set on api in prod overlay"


# ---------------------------------------------------------------------------
# Profile filtering (#692 / RFC-081 §Layer 1)
# ---------------------------------------------------------------------------


def test_api_available_profiles_env_set(resolved_compose: Dict[str, Any]) -> None:
    """``PODCAST_AVAILABLE_PROFILES`` filters the operator dropdown.

    Phase 1 only publishes ``pipeline-llm`` to GHCR; without this var the
    operator UI offers profiles whose backing image isn't deployed.
    """
    env = _api_service(resolved_compose).get("environment") or {}
    profiles = (env.get("PODCAST_AVAILABLE_PROFILES") or "").strip()
    assert profiles, "PODCAST_AVAILABLE_PROFILES must be set in the prod overlay"
    # Default value uses cloud_thin (license-clean published image set);
    # operators can override the env without code change.
    names = [p.strip() for p in profiles.split(",") if p.strip()]
    assert (
        "cloud_thin" in names
    ), f"PODCAST_AVAILABLE_PROFILES default should include cloud_thin; got {names}"


def test_api_default_profile_env_set_and_in_allowlist(
    resolved_compose: Dict[str, Any],
) -> None:
    """``PODCAST_DEFAULT_PROFILE`` is set + must be inside ``PODCAST_AVAILABLE_PROFILES``.

    The viewer preselects ``default_profile`` in the dropdown; if it's outside
    the allowlist, the operator sees a value they can't actually pick. Server
    code already guards this (``env_default_profile`` returns None when the
    requested default isn't allowed), but encoding the contract here catches
    the misconfig at compose-time before it reaches a user.
    """
    env = _api_service(resolved_compose).get("environment") or {}
    default = (env.get("PODCAST_DEFAULT_PROFILE") or "").strip()
    assert default, "PODCAST_DEFAULT_PROFILE must be set in the prod overlay"
    allowlist = [
        p.strip() for p in (env.get("PODCAST_AVAILABLE_PROFILES") or "").split(",") if p.strip()
    ]
    assert default in allowlist, (
        f"PODCAST_DEFAULT_PROFILE={default!r} must be inside "
        f"PODCAST_AVAILABLE_PROFILES={allowlist}; viewer would otherwise preselect "
        "a profile the operator can't pick."
    )


# ---------------------------------------------------------------------------
# Sane defaults — no surprise empty strings where a value is required
# ---------------------------------------------------------------------------


def test_api_required_env_has_sensible_defaults(resolved_compose: Dict[str, Any]) -> None:
    """Variables that must have a value when unset should not resolve to ``""``.

    The ``${VAR:-default}`` form must always give a non-empty value for any
    var that the api treats as required. Catches the "forgot to set a
    default" class of bug where compose silently substitutes empty string.
    """
    env = _api_service(resolved_compose).get("environment") or {}
    must_be_nonempty = (
        "PODCAST_PIPELINE_EXEC_MODE",
        "PODCAST_DOCKER_PROJECT_DIR",
        "PODCAST_DOCKER_COMPOSE_FILES",
        "COMPOSE_PROJECT_NAME",
        "PODCAST_AVAILABLE_PROFILES",
        "PODCAST_DEFAULT_PROFILE",
    )
    for k in must_be_nonempty:
        v = (env.get(k) or "").strip() if isinstance(env.get(k), str) else env.get(k)
        assert v, f"api env var {k} resolved to empty/missing — needs a default"


def test_api_image_pulled_from_ghcr(resolved_compose: Dict[str, Any]) -> None:
    """api uses the GHCR-published image, not the local-build tag from stack.yml."""
    image = _api_service(resolved_compose).get("image") or ""
    assert image.startswith(
        "ghcr.io/chipi/podcast-scraper-stack-api:"
    ), f"prod overlay must point api at GHCR-published image; got {image!r}"


def test_pipeline_llm_image_pulled_from_ghcr(
    resolved_compose_with_pipeline: Dict[str, Any],
) -> None:
    """pipeline-llm uses the GHCR-published image (the only published pipeline variant).

    Uses the with-pipeline fixture because pipeline-llm is profile-gated in
    stack.yml — only the api factory spawns it via ``--profile pipeline-llm``.
    """
    pipeline = (resolved_compose_with_pipeline.get("services") or {}).get("pipeline-llm") or {}
    image = pipeline.get("image") or ""
    assert image.startswith(
        "ghcr.io/chipi/podcast-scraper-stack-pipeline-llm:"
    ), f"prod overlay must point pipeline-llm at GHCR-published image; got {image!r}"
