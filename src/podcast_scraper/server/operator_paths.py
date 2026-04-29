"""Resolve viewer operator YAML path (per-corpus vs ``serve --config-file`` override)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from podcast_scraper.utils.path_validation import (
    safe_fixed_file_under_root,
    safe_resolve_directory,
)

VIEWER_OPERATOR_BASENAME = "viewer_operator.yaml"


def viewer_operator_yaml_path(app: Any, corpus_root: Path) -> Path:
    """Return operator file path for ``corpus_root``.

    When ``serve`` was started with ``--config-file`` / ``PODCAST_SERVE_CONFIG_FILE``, all
    corpora share that single resolved file. Otherwise the file lives next to
    ``feeds.spec.yaml``: ``<corpus>/viewer_operator.yaml``.
    """
    fixed = getattr(app.state, "operator_config_fixed_path", None)
    if fixed is not None:
        return Path(fixed)
    return corpus_root / VIEWER_OPERATOR_BASENAME


def viewer_operator_extras_source(app: Any, corpus_root: Path) -> Path:
    """YAML file that may declare ``pipeline_install_extras`` for Docker job validation only.

    The pipeline CLI ``--config`` stays the pinned pipeline file (``viewer_operator_yaml_path``);
    Pydantic ``Config`` rejects ``pipeline_install_extras`` there. When Docker job mode is on and
    ``<corpus>/viewer_operator.yaml`` exists, read extras from that file; else fall back to the
    same path as ``viewer_operator_yaml_path`` (native / single-file operators).
    """
    if os.environ.get("PODCAST_PIPELINE_EXEC_MODE", "").strip().lower() != "docker":
        return viewer_operator_yaml_path(app, corpus_root)
    root = safe_resolve_directory(corpus_root)
    if root is None:
        return viewer_operator_yaml_path(app, corpus_root)
    # CodeQL py/path-injection: ``safe_fixed_file_under_root`` applies ``normpath_if_under_root``
    # immediately before this function's ``isfile`` sink (Type 1; CODEQL_DISMISSALS.md).
    candidate_s = safe_fixed_file_under_root(root, VIEWER_OPERATOR_BASENAME)
    if candidate_s is None:
        return viewer_operator_yaml_path(app, corpus_root)
    # codeql[py/path-injection] -- candidate_s from safe_fixed_file_under_root (Type 1).
    if os.path.isfile(candidate_s):
        return Path(candidate_s)
    return viewer_operator_yaml_path(app, corpus_root)


def packaged_viewer_operator_example_path() -> Path | None:
    """Path to ``config/examples/viewer_operator.example.yaml`` when present.

    Resolves in order:
      1. Repo / sdist tree relative to the package install (``…/<src>/../../config/examples``).
         Works for ``pip install -e .`` development checkouts and source dists where the
         file lives at the repo root alongside the package.
      2. Working-directory relative (``./config/examples/``). This is how the file is
         shipped in the Docker api image — the example dir is COPY'd into the WORKDIR
         alongside ``config/profiles/`` (see docker/api/Dockerfile). pip installs from
         a wheel don't include files outside ``src/podcast_scraper/`` so the package-
         relative path returns nothing in container builds.
      3. None — no auto-seed; GET returns empty content.
    """
    try:
        import podcast_scraper as _pkg
    except ImportError:
        return None
    rel = Path("config") / "examples" / "viewer_operator.example.yaml"
    # Path 1: ``<install>/<package>/.. /.. /config/examples/...`` (dev / sdist).
    here = Path(_pkg.__file__).resolve().parent
    repo_candidate = here.parents[1] / rel
    if repo_candidate.is_file():
        return repo_candidate
    # Path 2: cwd-relative (Docker api image where examples are COPY'd in).
    cwd_candidate = Path.cwd() / rel
    if cwd_candidate.is_file():
        return cwd_candidate
    return None
