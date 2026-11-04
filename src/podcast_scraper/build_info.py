"""What build is running: the image's ``VERSION.json`` and the release it was deployed as.

The image build writes ``/app/VERSION.json`` (``component``, ``version``, ``git_sha``,
``built_at``). A source checkout has no such file, so ``git_sha`` falls back to the
``PODCAST_GIT_SHA`` environment variable and then to ``None``.

Two release identifiers exist and are kept apart: ``CL_RELEASE`` is the infra release id of the
whole deploy; ``PODCAST_RELEASE`` is the public image ref the container was started from.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VERSION_FILE = "PODCAST_VERSION_FILE"
DEFAULT_VERSION_FILE = "/app/VERSION.json"
ENV_MODULES_DIR = "PODCAST_MODULES_DIR"
DEFAULT_MODULES_DIR = "/app/modules"
ENV_RELEASE = "CL_RELEASE"
ENV_IMAGE_RELEASE = "PODCAST_RELEASE"
ENV_GIT_SHA = "PODCAST_GIT_SHA"


def read_version_file(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """The JSON object in a ``VERSION.json``, or ``{}`` when it is absent or unreadable."""
    target = Path(path or os.environ.get(ENV_VERSION_FILE) or DEFAULT_VERSION_FILE)
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("unreadable %s: %s", target, exc)
        return {}
    return doc if isinstance(doc, dict) else {}


def build_info() -> dict[str, Any]:
    """``{component, version, git_sha, built_at}`` of this image; ``None`` where unknown."""
    from podcast_scraper import __version__

    doc = read_version_file()
    return {
        "component": doc.get("component") or None,
        "version": doc.get("version") or __version__,
        "git_sha": doc.get("git_sha") or (os.environ.get(ENV_GIT_SHA) or "").strip() or None,
        "built_at": doc.get("built_at") or None,
    }


def release() -> tuple[str | None, str | None]:
    """``(release, source)``: ``CL_RELEASE``, else ``PODCAST_RELEASE`` (named as the source)."""
    for name in (ENV_RELEASE, ENV_IMAGE_RELEASE):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value, name
    return None, None


def deploy_release() -> str | None:
    """The infra release id (``CL_RELEASE``), or ``None``."""
    return (os.environ.get(ENV_RELEASE) or "").strip() or None


def image_release() -> str | None:
    """The public image ref (``PODCAST_RELEASE``), or ``None``."""
    return (os.environ.get(ENV_IMAGE_RELEASE) or "").strip() or None


def module_version_file(
    name: str, directory: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """An installed module's own build record, ``<modules dir>/<name>.json`` (``/app/modules``).

    A package's image writes one at build time; its ``Extension.version_info`` reads it.
    """
    base = Path(directory or os.environ.get(ENV_MODULES_DIR) or DEFAULT_MODULES_DIR)
    return read_version_file(base / f"{name}.json")


__all__ = [
    "build_info",
    "deploy_release",
    "image_release",
    "module_version_file",
    "read_version_file",
    "release",
]
