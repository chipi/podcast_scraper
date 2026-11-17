"""One Python minor across the repo: ``.python-version`` is the source.

The images run ``python:3.12``; CI tested 3.11.8 until chipi/infra #82, so neither tested what prod
runs. Every workflow reads ``.python-version`` through ``python-version-file``, every python image
uses its minor, and the package floor, black's target and mypy's target are that minor too.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]


def _minor() -> str:
    raw = (_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", raw)
    assert match, f".python-version holds {raw!r}, not a version"
    return f"{match.group(1)}.{match.group(2)}"


def _workflow_files() -> list[Path]:
    github = _ROOT / ".github"
    return sorted(p for p in github.rglob("*") if p.suffix in (".yml", ".yaml") and p.is_file())


def test_workflows_read_the_version_file() -> None:
    literal = re.compile(r"^\s*python-version:", re.MULTILINE)
    offenders = [
        str(p.relative_to(_ROOT)) for p in _workflow_files() if literal.search(p.read_text("utf-8"))
    ]
    assert (
        not offenders
    ), f"use python-version-file: .python-version instead of a literal in {offenders}"


def test_every_python_image_is_on_the_same_minor() -> None:
    minor = _minor()
    from_python = re.compile(r"^FROM\s+python:(\d+\.\d+)", re.MULTILINE | re.IGNORECASE)
    seen = []
    for dockerfile in sorted((_ROOT / "docker").rglob("Dockerfile*")):
        for image_minor in from_python.findall(dockerfile.read_text(encoding="utf-8")):
            seen.append((str(dockerfile.relative_to(_ROOT)), image_minor))
    assert seen, "found no FROM python:X.Y in docker/: the guard would check nothing"
    wrong = [entry for entry in seen if entry[1] != minor]
    assert not wrong, f".python-version is {minor}; these images differ: {wrong}"


def test_pyproject_targets_the_same_minor() -> None:
    minor = _minor()
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == f">={minor}"
    assert pyproject["tool"]["black"]["target-version"] == [f"py{minor.replace('.', '')}"]
    assert pyproject["tool"]["mypy"]["python_version"] == minor
