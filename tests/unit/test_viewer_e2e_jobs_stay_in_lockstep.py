"""``viewer-e2e`` and ``nightly-viewer-e2e`` run the same suite; they must run it the same way.

#1619 moved the gi-kg-viewer Playwright suite off route-fulfilled mocks and onto a real Python
API. ``python-app.yml``'s ``viewer-e2e`` was updated for that — Python 3.11, a venv with
``[dev,search]``, ffmpeg, the MiniLM preload. Its nightly twin was not, and stayed node-only. It
went red on 2026-08-21 with::

    [WebServer] ModuleNotFoundError: No module named 'dotenv'

which reads like a missing dependency and is not one: ``python-dotenv`` is a core dependency
(pyproject), and ``playwright.config.ts`` falls back from ``<repo>/.venv/bin/python`` to a bare
``python3`` while still setting ``PYTHONPATH=<repo>/src``. So with no venv, ``podcast_scraper``
imports off the source tree with none of its dependencies installed, and the first ``import`` in
``config.py`` is what breaks. The environment was wrong; the error pointed at a library.

A "KEEP IN LOCKSTEP" comment is exactly the kind of instruction that rots — the drift it warns
about is silent, only shows up nightly, and looks like a code failure when it does. So assert it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _steps(workflow: str, job: str) -> list[dict]:
    data: dict = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    assert job in data["jobs"], f"{job} is gone from {workflow} — was it renamed?"
    steps: list[dict] = data["jobs"][job]["steps"]
    return steps


def test_the_two_viewer_e2e_jobs_have_identical_steps() -> None:
    """Same interpreter, same extras, same model cache, same command — or the twin is a fiction."""
    pr = _steps("python-app.yml", "viewer-e2e")
    nightly = _steps("nightly.yml", "nightly-viewer-e2e")
    assert nightly == pr, (
        "nightly-viewer-e2e has drifted from python-app viewer-e2e. They run the same Playwright "
        "suite against a real API, so a difference in setup means nightly is testing a different "
        "system — or, as in #1619's aftermath, no system at all."
    )


def test_both_provision_the_venv_the_playwright_config_looks_for() -> None:
    """The failure mode was a silent fallback, so pin the thing that made it silent."""
    for workflow, job in (("python-app.yml", "viewer-e2e"), ("nightly.yml", "nightly-viewer-e2e")):
        run_bodies = "\n".join(s.get("run", "") for s in _steps(workflow, job))
        assert "python -m venv .venv" in run_bodies, f"{job} never creates <repo>/.venv"
        assert (
            '.venv/bin/pip install -e ".[dev,search]"' in run_bodies
        ), f"{job} must install the extras the viewer's real API serves from"
