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


def _pytest_gates(workflow: str, job: str, step_marker: str) -> dict:
    """The pytest quality gates in a job's test step: {flag: value}.

    Read out of the shell body rather than a config file because that is where they live —
    every gate in this repo is a literal `--cov-fail-under=N` inside a workflow `run:` block, so
    a config-file assertion would check something the CI does not read.
    """
    import re

    for step in _steps(workflow, job):
        body = step.get("run", "") or ""
        if step_marker in body:
            return dict(re.findall(r"--(cov-fail-under)=([0-9.]+)", body))
    raise AssertionError(f"no step in {workflow}:{job} runs `{step_marker}`")


def test_the_e2e_coverage_floor_matches_across_workflows() -> None:
    """A floor recalibrated in one workflow and not the other fails nightly, silently, nightly.

    `b9fb37db` lowered the e2e floor 39 -> 38.5 in python-app.yml after the RFC-118 + viewer-perf
    batch added subsystems the e2e tier structurally never executes. nightly.yml kept 39, so from
    2026-08-25 nightly failed every night with `381 passed` and `Total coverage: 38.62%` — a green
    suite reported as a red build. `nightly-viewer-e2e` gates on that job, so it stopped running
    too and the viewer suite went unexercised for days without anyone being told.

    The two run the SAME selector over the SAME tests. If one floor is right the other is.
    """
    push = _pytest_gates("python-app.yml", "test-e2e", "tests/e2e/")
    nightly = _pytest_gates("nightly.yml", "nightly-test-e2e", "tests/e2e/")
    assert nightly == push, (
        "the e2e coverage floor has drifted between python-app and nightly. Recalibrating one "
        f"and not the other turns a passing suite into a nightly-only red: {push} vs {nightly}"
    )


def test_the_integration_coverage_floor_matches_across_workflows() -> None:
    """Same class, same trap — asserted before it bites rather than after."""
    push = _pytest_gates("python-app.yml", "test-integration", "tests/integration/")
    nightly = _pytest_gates("nightly.yml", "nightly-test-integration", "tests/integration/")
    assert nightly == push, f"integration coverage floor drifted: {push} vs {nightly}"


def test_the_makefile_e2e_threshold_matches_ci() -> None:
    """A third copy of the same number, and it had already drifted.

    `Makefile: COVERAGE_THRESHOLD_E2E` drives `make coverage-check-e2e`, the target a contributor
    runs locally to find out whether the gate will pass. It sat at 39 while CI had moved to 38.5
    — so the local check gated on a number CI had abandoned, and would fail a change CI accepts.
    A gate that disagrees with the gate teaches people to ignore it.
    """
    import re

    makefile = (_WORKFLOWS.parents[1] / "Makefile").read_text(encoding="utf-8")
    m = re.search(r"^COVERAGE_THRESHOLD_E2E\s*:=\s*([0-9.]+)", makefile, re.M)
    assert m, "COVERAGE_THRESHOLD_E2E is gone from the Makefile — was it renamed?"
    ci = _pytest_gates("python-app.yml", "test-e2e", "tests/e2e/")["cov-fail-under"]
    assert float(m.group(1)) == float(ci), (
        f"Makefile COVERAGE_THRESHOLD_E2E={m.group(1)} but CI gates at {ci} — "
        "the local check and the real gate disagree"
    )
