"""P3 — the prod-secret-staging gate must be PROXIMITY-aware, not existence-aware.

The D5 false-green (2026-08-23): deploy-prod staged the tmpfs secrets once, ran `compose up`
(create #1), then ran the D5 gateway probe (create #2) many steps later without re-staging.
The RAM dir had been reaped by systemd RemoveIPC when the earlier ssh session ended, so create
#2 got a 401 that looked like a bad key. The old gate saw a stage AND a create and passed.

These pin the step-walk proximity model in ``_proximity_problems``: every cross-step container
creation needs a (re)stage since the last one. (#1811 P3)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "check_prod_secret_staging_under_test",
    ROOT / "scripts" / "tools" / "check_prod_secret_staging.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]

_STAGE = {"name": "Stage", "uses": "./.github/actions/stage-prod-secrets"}


_RUN = "ssh deploy@x 'docker compose run --rm -T pipeline-llm ...'"


def _create(name: str) -> dict:
    return {"name": name, "run": _RUN}


def _job(steps: list) -> dict:
    return {"jobs": {"deploy": {"steps": steps}}}


def test_stage_once_create_twice_is_flagged() -> None:
    """The exact D5 shape: one stage, `compose up`, then a second create with no re-stage."""
    doc = _job(
        [
            _STAGE,
            {"name": "Deploy", "run": "ssh deploy@x 'docker compose up -d'"},
            {"name": "D5 probe", "run": _RUN},
        ]
    )
    problems = _mod._proximity_problems("deploy-prod.yml", doc)
    assert len(problems) == 1, problems
    assert "D5 probe" in problems[0]


def test_re_stage_before_second_create_passes() -> None:
    """The fix: a re-stage step immediately before the second create clears it."""
    doc = _job(
        [
            _STAGE,
            {"name": "Deploy", "run": "ssh deploy@x 'docker compose up -d'"},
            {"name": "Re-stage before D5", "uses": "./.github/actions/stage-prod-secrets"},
            {"name": "D5 probe", "run": _RUN},
        ]
    )
    assert _mod._proximity_problems("deploy-prod.yml", doc) == []


def test_single_stage_single_create_passes() -> None:
    """The common one-container workflow (reprocess/gi-repair/reenrich) stays green."""
    doc = _job([_STAGE, _create("Reprocess")])
    assert _mod._proximity_problems("reprocess-prod.yml", doc) == []


def test_create_with_no_stage_at_all_is_flagged() -> None:
    doc = _job([_create("Run something")])
    problems = _mod._proximity_problems("bad.yml", doc)
    assert len(problems) == 1
    assert "no stage step precedes it" in problems[0]


def test_shell_comment_mentioning_compose_run_is_not_a_create() -> None:
    """A step whose run script only *comments* about a nested `docker compose run` must not be
    read as a container creation (the false positive that flagged deploy-prod's .env step)."""
    doc = _job(
        [
            {
                "name": "Stage .env",
                "run": (
                    "ssh deploy@x '\n"
                    "# the nested docker compose run pipeline-llm reads these\n"
                    "echo writing .env'"
                ),
            },
            _STAGE,
            _create("Reprocess"),
        ]
    )
    assert _mod._proximity_problems("deploy-prod.yml", doc) == []


def test_real_workflows_pass_the_full_gate() -> None:
    """Regression: the live prod workflows must all pass (they carry the correct staging)."""
    assert _mod.main() == 0
