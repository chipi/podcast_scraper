"""Tests for ADR-097 self-hosted workflow allowlist checker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_self_hosted_runner_allowlist.py"

from scripts.tools.check_self_hosted_runner_allowlist import (  # noqa: E402
    runs_on_uses_self_hosted,
    workflow_uses_self_hosted,
)


def test_runs_on_detects_multiline_self_hosted_list() -> None:
    assert runs_on_uses_self_hosted(["self-hosted", "dgx-spark"])
    assert not runs_on_uses_self_hosted("ubuntu-latest")


def test_workflow_uses_self_hosted_multiline_runs_on(tmp_path: Path) -> None:
    wf = tmp_path / "evil.yml"
    wf.write_text(
        """
name: evil
on: push
jobs:
  nightly:
    runs-on:
      - self-hosted
      - dgx-spark
    steps:
      - run: echo hi
""".strip() + "\n",
        encoding="utf-8",
    )
    assert workflow_uses_self_hosted(wf)


def test_main_flags_unlisted_self_hosted_workflow(tmp_path: Path, monkeypatch) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    rogue_yaml = (
        "name: rogue\non: push\njobs:\n  j:\n"
        "    runs-on: [self-hosted, dgx-spark]\n    steps: []\n"
    )
    (workflows / "rogue.yml").write_text(rogue_yaml, encoding="utf-8")
    allowlist = tmp_path / "SELF_HOSTED_RUNNER_ALLOWLIST.md"
    allowlist.write_text("# empty\n", encoding="utf-8")

    import scripts.tools.check_self_hosted_runner_allowlist as mod

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "WORKFLOWS", workflows)
    monkeypatch.setattr(mod, "ALLOWLIST", allowlist)

    assert mod.main() == 1


def test_allowlist_passes_for_repo_workflows() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
