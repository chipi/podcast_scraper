"""Tests for ADR-097 self-hosted workflow allowlist checker."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_self_hosted_runner_allowlist.py"


def _load_checker_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_self_hosted_runner_allowlist", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runs_on_detects_multiline_self_hosted_list() -> None:
    mod = _load_checker_module()
    assert mod.runs_on_uses_self_hosted(["self-hosted", "dgx-spark"])
    assert not mod.runs_on_uses_self_hosted("ubuntu-latest")


def test_workflow_uses_self_hosted_multiline_runs_on(tmp_path: Path) -> None:
    mod = _load_checker_module()
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
    assert mod.workflow_uses_self_hosted(wf)


def test_main_flags_unlisted_self_hosted_workflow(tmp_path: Path, monkeypatch) -> None:
    mod = _load_checker_module()
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    rogue_yaml = (
        "name: rogue\non: push\njobs:\n  j:\n"
        "    runs-on: [self-hosted, dgx-spark]\n    steps: []\n"
    )
    (workflows / "rogue.yml").write_text(rogue_yaml, encoding="utf-8")
    allowlist = tmp_path / "SELF_HOSTED_RUNNER_ALLOWLIST.md"
    allowlist.write_text("# empty\n", encoding="utf-8")

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "WORKFLOWS", workflows)
    monkeypatch.setattr(mod, "ALLOWLIST", allowlist)

    assert mod.main() == 1


def test_main_passes_when_workflow_is_allowlisted(tmp_path: Path, monkeypatch) -> None:
    mod = _load_checker_module()
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "nightly.yml").write_text(
        "name: nightly\non: push\njobs:\n  j:\n"
        "    runs-on: [self-hosted, dgx-spark]\n    steps: []\n",
        encoding="utf-8",
    )
    allowlist = tmp_path / "SELF_HOSTED_RUNNER_ALLOWLIST.md"
    allowlist.write_text("nightly.yml\n", encoding="utf-8")

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "WORKFLOWS", workflows)
    monkeypatch.setattr(mod, "ALLOWLIST", allowlist)

    assert mod.main() == 0


def test_workflow_without_self_hosted_returns_false(tmp_path: Path) -> None:
    mod = _load_checker_module()
    wf = tmp_path / "ubuntu.yml"
    wf.write_text(
        "name: ci\non: push\njobs:\n  lint:\n    runs-on: ubuntu-latest\n    steps: []\n",
        encoding="utf-8",
    )
    assert mod.workflow_uses_self_hosted(wf) is False


def test_allowlist_passes_for_repo_workflows() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
