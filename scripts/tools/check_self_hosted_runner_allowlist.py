#!/usr/bin/env python3
"""Enforce ADR-097: self-hosted workflows must be allow-listed (RFC-089 P3)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = REPO_ROOT / ".github" / "SELF_HOSTED_RUNNER_ALLOWLIST.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

SELF_HOSTED_VALUE_RE = re.compile(r"self-hosted|dgx-spark")


def _load_allowlist() -> set[str]:
    if not ALLOWLIST.is_file():
        raise SystemExit(f"Missing allowlist: {ALLOWLIST}")
    names: set[str] = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.add(stripped)
    return names


def runs_on_uses_self_hosted(runs_on: Any) -> bool:
    """Return True when a workflow job runs-on references self-hosted or dgx-spark."""
    if runs_on is None:
        return False
    if isinstance(runs_on, str):
        return bool(SELF_HOSTED_VALUE_RE.search(runs_on))
    if isinstance(runs_on, list):
        return any(runs_on_uses_self_hosted(item) for item in runs_on)
    return bool(SELF_HOSTED_VALUE_RE.search(str(runs_on)))


def workflow_uses_self_hosted(path: Path) -> bool:
    """Parse workflow YAML and detect self-hosted runner labels on any job."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        return False
    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        return False
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        if runs_on_uses_self_hosted(job.get("runs-on")):
            return True
    return False


def main() -> int:
    allowed = _load_allowlist()
    violations: list[str] = []
    for wf in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        if not workflow_uses_self_hosted(wf):
            continue
        if wf.name not in allowed:
            violations.append(wf.name)
    if violations:
        print(
            "Self-hosted runner policy violation (ADR-097). "
            "Add workflow to .github/SELF_HOSTED_RUNNER_ALLOWLIST.md or remove "
            "runs-on self-hosted / dgx-spark:",
            file=sys.stderr,
        )
        for name in violations:
            print(f"  - {name}", file=sys.stderr)
        return 1
    print(f"OK: {len(allowed)} allow-listed workflow(s); no unlisted self-hosted usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
