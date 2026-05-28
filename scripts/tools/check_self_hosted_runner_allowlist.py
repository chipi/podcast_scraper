#!/usr/bin/env python3
"""Enforce ADR-097: self-hosted workflows must be allow-listed (RFC-089 P3)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = REPO_ROOT / ".github" / "SELF_HOSTED_RUNNER_ALLOWLIST.md"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

SELF_HOSTED_RE = re.compile(
    r"^\s*runs-on:\s*(.+)$",
    re.MULTILINE,
)
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


def _workflow_uses_self_hosted(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    for match in SELF_HOSTED_RE.finditer(text):
        value = match.group(1)
        if SELF_HOSTED_VALUE_RE.search(value):
            return True
    return False


def main() -> int:
    allowed = _load_allowlist()
    violations: list[str] = []
    for wf in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        if not _workflow_uses_self_hosted(wf):
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
