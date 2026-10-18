#!/usr/bin/env python3
"""Refuse writing content code before its change record has the method's pre-build steps.

Two modes, one rule (``podcast_scraper.quality.change_gate.check_edit``):

- Claude Code ``PreToolUse`` (default): reads the tool call on stdin. Edit/Write/MultiEdit/
  NotebookEdit/ctx_patch name their target. For a shell command, a guarded path is a target when it
  is a redirect's target or the command rewrites what it names (``sed -i``, ``tee``, ``cp``, ``mv``,
  ``git mv``, ``patch``, a formatter); a script that edits files names no target and is caught at
  the commit. A refusal is a ``deny`` with the reasons, so the agent sees what is missing.
- ``--staged``: the git pre-commit hook. Checks the staged files, whatever wrote them (a script run
  from elsewhere names no guarded path in its command line, so the commit is where it is caught).

Stdlib only. A call that writes no guarded file is never touched. A gate that errors on a call
that does write one DENIES it (fails closed): a guard that silently stops guarding is the failure
this exists to prevent.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _gate(root: Path):
    """``change_gate`` loaded from ``root``'s own file: importing the package would pull in its
    whole ``__init__`` (yaml and the rest), which the system Python running this hook lacks."""
    path = root / "src" / "podcast_scraper" / "quality" / "change_gate.py"
    name = f"change_gate_{abs(hash(str(root)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


_PY_PATH = r"(?:[\w./-]*/)?src/podcast_scraper/[\w./*-]+\.py"
#: Commands that rewrite the files they name: every guarded path in such a command is a target.
_IN_PLACE = re.compile(
    r"\bsed\s+(?:-[a-zA-Z]*\s+)*-i|\bperl\s+-[a-z]*i|\bblack\b(?!.*--check)|"
    r"\bisort\b(?!.*--check)|\bpatch\b|\bgit\s+mv\b|\bmv\b|\bcp\b|\btee\b|\btruncate\b|\brm\b"
)
#: A redirect's target (``> path`` / ``>> path``), not a file descriptor duplication like 2>&1.
_REDIRECT = re.compile(r"(?<![0-9&])>>?\s*(" + _PY_PATH + ")")


def _root_and_rel(path: str, cwd: str) -> tuple:
    """``(checkout root, repo-relative path)`` for a target, or ``(None, "")`` outside any checkout.

    The root is the nearest parent holding the gate config, so a session whose hook comes from one
    checkout still judges an edit in another worktree by THAT worktree's config and records.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd or REPO) / p
    p = p.resolve()
    for parent in p.parents:
        if (parent / "config" / "corpus_change_gate.json").is_file():
            return parent, p.relative_to(parent).as_posix()
    return None, ""


def _targets(call: dict) -> list:
    """``[(root, rel)]`` for every guarded-looking file this tool call would write."""
    tool = str(call.get("tool_name") or "")
    args = call.get("tool_input") or {}
    cwd = str(call.get("cwd") or "")
    paths: list = []
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit") or tool.endswith("ctx_patch"):
        paths = [str(args.get(k)) for k in ("file_path", "notebook_path", "path") if args.get(k)]
    elif tool.lower() == "bash" or tool.endswith("ctx_shell"):
        cmd = str(args.get("command") or "")
        paths = [m.group(1) for m in _REDIRECT.finditer(cmd)]
        if _IN_PLACE.search(cmd):
            paths += re.findall(_PY_PATH, cmd)
    out = [_root_and_rel(x, cwd) for x in paths]
    return [(root, rel) for root, rel in out if root is not None and _guarded(root, rel)]


def _guarded(root: Path, rel: str) -> bool:
    """Is ``rel`` content code in ``root``'s config? Read here, from the plain JSON, so a gate
    module that fails to load still knows which files it would have guarded."""
    patterns = json.loads((root / "config" / "corpus_change_gate.json").read_text())["content_code"]
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Corpus change gate "
                    "(docs/guides/CORPUS_CHANGE_METHOD.md): " + reason,
                }
            }
        )
    )


def main() -> int:
    if "--staged" in sys.argv:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        problems = _gate(REPO).check_edit(REPO, out)
        for p in problems:
            print(f"corpus change gate: {p}", file=sys.stderr)
        return 1 if problems else 0
    try:
        call = json.load(sys.stdin)
    except ValueError:
        return 0
    targets = _targets(call)
    if not targets:
        return 0
    by_root: dict = {}
    for root, rel in targets:
        by_root.setdefault(root, []).append(rel)
    try:
        problems = [p for root, rels in by_root.items() for p in _gate(root).check_edit(root, rels)]
    except Exception as exc:  # noqa: BLE001 - a broken gate fails CLOSED on the files it guards
        _deny(
            f"the gate itself failed ({type(exc).__name__}: {exc}); fix it before writing {targets}"
        )
        return 0
    if problems:
        _deny(" | ".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
