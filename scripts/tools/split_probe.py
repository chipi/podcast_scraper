#!/usr/bin/env python3
"""Measure how far the ADR-162 split is from clean: the seams, in both directions.

Run after ``split_copy.py`` has filled ``apps/``. In a throwaway worktree of HEAD it deletes
everything the manifest moves, then:

1. **public without private** — imports every remaining ``podcast_scraper`` module and counts
   the failures, grouped by the missing module that caused them;
2. **edges** — lists every import (and dotted string) in the remaining ``src/``, ``tests/`` and
   ``scripts/`` that still names moved code, with file:line;
3. **private on top of public** — imports every module of the private packages from ``apps/``
   with the pruned public tree first on the path.

Nothing is installed and the working tree is never touched; the worktree is removed at the end.
The progress measure for the seams is that 1 and 3 reach zero failures and 2 lists nothing.

Usage::

    python scripts/tools/split_copy.py
    python scripts/tools/split_probe.py            # summary
    python scripts/tools/split_probe.py --edges    # plus every edge with file:line
    python scripts/tools/split_probe.py --fails    # plus every failing module and why
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
import split_copy as sc  # noqa: E402

_IMPORT_ALL = """
import importlib, json, pkgutil, sys
fails = {}
count = 0
for top in sys.argv[1:]:
    pkg = importlib.import_module(top)
    for mi in pkgutil.walk_packages(pkg.__path__, top + "."):
        count += 1
        try:
            importlib.import_module(mi.name)
        except Exception as e:  # noqa: BLE001 - every failure is a measurement
            fails[mi.name] = f"{type(e).__name__}: {e}"
print(json.dumps({"modules": count, "fails": fails}))
"""

_ROOT_CAUSE = re.compile(r"(?:No module named|cannot import name) '([^']+)'")


def prune(worktree: Path, manifest: dict, mapping: dict[str, str]) -> int:
    """Delete from *worktree* everything the manifest moves; return how many paths went."""
    gone = 0
    for f in sc.tracked_files():
        if f.startswith("src/") and f.endswith(".py") and sc.module_of(f) in mapping:
            (worktree / f).unlink(missing_ok=True)
            gone += 1
    for top in manifest.values():
        # forked_trees are not pruned: the public repo keeps its own copy.
        for tree in top.get("trees", {}):
            shutil.rmtree(worktree / tree, ignore_errors=True)
            gone += 1
        for _pkg, spec in sc.units(top):
            for pkg in spec.get("packages", {}):
                shutil.rmtree(worktree / "src" / pkg.replace(".", "/"), ignore_errors=True)
            for f in spec.get("files", []):
                if (worktree / f).exists():
                    (worktree / f).unlink()
                    gone += 1
            for tree in spec.get("file_trees", []):
                if (worktree / tree).exists():
                    shutil.rmtree(worktree / tree)
                    gone += 1
    for repo in manifest:
        for p in (sc.APPS / repo / "tests").rglob("*.py"):
            rel = p.relative_to(sc.APPS / repo)
            if p.name != "conftest.py" and (worktree / rel).exists():
                (worktree / rel).unlink()
                gone += 1
    # An emptied directory would still import as a namespace package and hide a missed caller.
    for d in sorted((worktree / "src").rglob("*"), key=lambda x: -len(x.parts)):
        if d.is_dir() and not [c for c in d.iterdir() if c.name != "__pycache__"]:
            shutil.rmtree(d)
    return gone


def import_all(pythonpath: list[Path], packages: list[str], cwd: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath)}
    out = subprocess.run(
        [sys.executable, "-c", _IMPORT_ALL, *packages],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict = json.loads(out.strip().splitlines()[-1])
    return result


def edges(worktree: Path, moved: set[str]) -> list[tuple[str, int, str]]:
    """Every import or dotted string in the pruned tree that names moved code."""
    names = sorted(moved, key=len, reverse=True)
    rx = re.compile(r"(?<![\w.])(" + "|".join(re.escape(n) for n in names) + r")(?![\w])")
    found: list[tuple[str, int, str]] = []
    for area in ("src", "tests", "scripts"):
        for p in (worktree / area).rglob("*.py"):
            rel = p.relative_to(worktree).as_posix()
            try:
                tree = ast.parse(p.read_text(errors="ignore"))
            except SyntaxError:
                continue
            module = sc.module_of(rel) if rel.startswith("src/") else None
            is_pkg = p.name == "__init__.py"
            docstrings = {
                id(n.body[0].value)
                for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and n.body
                and isinstance(n.body[0], ast.Expr)
                and isinstance(n.body[0].value, ast.Constant)
            }
            for n in ast.walk(tree):
                targets: list[str] = []
                if isinstance(n, ast.ImportFrom):
                    if n.level and module:
                        base = sc.absolute_from(module, is_pkg, n)
                    else:
                        base = n.module or ""
                    if base in moved:
                        targets.append(base)
                    targets += [f"{base}.{a.name}" for a in n.names if f"{base}.{a.name}" in moved]
                elif isinstance(n, ast.Import):
                    targets += [a.name for a in n.names if a.name in moved]
                elif (
                    isinstance(n, ast.Constant)
                    and isinstance(n.value, str)
                    and id(n) not in docstrings
                ):
                    targets += rx.findall(n.value)
                found += [(rel, n.lineno, t) for t in targets]
    return sorted(set(found))


#: (importer, imported): a subpackage that must never import another (ADR-162 decision 1:
#: an app that needs sign-in alone installs Common and uses identity without intelligence).
FORBIDDEN = [("closelistening_common.identity", "closelistening_common.intelligence")]


def forbidden_imports(manifest: dict) -> list[tuple[str, int, str]]:
    """Every import in ``apps/`` that crosses a FORBIDDEN line, as (file, line, target)."""
    repo_of = {spec["package"]: repo for repo, spec in manifest.items()}
    found: list[tuple[str, int, str]] = []
    for importer, imported in FORBIDDEN:
        repo_src = sc.APPS / repo_of[importer.split(".")[0]] / "src"
        for p in (repo_src / importer.replace(".", "/")).rglob("*.py"):
            rel = p.relative_to(repo_src).with_suffix("").as_posix().replace("/", ".")
            module = rel.removesuffix(".__init__")
            is_pkg = p.name == "__init__.py"
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n, ast.ImportFrom):
                    names = [sc.absolute_from(module, is_pkg, n)]
                    names += [f"{names[0]}.{a.name}" for a in n.names]
                elif isinstance(n, ast.Import):
                    names = [a.name for a in n.names]
                else:
                    continue
                if any(x == imported or x.startswith(imported + ".") for x in names):
                    found.append((str(p.relative_to(sc.APPS)), n.lineno, imported))
    return sorted(set(found))


def roots(fails: dict[str, str]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for msg in fails.values():
        m = _ROOT_CAUSE.search(msg)
        counter[m.group(1) if m else msg[:80]] += 1
    return counter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edges", action="store_true", help="print every edge with file:line")
    ap.add_argument("--fails", action="store_true", help="print every failing module and why")
    args = ap.parse_args()

    manifest = yaml.safe_load(sc.MANIFEST.read_text())
    mapping, _owner = sc.build_mapping(manifest, sc.tracked_files())
    moved = {old for old, new in mapping.items() if old != new}
    private_pkgs = [spec["package"] for spec in manifest.values()]
    for repo in manifest:
        if not (sc.APPS / repo / "src").is_dir():
            print(f"error: apps/{repo} is empty; run split_copy.py first", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory(prefix="split-probe-") as tmp:
        wt = Path(tmp) / "public"
        subprocess.run(
            ["git", "worktree", "add", "--quiet", "--detach", str(wt), "HEAD"],
            cwd=ROOT,
            check=True,
        )
        try:
            gone = prune(wt, manifest, mapping)
            public = import_all([wt / "src"], ["podcast_scraper"], wt)
            found = edges(wt, moved)
            private = import_all(
                [wt / "src", *(sc.APPS / r / "src" for r in manifest)], private_pkgs, wt
            )
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT, check=True)

    print(f"pruned {gone} paths from a worktree of HEAD")
    print(f"public without private: {len(public['fails'])} of {public['modules']} modules fail")
    for cause, n in roots(public["fails"]).most_common():
        print(f"  {n:3}  {cause}")
    if args.fails:
        for module, msg in sorted(public["fails"].items()):
            print(f"    {module}: {msg}")
    files = {f for f, _, _ in found}
    print(f"edges into moved code: {len(found)} in {len(files)} files")
    if args.edges:
        for f, line, target in found:
            print(f"  {f}:{line} -> {target}")
    print(f"private on top of public: {len(private['fails'])} of {private['modules']} modules fail")
    for cause, n in roots(private["fails"]).most_common():
        print(f"  {n:3}  {cause}")
    if args.fails:
        for module, msg in sorted(private["fails"].items()):
            print(f"    {module}: {msg}")
    crossings = forbidden_imports(manifest)
    print(f"forbidden imports between private subpackages: {len(crossings)}")
    for f, line, target in crossings:
        print(f"  {f}:{line} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
