#!/usr/bin/env python3
"""Copy the private surfaces of ADR-158 into ./apps/<repo>/, rewriting imports.

Re-runnable by design: each run wipes ``apps/<repo>/`` (everything except ``.git``)
and copies again from the tracked files of this checkout, so the copy never drifts
from ``main``. The public tree is never modified.

What moves is listed in ``scripts/tools/split_manifest.yaml``:

* ``modules`` — Python modules that move to the repo's own package. Every import of
  them, in every copied file, is rewritten to the new location.
* ``packages`` — whole Python packages that move and are renamed.
* ``verbatim_packages`` — whole top-level packages that move under their own name.
* ``subpackages`` — named groups inside the repo's package (``<package>.<name>``), each
  with its own ``modules`` / ``packages``.
* ``trees`` / ``file_trees`` / ``files`` — copied byte for byte.
* ``entry_points`` — ``name: module:attr`` registered under ``podcast_scraper.extensions``
  in the generated ``pyproject.toml``, the module rewritten to its new location.
* ``forked_trees`` — copied byte for byte like ``trees``, but the public repo keeps its
  own copy (the probe does not prune them).

Tests under ``tests/`` that reference a moved module are copied too (with the
``conftest.py`` files above them), into the repo that owns most of what they
reference.

Usage::

    python scripts/tools/split_copy.py            # copy into apps/
    python scripts/tools/split_copy.py --dry-run  # print what would be copied
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts" / "tools" / "split_manifest.yaml"
APPS = ROOT / "apps"


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def module_of(path: str) -> str:
    """``src/a/b/c.py`` -> ``a.b.c``; ``src/a/b/__init__.py`` -> ``a.b``."""
    parts = Path(path).with_suffix("").parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_mapping(manifest: dict, files: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (old module -> new module, old module -> repo)."""
    mapping: dict[str, str] = {}
    owner: dict[str, str] = {}
    for repo, top in manifest.items():
        for pkg, spec in units(top):
            _map_unit(repo, pkg, spec, files, mapping, owner)
    return mapping, owner


def units(spec: dict) -> list[tuple[str, dict]]:
    """The (package, spec) pairs one repo entry defines: itself, then each of its
    ``subpackages`` as ``<package>.<name>``."""
    pkg = spec["package"]
    return [(pkg, spec)] + [
        (f"{pkg}.{name}", sub) for name, sub in spec.get("subpackages", {}).items()
    ]


def _map_unit(repo: str, pkg: str, spec: dict, files, mapping: dict, owner: dict) -> None:
    for parent, names in spec.get("modules", {}).items():
        tail = parent.removeprefix("podcast_scraper.")
        if tail.startswith("enrichment."):
            tail = tail.removeprefix("enrichment.")
        for name in names:
            old = f"{parent}.{name}"
            # A top-level module (``podcast_scraper.<name>``) lands at the package root.
            mapping[old] = (
                f"{pkg}.{name}" if parent == "podcast_scraper" else f"{pkg}.{tail}.{name}"
            )
            owner[old] = repo
    for old_pkg, new_tail in spec.get("packages", {}).items():
        prefix = "src/" + old_pkg.replace(".", "/") + "/"
        for f in files:
            if f.startswith(prefix) and f.endswith(".py"):
                old = module_of(f)
                mapping[old] = f"{pkg}.{new_tail}" + old[len(old_pkg) :]
                owner[old] = repo
    for top in spec.get("verbatim_packages", []):
        for f in files:
            if f.startswith(f"src/{top}/") and f.endswith(".py"):
                old = module_of(f)
                mapping[old] = old
                owner[old] = repo


def new_path(new_module: str, is_package: bool) -> str:
    rel = new_module.replace(".", "/")
    return f"src/{rel}/__init__.py" if is_package else f"src/{rel}.py"


def absolute_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    parts = module.split(".")
    base = parts if is_package else parts[:-1]
    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    return ".".join(base + ([node.module] if node.module else []))


def rewrite(source: str, module: str | None, is_package: bool, mapping: dict[str, str]) -> str:
    """Rewrite imports of moved modules. ``module`` is the file's ORIGINAL dotted name
    (None for test files), used to resolve relative imports, which all become absolute
    in moved files because their package changes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level and module:
            absmod: str | None = absolute_from(module, is_package, node)
        else:
            absmod = node.module
        if absmod is None:
            continue
        moved_mod = mapping.get(absmod)
        groups: list[tuple[str, list[ast.alias]]] = []
        for alias in node.names:
            full = f"{absmod}.{alias.name}"
            if full in mapping:  # `from pkg import moved_module`
                new_parent, _, new_name = mapping[full].rpartition(".")
                asname = alias.asname or (alias.name if new_name != alias.name else None)
                groups.append((new_parent, [ast.alias(name=new_name, asname=asname)]))
            else:
                groups.append((moved_mod or absmod, [alias]))
        changed = node.level or moved_mod or any(g[0] != absmod for g in groups)
        if not changed:
            continue
        merged: dict[str, list[ast.alias]] = {}
        for parent, aliases in groups:
            merged.setdefault(parent, []).extend(aliases)
        first = lines[node.lineno - 1]
        indent = first[: len(first) - len(first.lstrip())]
        text = "".join(
            indent + ast.unparse(ast.ImportFrom(module=parent, names=aliases, level=0)) + "\n"
            for parent, aliases in merged.items()
        )
        edits.append((node.lineno, node.end_lineno or node.lineno, text))
    for start, end, text in sorted(edits, reverse=True):
        lines[start - 1 : end] = [text]
    out = "".join(lines)
    # Dotted references outside import statements: `import a.b.c`, mock.patch targets,
    # importlib strings. Longest first so a package prefix never shadows a module.
    for old in sorted(mapping, key=len, reverse=True):
        new = mapping[old]
        if old != new:
            out = re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", new, out)
    return out


def references(source: str, mapping: dict[str, str], owner: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for old in mapping:
        parent, _, name = old.rpartition(".")
        hits = len(re.findall(rf"(?<![\w.]){re.escape(old)}(?![\w])", source))
        pattern = rf"from {re.escape(parent)} import \(?[^)]*\b{re.escape(name)}\b"
        hits += len(re.findall(pattern, source))
        if hits:
            counts[owner[old]] = counts.get(owner[old], 0) + hits
    return counts


def write(dest: Path, data: bytes, dry_run: bool) -> None:
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def copy_file(src: Path, dest: Path, dry_run: bool) -> None:
    # Keeps the mode: gradlew and the stack scripts must stay executable.
    write(dest, src.read_bytes(), dry_run)
    if not dry_run:
        shutil.copymode(src, dest)


def wipe(manifest: dict, dry_run: bool) -> bool:
    for repo in manifest:
        target = APPS / repo
        if not (target / ".git").exists():
            print(f"error: {target} is not a git repo; run `git init` there first", file=sys.stderr)
            return False
        if not dry_run:
            for child in target.iterdir():
                if child.name != ".git":
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
    return True


def copy_modules(files, mapping, owner, bump, dry_run: bool) -> None:
    for f in files:
        if not (f.startswith("src/") and f.endswith(".py")) or module_of(f) not in mapping:
            continue
        old = module_of(f)
        is_pkg = f.endswith("__init__.py")
        out = rewrite((ROOT / f).read_text(), old, is_pkg, mapping)
        write(APPS / owner[old] / new_path(mapping[old], is_pkg), out.encode(), dry_run)
        bump(owner[old], "python modules")
    if dry_run:
        return
    # Every new package directory needs an __init__.py up to src/.
    for old, new in mapping.items():
        for parent in Path(new_path(new, False)).parents:
            if parent in (Path("src"), Path(".")):
                break
            init = APPS / owner[old] / parent / "__init__.py"
            if not init.exists():
                write(init, b"", False)


def copy_verbatim(manifest: dict, files, bump, dry_run: bool) -> None:
    for repo, spec in manifest.items():
        trees = {**spec.get("trees", {}), **spec.get("forked_trees", {})}
        for tree, dest in trees.items():
            for f in (f for f in files if f.startswith(tree + "/")):
                copy_file(ROOT / f, APPS / repo / dest / f[len(tree) + 1 :], dry_run)
                bump(repo, f"tree {tree}")
        for tree in spec.get("file_trees", []):
            for f in (f for f in files if f.startswith(tree + "/")):
                copy_file(ROOT / f, APPS / repo / f, dry_run)
                bump(repo, "docs")
        for f in spec.get("files", []):
            copy_file(ROOT / f, APPS / repo / f, dry_run)
            bump(repo, "docs")


def copy_tests(files, mapping, owner, bump, dry_run: bool) -> None:
    """Tests that reference a moved module go to the repo owning most of what they
    reference (player wins ties: it depends on common), with their conftest chain."""
    for f in files:
        if not (f.startswith("tests/") and f.endswith(".py")) or Path(f).name == "conftest.py":
            continue
        src = (ROOT / f).read_text(errors="ignore")
        refs = references(src, mapping, owner)
        if not refs:
            continue
        repo = "player" if refs.get("player") else max(refs, key=lambda r: refs[r])
        write(APPS / repo / f, rewrite(src, None, False, mapping).encode(), dry_run)
        bump(repo, "tests")
        for parent in Path(f).parents:
            conf = parent / "conftest.py"
            if str(conf) in files and not (APPS / repo / conf).exists():
                text = rewrite((ROOT / conf).read_text(), None, False, mapping)
                write(APPS / repo / conf, text.encode(), dry_run)


def write_scaffold(manifest: dict, mapping: dict[str, str], dry_run: bool) -> None:
    for repo, spec in manifest.items():
        pyproject = (
            "[project]\n"
            f'name = "{spec["package"].replace("_", "-")}"\n'
            'version = "0.0.0"\n'
            'requires-python = ">=3.12"\n'
            "\n[tool.setuptools.packages.find]\n"
            'where = ["src"]\n'
        )
        if spec.get("entry_points"):
            pyproject += '\n[project.entry-points."podcast_scraper.extensions"]\n'
            for name, target in spec["entry_points"].items():
                module, _, attr = target.partition(":")
                pyproject += f'{name} = "{mapping[module]}:{attr}"\n'
        write(APPS / repo / "pyproject.toml", pyproject.encode(), dry_run)
        readme = (
            f"# {spec['package']}\n\nGenerated by `scripts/tools/split_copy.py` in the public "
            "repo (ADR-158). Do not edit here: every run replaces this tree.\n"
        )
        write(APPS / repo / "README.md", readme.encode(), dry_run)
        ignore = "__pycache__/\n*.py[cod]\n*.egg-info/\n.pytest_cache/\n.mypy_cache/\n"
        ignore += "".join(f"{rule}\n" for rule in spec.get("gitignore", []))
        write(APPS / repo / ".gitignore", ignore.encode(), dry_run)
        init = APPS / repo / "src" / spec["package"] / "__init__.py"
        if not dry_run and not init.exists():
            write(init, b"", False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # An ``own_source`` repo (Studio) is edited in place, not generated: never wiped or
    # scaffolded, or a re-run deletes it.
    manifest = {
        repo: spec
        for repo, spec in yaml.safe_load(MANIFEST.read_text()).items()
        if not spec.get("own_source")
    }
    files = tracked_files()
    mapping, owner = build_mapping(manifest, files)
    if not wipe(manifest, args.dry_run):
        return 1

    counts: dict[str, dict[str, int]] = {r: {} for r in manifest}

    def bump(repo: str, kind: str) -> None:
        counts[repo][kind] = counts[repo].get(kind, 0) + 1

    copy_modules(files, mapping, owner, bump, args.dry_run)
    copy_verbatim(manifest, files, bump, args.dry_run)
    copy_tests(files, mapping, owner, bump, args.dry_run)
    write_scaffold(manifest, mapping, args.dry_run)

    for repo, c in counts.items():
        print(f"{repo}: " + ", ".join(f"{v} {k}" for k, v in sorted(c.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
