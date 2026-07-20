#!/usr/bin/env python3
"""Fail if any src/ file references a forbidden search-v3 symbol.

Consumed by ``make lint-search-v3``. Rule lives in
``.github/lint/search-v3-forbidden-imports.txt``.

Rule (RFC-107 §S, PRD-045 FR12; #1205):
    The LanceDB native hybrid combine
    (``_combine_hybrid_results`` / ``_normalize_scores`` via
    ``pyarrow.compute``) segfaulted the api under stack-test. Fix
    ``0fe0854b`` bypassed the native combine via the Python-side
    ``search_bm25 + search_vector + rrf_fuse`` fan-out in ``retrieval.py``.
    No code path may re-enable it until #1205's root cause is fixed
    upstream and independently verified.

Exit 0 = clean. Exit 1 = at least one hit outside the whitelist.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def parse_rules(path: Path) -> tuple[list[str], set[str]]:
    """Return (forbidden_symbols, whitelisted_relative_paths).

    Lines starting with ``#`` or empty are ignored. Lines starting with
    ``WHITELIST:`` name a repo-relative file path that may reference the
    forbidden symbols; everything else is a forbidden symbol.
    """
    forbidden: list[str] = []
    whitelist: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("WHITELIST:"):
            entry = line[len("WHITELIST:") :].strip()
            if entry:
                whitelist.add(entry)
            continue
        forbidden.append(line)
    return forbidden, whitelist


def _scan_ast(path: Path, source: str, forbidden: set[str]) -> list[tuple[Path, int, str, str]]:
    """AST-walk source; return real code references to any forbidden symbol.

    Ignores comments and string literals — the rule (RFC-107 §S1) is "no new
    call site", not "no mention". A comment documenting the #1205 fix or a
    docstring naming the forbidden symbol is fine; only actual Name /
    Attribute references (calls, imports, attribute access) fail the lint.
    """
    hits: list[tuple[Path, int, str, str]] = []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return hits
    lines = source.splitlines()
    for node in ast.walk(tree):
        found: str | None = None
        if isinstance(node, ast.Name) and node.id in forbidden:
            found = node.id
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            found = node.attr
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                if alias.name in forbidden or (alias.asname and alias.asname in forbidden):
                    found = alias.name if alias.name in forbidden else (alias.asname or alias.name)
                    break
        if found is None:
            continue
        lineno = getattr(node, "lineno", 0) or 0
        line = lines[lineno - 1].rstrip() if 0 < lineno <= len(lines) else ""
        hits.append((path, lineno, found, line))
    return hits


def scan(root: Path, forbidden: list[str], whitelist: set[str]) -> list[tuple[Path, int, str, str]]:
    """Return (path, line_no, symbol, line_text) for every hit outside the whitelist."""
    hits: list[tuple[Path, int, str, str]] = []
    forbidden_set = set(forbidden)
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(root.parent).as_posix()
        if rel in whitelist:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Cheap prefilter: skip files that don't textually contain any
        # forbidden symbol at all (avoids parsing every file in the tree).
        if not any(sym in text for sym in forbidden):
            continue
        hits.extend(_scan_ast(path, text, forbidden_set))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path(".github/lint/search-v3-forbidden-imports.txt"),
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("src/podcast_scraper"),
    )
    args = parser.parse_args()

    if not args.rules.exists():
        print(f"lint-search-v3: rules file not found: {args.rules}", file=sys.stderr)
        return 2
    if not args.src.exists():
        print(f"lint-search-v3: src root not found: {args.src}", file=sys.stderr)
        return 2

    forbidden, whitelist = parse_rules(args.rules)
    if not forbidden:
        print("lint-search-v3: no forbidden symbols configured (nothing to check)")
        return 0

    hits = scan(args.src, forbidden, whitelist)
    if hits:
        print(
            f"lint-search-v3: {len(hits)} forbidden-symbol hit(s) under {args.src} "
            f"(rules: {args.rules})",
            file=sys.stderr,
        )
        for path, lineno, sym, line in hits:
            print(f"  {path}:{lineno}: {sym!r}: {line}", file=sys.stderr)
        print(
            "\nSee .github/lint/search-v3-forbidden-imports.txt for the rule + "
            "whitelist mechanism. Adding a whitelist entry requires an ADR update "
            "per RFC-107 §S.",
            file=sys.stderr,
        )
        return 1

    checked = sum(1 for _ in args.src.rglob("*.py"))
    print(
        f"lint-search-v3: OK ({checked} files scanned; "
        f"{len(forbidden)} forbidden symbols; "
        f"{len(whitelist)} whitelisted paths)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
