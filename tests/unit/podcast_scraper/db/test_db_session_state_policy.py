"""Policy: no per-connection session state in app code (ADR-173).

Behind PgBouncer in transaction mode a session is not the app's: session advisory locks, ``SET``
outside a transaction, LISTEN/NOTIFY, temp tables and ``WITH HOLD`` cursors all break or leak
across clients. Transaction-scoped forms (``pg_advisory_xact_lock``, ``SET LOCAL``) are fine, and
connection-level settings go through the engines' ``options`` (``db/engines.py``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

#: (pattern, why). Matched against every string literal in app code.
FORBIDDEN = [
    (re.compile(r"\bpg_(try_)?advisory_lock(_shared)?\b", re.I), "session advisory lock"),
    (re.compile(r"\bpg_advisory_unlock\w*\b", re.I), "session advisory unlock"),
    (re.compile(r"^\s*(UN)?LISTEN\s+\w+\s*;?\s*$", re.I), "LISTEN"),
    (re.compile(r"^\s*NOTIFY\s+\w+\s*(,.*)?$|\bpg_notify\b", re.I), "NOTIFY"),
    (re.compile(r"\bCREATE\s+(GLOBAL\s+|LOCAL\s+)?TEMP(ORARY)?\s+TABLE\b", re.I), "temp table"),
    (re.compile(r"\bWITH\s+HOLD\b", re.I), "WITH HOLD cursor"),
    (re.compile(r"\bset_config\s*\([^)]*,\s*(false|'f'|0)\s*\)", re.I), "session set_config"),
    # SET is matched upper-case and lower-case, never title-case: error text and prose say
    # "Set FOO=bar to ...".
    (re.compile(r"^\s*SET\s+(SESSION\s+)?(ROLE|SESSION\s+AUTHORIZATION)\b"), "session SET"),
    (
        re.compile(r"^\s*SET\s+(SESSION\s+)?(?!LOCAL\b|TRANSACTION\b)[A-Za-z_][\w.]*\s*(=|TO\b)"),
        "session SET",
    ),
    (re.compile(r"^\s*set\s+(?i:(session\s+)?(role|session\s+authorization))\b"), "session SET"),
    (
        re.compile(r"^\s*set\s+(?i:(session\s+)?(?!local\b|transaction\b)[a-z_][\w.]*\s*(=|to\b))"),
        "session SET",
    ),
]

#: ``relative/path.py`` -> reason. Empty: nothing in app code needs session state.
ALLOWED: dict[str, str] = {}


def _strings(tree: ast.AST):
    """String literals that are not docstrings (prose says "set the ..." and "listen ...")."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def violations(source: str) -> list[tuple[int, str]]:
    found = []
    for lineno, text in _strings(ast.parse(source)):
        for pattern, why in FORBIDDEN:
            if pattern.search(text):
                found.append((lineno, why))
    return found


def _app_sources() -> list[Path]:
    roots = [ROOT / "src" / "podcast_scraper", *sorted((ROOT / "apps").glob("*/src"))]
    return [p for r in roots if r.is_dir() for p in sorted(r.rglob("*.py"))]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_advisory_lock(1)",
        "SELECT pg_try_advisory_lock(hashtext(:k))",
        "SELECT pg_advisory_unlock(1)",
        "SET search_path TO player",
        "SET SESSION lock_timeout = '1s'",
        "SET ROLE migrator",
        "set search_path to player",
        "set session lock_timeout = '1s'",
        "set role migrator",
        "set statement_timeout = 0",
        "LISTEN events",
        "NOTIFY events, 'x'",
        "CREATE TEMP TABLE t (x int)",
        "DECLARE c CURSOR WITH HOLD FOR SELECT 1",
        "SELECT set_config('search_path', 'player', false)",
    ],
)
def test_the_scanner_catches_session_state(sql: str) -> None:
    assert violations(f"q = {sql!r}")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_advisory_xact_lock(hashtext(:k))",
        "SET LOCAL statement_timeout = 5",
        "SELECT set_config('x', 'y', true)",
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
        "Set the value of the thing to one",
        "Set auto_speakers=True to enable speaker detection",
        "set local statement_timeout = 5",
        "set transaction isolation level serializable",
    ],
)
def test_the_scanner_passes_transaction_scoped_forms(sql: str) -> None:
    assert not violations(f"q = {sql!r}")


def test_app_code_holds_no_session_state() -> None:
    sources = _app_sources()
    assert sources, "found no app sources to scan"
    bad = []
    for path in sources:
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        bad += [f"{rel}:{n} {why}" for n, why in violations(path.read_text(encoding="utf-8"))]
    assert not bad, "per-connection session state breaks PgBouncer transaction mode:\n" + "\n".join(
        bad
    )
