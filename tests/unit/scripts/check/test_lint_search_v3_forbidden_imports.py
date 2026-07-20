"""Tests for scripts/check/lint_search_v3_forbidden_imports.py.

The guard exists to enforce RFC-107 §S / PRD-045 FR12: no code path
may re-enable the LanceDB native hybrid combine
(``_combine_hybrid_results`` / ``_normalize_scores``) — see #1205.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check" / "lint_search_v3_forbidden_imports.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lint_search_v3_guard", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rules_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.txt"
    path.write_text(
        "# comment\n"
        "_combine_hybrid_results\n"
        "_normalize_scores\n"
        "\n"
        "WHITELIST: src/whitelisted.py\n"
    )
    return path


@pytest.fixture
def src_root(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    return root


def test_parse_rules_extracts_symbols_and_whitelist(rules_file: Path) -> None:
    mod = _load_module()
    forbidden, whitelist = mod.parse_rules(rules_file)
    assert forbidden == ["_combine_hybrid_results", "_normalize_scores"]
    assert whitelist == {"src/whitelisted.py"}


def test_clean_file_produces_no_hits(rules_file: Path, src_root: Path) -> None:
    (src_root / "clean.py").write_text("def foo():\n    return 1\n")
    mod = _load_module()
    forbidden, whitelist = mod.parse_rules(rules_file)
    hits = mod.scan(src_root, forbidden, whitelist)
    assert hits == []


def test_comments_and_docstrings_are_ignored(rules_file: Path, src_root: Path) -> None:
    (src_root / "docs_only.py").write_text(
        '''"""This docstring names _combine_hybrid_results but never calls it.

    We MUST NOT re-enable _normalize_scores — see RFC-107 §S.
    """

# _combine_hybrid_results — do not use (RFC-107 §S)
def safe():
    return 1
'''
    )
    mod = _load_module()
    forbidden, whitelist = mod.parse_rules(rules_file)
    hits = mod.scan(src_root, forbidden, whitelist)
    assert hits == []


def test_real_call_site_is_flagged(rules_file: Path, src_root: Path) -> None:
    (src_root / "bad_call.py").write_text(
        "from somewhere import _combine_hybrid_results\n"
        "\n"
        "def bad():\n"
        "    return _combine_hybrid_results(1, 2)\n"
    )
    mod = _load_module()
    forbidden, whitelist = mod.parse_rules(rules_file)
    hits = mod.scan(src_root, forbidden, whitelist)
    assert len(hits) >= 2
    symbols = {h[2] for h in hits}
    assert "_combine_hybrid_results" in symbols


def test_real_attribute_access_is_flagged(rules_file: Path, src_root: Path) -> None:
    (src_root / "bad_attr.py").write_text("class X: pass\n" "x = X()\n" "y = x._normalize_scores\n")
    mod = _load_module()
    forbidden, whitelist = mod.parse_rules(rules_file)
    hits = mod.scan(src_root, forbidden, whitelist)
    assert any(h[2] == "_normalize_scores" for h in hits)


def test_whitelist_suppresses_real_call_site(rules_file: Path, src_root: Path) -> None:
    (src_root / "whitelisted.py").write_text(
        "from somewhere import _combine_hybrid_results\n"
        "def use():\n"
        "    return _combine_hybrid_results()\n"
    )
    mod = _load_module()
    forbidden, whitelist = mod.parse_rules(rules_file)
    hits = mod.scan(src_root, forbidden, whitelist)
    assert hits == []


def test_repo_lint_is_green() -> None:
    """The shipped repo state must pass the guard — this is a regression seat belt.

    If someone accidentally re-imports one of the forbidden symbols, this test
    fires at ci-fast time as a per-PR safety net redundant with make lint-search-v3.
    """
    mod = _load_module()
    rules = REPO_ROOT / ".github" / "lint" / "search-v3-forbidden-imports.txt"
    src = REPO_ROOT / "src" / "podcast_scraper"
    forbidden, whitelist = mod.parse_rules(rules)
    hits = mod.scan(src, forbidden, whitelist)
    assert hits == [], f"Forbidden-symbol call sites detected: {hits!r}"
