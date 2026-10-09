"""Every call to a metadata name reader passes the feed's language (D13, D22).

D13 gave `is_network_or_org_author` / `has_org_markers` a language and claimed every pipeline call
passed it; eleven calls did not, and Senado's feed seated "Rádio Senado" as its host because the
Portuguese marker `rádio` was never consulted. D22 gave the same parameter to the readers below. A
parameter declared and not passed fails silently (every default reads English), so this asserts
the delivery, by AST, across `src/` — the frozen migrations excepted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import podcast_scraper

SRC = Path(podcast_scraper.__file__).parent

#: callee -> the positional index of `language` (it may also be passed by keyword).
READERS = {
    "split_author_names": 1,
    "names_the_show": 2,
    "looks_like_a_person_name": 1,
    "is_plausible_mononym": 1,
    "is_network_or_org_author": 1,
    "has_org_markers": 1,
    "looks_like_publisher": 1,
    "normalize_host_names": None,
    "drop_non_person_names": 3,
    "recurrent_hosts_across_episodes": None,
    "_speaker_lists_for_graph": 2,
    "_intro_names": 1,
}

#: (file relative to the package, enclosing function) -> why English alone is right there.
ENGLISH_BY_DESIGN = {
    ("kg/llm_extract.py", "person_entity_names"): "reads the KG model's output, English (D-44)",
    ("gi/pipeline.py", "_label_names_the_show"): (
        "GI attributes quotes in the English canonical body; the label is already resolved"
    ),
    ("server/feed_signals.py", "_accumulate_kg_entities"): (
        "KG person nodes, extracted from the English canonical body (D-44)"
    ),
    ("providers/ml/diarization/roster.py", "_greeted_names"): (
        "its greeting patterns are English-only (`_GUEST_GREETED_RE`); listed as an open reader"
    ),
}


def _enclosing(tree: ast.AST) -> dict[int, str]:
    out: dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    # innermost wins: walk order visits outer functions first
                    out[id(node)] = fn.name
    return out


def _calls_without_language() -> list[str]:
    missing = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("upgrade/migrations/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        where = _enclosing(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name not in READERS:
                continue
            pos = READERS[name]
            passed = any(k.arg == "language" for k in node.keywords) or (
                pos is not None and len(node.args) > pos
            )
            if passed or (rel, where.get(id(node), "")) in ENGLISH_BY_DESIGN:
                continue
            missing.append(f"{rel}:{node.lineno} {name}() in {where.get(id(node), '<module>')}")
    return missing


def test_every_name_reader_call_passes_the_feed_language() -> None:
    assert _calls_without_language() == []


@pytest.mark.parametrize("key", sorted(ENGLISH_BY_DESIGN))
def test_each_english_exception_still_exists(key: tuple[str, str]) -> None:
    """An exception whose function is gone is a stale excuse, not a decision."""
    rel, func = key
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == func for n in ast.walk(tree)), key
