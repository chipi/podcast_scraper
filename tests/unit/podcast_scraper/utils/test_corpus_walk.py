"""A corpus walk never collects its bookkeeping copies as content.

Prod 2026-10-07: ``.podcast_scraper/upgrade-backups/`` held 593 GI, 401 KG and 2,826 metadata
copies from BEFORE each repair; 1,444 of the 2,587 artifacts the operator viewer's graph listed,
and 125 of the KG files the entity-id map was built from, were those copies.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List

import pytest

from podcast_scraper.gi.explore import scan_artifact_paths
from podcast_scraper.kg.corpus import scan_kg_artifact_paths
from podcast_scraper.utils.corpus_walk import corpus_rglob, prune_excluded_dirs

pytestmark = [pytest.mark.unit]

LIVE = "feeds/f/run_1/metadata/ep"
BOOKKEEPING = (
    ".podcast_scraper/upgrade-backups/0020/feeds/f/run_1/metadata/ep",
    ".trash/20261001T000000Z/feeds/f/run_1/metadata/ep",
    "feeds/f/.trash/run_0/metadata/ep",
)


def _corpus(root: Path) -> None:
    for stem in (LIVE, *BOOKKEEPING):
        for suffix in (".gi.json", ".kg.json", ".metadata.json"):
            path = root / f"{stem}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")


def _rels(root: Path, paths) -> set:
    return {Path(p).relative_to(root).as_posix() for p in paths}


def test_corpus_rglob_skips_every_dot_directory(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert _rels(tmp_path, corpus_rglob(tmp_path, "*.gi.json")) == {f"{LIVE}.gi.json"}
    # A pattern with a directory part, as glob("**/metadata/*.metadata.json") was written.
    assert _rels(tmp_path, corpus_rglob(tmp_path, "metadata/*.metadata.json")) == {
        f"{LIVE}.metadata.json"
    }


def test_a_root_that_itself_sits_under_a_dot_directory_is_walked(tmp_path: Path) -> None:
    root = tmp_path / ".hidden-parent" / "corpus"
    _corpus(root)
    assert _rels(root, corpus_rglob(root, "*.kg.json")) == {f"{LIVE}.kg.json"}


def test_prune_excluded_dirs_keeps_the_walk_out() -> None:
    dirnames = ["feeds", ".podcast_scraper", ".trash", "run_1", ".viewer"]
    prune_excluded_dirs(dirnames)
    assert dirnames == ["feeds", "run_1"]


def test_the_gi_and_kg_scanners_ignore_backups(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert _rels(tmp_path, scan_artifact_paths(tmp_path)) == {f"{LIVE}.gi.json"}
    assert _rels(tmp_path, scan_kg_artifact_paths(tmp_path)) == {f"{LIVE}.kg.json"}


# The guard. A recursive scan written as ``root.rglob(...)`` walks straight into the backups; it
# happened in `upgrade verify` (2026-10-03, fixed alone) and then in ~30 other places. New corpus
# scanners use corpus_rglob; anything that walks a NON-corpus tree is named here with its reason.
_SRC = Path(__file__).resolve().parents[4] / "src" / "podcast_scraper"
_NOT_A_CORPUS = {
    "utils/corpus_walk.py": "the helper itself",
    "cache/manager.py": "model cache dirs",
    "providers/ml/model_loader.py": "model cache dirs",
    "providers/ml/summarizer.py": "model cache dirs",
    "search/lance_index_stats.py": "the vector index dir",
    "search/corpus_scope.py": "its os.walk calls prune with the same rule",
    "server/routes/ops.py": "disk usage: sizes everything, bookkeeping included",
    "utils/usage_status.py": "run/cost logs, not artifacts",
    "upgrade/migrations/m0011_shared_artwork_store.py": "the artwork store dir",
    "upgrade/migrations/m0013_artwork_thumbnails.py": "the artwork store dir",
    "upgrade/migrations/m0026_artwork_medium.py": "the artwork store dir",
}


def _raw_walks(tree: ast.AST) -> List[int]:
    """Lines calling ``.rglob(...)``, ``.glob("**/...")`` or ``os.walk(...)`` — code, not text."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        first = node.args[0] if node.args else None
        literal = first.value if isinstance(first, ast.Constant) else None
        if isinstance(first, ast.JoinedStr) and first.values:
            head = first.values[0]
            literal = head.value if isinstance(head, ast.Constant) else None
        if (
            name == "rglob"
            or name == "walk"
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id == "os")
        ):
            hits.append(node.lineno)
        elif name == "glob" and isinstance(literal, str) and literal.startswith("**/"):
            hits.append(node.lineno)
    return hits


def test_no_corpus_scanner_walks_without_the_exclusion_rule() -> None:
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        if rel in _NOT_A_CORPUS:
            continue
        source = path.read_text(encoding="utf-8")
        walks = _raw_walks(ast.parse(source))
        # An os.walk that prunes with the rule is fine; nothing else is.
        lines = source.splitlines()
        for n in walks:
            if "os.walk" in lines[n - 1] and "prune_excluded_dirs" in source:
                continue
            offenders.append(f"{rel}:{n}: {lines[n - 1].strip()}")
    assert not offenders, "use corpus_rglob / prune_excluded_dirs:\n" + "\n".join(offenders)
