"""Every cache in the served code says how it is invalidated (2026-10-09).

The case of record: the storyline and theme label maps behind ``resolve_entity`` were
``lru_cache``d on the corpus root ALONE, so after a re-enrichment rewrote their artifacts, search
resolved names against the old clusters until the process restarted. Nothing listed the caches, so
nothing asked what each one's freshness signal was.

This finds every ``lru_cache`` / ``functools.cache`` function and every module-level dict whose name
contains "cache" under ``server/`` and ``search/``, and requires each to be in ``CONTRACT`` with its
invalidation. A ``token`` entry must name a parameter the function really takes — an mtime or a
time bucket the CALLER computes — which is what makes a cached answer go stale when its input does.
``perf_cache.get_or_compute`` callers are not listed: its signature already requires a token.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "src" / "podcast_scraper"
SCANNED = ("server", "search")

# "<path relative to src/podcast_scraper>::<name>" -> (kind, detail)
#   token:  detail = the parameter that carries the freshness token
#   ttl:    detail = how long, and why staleness that long is acceptable
#   mtime:  detail = the stat the entry is checked against on every read
#   static: detail = why the cached value can never change while the process runs
CONTRACT: dict[str, tuple[str, str]] = {
    "server/app_relational_view.py::_theme_ref_by_norm_at": ("token", "_token"),
    "server/app_relational_view.py::_storyline_ref_by_norm_at": ("token", "_token"),
    "server/og/build.py::_trend_map_cached": ("token", "_mtime"),
    "server/og/card.py::_font": ("static", "a font file shipped with the code"),
    "server/app_discover_view.py::_INTEREST_INDEX_CACHE": (
        "mtime",
        "search/metadata.json (mtime, size), checked on every read",
    ),
    "server/routes/app_episodes.py::_related_cache": (
        "ttl",
        "bounded TTL; related episodes are corpus content, not user state",
    ),
    "server/routes/app_episodes.py::_episode_reach_cache": (
        "ttl",
        "30 s; anonymous aggregate counts",
    ),
    "server/routes/app_relational.py::_conversation_arc_cache": ("ttl", "30 s"),
    "server/routes/corpus_enrichments.py::_CATALOGUE_CACHE": (
        "mtime",
        "the enrichments directory mtime, checked on every read",
    ),
    "search/index_source_mtime.py::_cache": (
        "ttl",
        "10 s; also dropped by invalidate_newest_index_source_mtime_cache",
    ),
}


def _found() -> dict[str, list[str]]:
    """``key -> parameter names`` (empty for module-level dicts)."""
    out: dict[str, list[str]] = {}
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorators = [ast.unparse(d) for d in node.decorator_list]
                    if any(
                        "lru_cache" in d or d in ("cache", "functools.cache") for d in decorators
                    ):
                        out[f"{rel}::{node.name}"] = [a.arg for a in node.args.args]
            for node in tree.body:
                name, value = None, None
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    name, value = node.target.id, node.value
                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    name, value = node.targets[0].id, node.value
                if (
                    name
                    and "cache" in name.lower()
                    and isinstance(value, (ast.Dict, ast.Call))
                    and not (isinstance(value, ast.Call) and "lru_cache" in ast.unparse(value))
                ):
                    out[f"{rel}::{name}"] = []
    return out


def test_the_scan_still_finds_caches() -> None:
    found = _found()
    assert "server/app_relational_view.py::_storyline_ref_by_norm_at" in found
    assert len(found) >= 8


def test_every_cache_states_its_invalidation() -> None:
    missing = sorted(set(_found()) - set(CONTRACT))
    assert not missing, (
        "New cache(s) with no stated invalidation — add each to CONTRACT with a token parameter, "
        f"a TTL, an mtime check, or why it is static: {missing}"
    )


def test_no_stale_contract_entries() -> None:
    assert not sorted(set(CONTRACT) - set(_found()))


def test_a_token_cache_really_takes_its_token() -> None:
    found = _found()
    for key, (kind, detail) in CONTRACT.items():
        if kind == "token":
            assert detail in found.get(key, []), (
                f"{key} is listed as keyed on {detail!r} but takes no such parameter — "
                "a cache keyed on its inputs alone never sees the artifact change"
            )
