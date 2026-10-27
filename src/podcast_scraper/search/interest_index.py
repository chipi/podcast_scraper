"""``search/interest_index.json``: interest token -> the episodes (metadata relpaths) carrying it.

The serving path needs only this map, but it used to derive it from ``search/metadata.json``,
which holds every row of the two-tier index (500 MB on prod, 2026-10-27). One ``json.loads`` of
that file inside a request allocated several GB while holding the GIL and got the player api
OOM-killed. The indexer already has every row's metadata in memory when it writes
``metadata.json``, so it writes this small map next to it, and readers load only the map.

Only ``kg_topic`` / ``kg_entity`` rows contribute: their ``source_id`` (``topic:…`` /
``person:…``) is the same id space interests live in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

INTEREST_INDEX_FILE = "interest_index.json"
KG_DOC_TYPES = ("kg_topic", "kg_entity")


def build_interest_index(index_metadata: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Token -> sorted metadata relpaths, from the index's ``doc_id -> meta`` map."""
    out: Dict[str, set] = {}
    for entry in index_metadata.values():
        if not isinstance(entry, dict) or entry.get("doc_type") not in KG_DOC_TYPES:
            continue
        token = entry.get("source_id")
        relpath = entry.get("source_metadata_relative_path")
        if isinstance(token, str) and isinstance(relpath, str):
            out.setdefault(token, set()).add(relpath)
    return {token: sorted(paths) for token, paths in sorted(out.items())}


def write_interest_index(search_dir: Path, index: Mapping[str, List[str]]) -> Path:
    """Write the map atomically (temp file + rename), so a reader never sees a torn file."""
    search_dir = Path(search_dir)
    target = search_dir / INTEREST_INDEX_FILE
    tmp = search_dir / f".{INTEREST_INDEX_FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    return target


def rebuild_from_metadata(search_dir: Path) -> int:
    """Write the map from an existing ``metadata.json``; returns the number of tokens.

    For a corpus indexed before the indexer wrote this file. It parses the big sidecar once, so
    run it as a one-off command, never inside a serving process.
    """
    search_dir = Path(search_dir)
    raw = json.loads((search_dir / "metadata.json").read_text(encoding="utf-8"))
    index = build_interest_index(raw if isinstance(raw, dict) else {})
    write_interest_index(search_dir, index)
    return len(index)


def main(argv: List[str] | None = None) -> int:
    """``python -m podcast_scraper.search.interest_index <corpus_dir>``: rebuild the map once."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m podcast_scraper.search.interest_index <corpus_dir>")
        return 2
    search_dir = Path(args[0]) / "search"
    if not (search_dir / "metadata.json").is_file():
        print(f"no {search_dir / 'metadata.json'}")
        return 1
    tokens = rebuild_from_metadata(search_dir)
    print(f"wrote {search_dir / INTEREST_INDEX_FILE}: {tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
