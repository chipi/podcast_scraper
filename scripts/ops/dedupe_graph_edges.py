#!/usr/bin/env python3
"""Remove exact-duplicate edges from every served gi.json / kg.json. Dry-run unless ``--apply``.

Why: before ``d752d21e5`` every enrich-edges run copied SPOKEN_BY edges again (prod 2026-10-09:
230,665 byte-identical edges in 110 gi.json, plus 62 in kg.json). The fix stops new copies and
cleans the files a later enrich-edges run touches; this pass cleans the rest, once.

An edge is a duplicate only when it is byte-identical (sorted-key JSON) to an earlier edge in the
same file; the first one is kept, nothing else in the file changes. Each rewrite is backed up and
receipted with the migrations' own helpers, so ``--undo`` restores every file still as this pass
left it. After an apply the enrich-edges cache stamp is bumped, so the API reloads the edges.

    python scripts/ops/dedupe_graph_edges.py --corpus /app/output            # dry-run
    python scripts/ops/dedupe_graph_edges.py --corpus /app/output --apply
    python scripts/ops/dedupe_graph_edges.py --corpus /app/output --undo

Exit 1 when an apply leaves any duplicate behind.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

TAG = "dedupe-graph-edges"
RECEIPTS_FILE = "dedupe_graph_edges.jsonl"


def dedupe_edges(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``(payload with exact-duplicate edges dropped, the dropped copies)``; first copy kept."""
    edges = payload.get("edges")
    if not isinstance(edges, list):
        return payload, []
    seen: Set[str] = set()
    kept: List[Any] = []
    dropped: List[Dict[str, Any]] = []
    for edge in edges:
        if isinstance(edge, dict):
            key = json.dumps(edge, sort_keys=True, default=str)
            if key in seen:
                dropped.append(edge)
                continue
            seen.add(key)
        kept.append(edge)
    if not dropped:
        return payload, []
    return {**payload, "edges": kept}, dropped


def _artifacts(root: Path) -> List[Path]:
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts

    out: List[Path] = []
    for suffix in (".gi.json", ".kg.json"):
        out.extend(select_served_artifacts(root, suffix)[0])
    return out


def run(root: Path, apply: bool, logger: logging.Logger) -> Dict[str, Any]:
    from podcast_scraper.upgrade.file_rewrite import append_receipts, write_with_backup

    stats: Dict[str, Any] = {"files": 0, "gi_files": 0, "kg_files": 0, "edges_dropped": 0}
    by_type: Dict[str, int] = {}
    receipts: List[Dict[str, str]] = []
    for path in _artifacts(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("skip %s (%s)", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        out, dropped = dedupe_edges(payload)
        if not dropped:
            continue
        for edge in dropped:
            by_type[str(edge.get("type"))] = by_type.get(str(edge.get("type")), 0) + 1
        stats["files"] += 1
        stats["gi_files" if path.name.endswith(".gi.json") else "kg_files"] += 1
        stats["edges_dropped"] += len(dropped)
        if apply:
            receipts.append(write_with_backup(root, TAG, path, out))
    stats["by_type"] = by_type
    if apply and receipts:
        append_receipts(root, RECEIPTS_FILE, {"tag": TAG, "files": len(receipts)}, receipts)
        from podcast_scraper.search.cli_handlers import _write_edges_stamp

        _write_edges_stamp(
            root, {"gi_rewritten": stats["gi_files"], "kg_scoped": stats["kg_files"]}, logger
        )
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    mode.add_argument("--undo", action="store_true", help="restore every file this pass wrote")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("dedupe_graph_edges")
    root = args.corpus.resolve()

    if args.undo:
        from podcast_scraper.upgrade.file_rewrite import undo_from_receipts

        restored, refused = undo_from_receipts(root, RECEIPTS_FILE, TAG, TAG)
        print(json.dumps({"restored": restored, "refused": refused[:20], "total": len(refused)}))
        return 1 if refused else 0

    stats = run(root, args.apply, logger)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **stats}))
    if args.apply:
        left = run(root, False, logger)
        print(json.dumps({"left_after_apply": left["edges_dropped"]}))
        return 1 if left["edges_dropped"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
