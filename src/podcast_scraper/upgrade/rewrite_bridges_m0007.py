"""Bring ``*.bridge.json`` back in step with the corpus after the #1685 backfill.

m0007 rewrites ``.gi.json`` and ``.kg.json`` only. ``*.bridge.json`` carries the SAME ``person:``
ids in its CIL identity list, and ``server/cil_queries.py`` walks every bridge at request time —
timeline, position-arc, conversation-arc. So immediately after the migration those surfaces serve
ids that no longer exist anywhere in the graph, and nothing errors. A dangling cross-reference
that renders as an empty result is exactly the silent-inconsistency class #1685 exists to remove,
so leaving it is not an option and neither is discovering it from a user report.

WHY NOT JUST RE-RUN ``build_bridge``

Because it would change more than this migration did. ``build_bridge`` defaults to
``fuzzy_reconcile=True`` at ``fuzzy_threshold=0.85``, merging single-layer identities by
display-name embedding similarity. Re-deriving 900+ bridges would re-decide every one of those
merges — with a different model state and a corpus that has moved — and any difference would be
indistinguishable from something m0007 did. It also costs an embedding pass.

This applies the SAME substitution m0007 applied, to the ids m0007 changed, and nothing else. The
diff is auditable and provably scoped to the migration.

WHY IT DOES NOT NEED m0007'S ID MAP

m0007 does not persist one, and after it has run the bare ids are gone from the graph, so the map
cannot be recovered from the artifacts. It does not have to be: scoping is deterministic —
``person:jensen`` in episode E always becomes ``person:unresolved-jensen-{E}`` — and the bridge
carries its own ``episode_id``. So the target is computable.

The safety comes from VERIFYING rather than trusting that: a rewrite is applied only when the
computed id actually exists in that episode's migrated ``.gi.json`` / ``.kg.json``. A bare id whose
scoped form is absent is left alone and reported, because that means the bridge and the graph
disagree for some reason this pass does not understand, and guessing would make it worse.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..identity.bare_name_scope import is_bare_person_id, scoped_person_id

_PERSON = "person:"


def _iter_bridges(root: Path) -> Iterable[Path]:
    """All ``*.bridge.json`` under *root* (recursive). Stable order."""
    return sorted(root.rglob("*.bridge.json"))


def _graph_person_ids(bridge_path: Path) -> Set[str]:
    """Every `person:` node id in the episode's MIGRATED gi/kg siblings.

    This is the set a bridge id is allowed to point at. Reading it per episode is what turns the
    rewrite from "compute the obvious target" into "confirm the target is really there".
    """
    stem = bridge_path.name[: -len(".bridge.json")]
    ids: Set[str] = set()
    for suffix in (".gi.json", ".kg.json"):
        sibling = bridge_path.with_name(stem + suffix)
        try:
            payload = json.loads(sibling.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for node in payload.get("nodes") or []:
            if isinstance(node, dict):
                nid = node.get("id")
                if isinstance(nid, str) and nid.startswith(_PERSON):
                    ids.add(nid)
    return ids


def plan_bridge(payload: dict, graph_ids: Set[str]) -> Tuple[Dict[str, str], List[str]]:
    """``({old: new}, [unresolved])`` for one bridge payload. Pure; no I/O."""
    episode_id = str(payload.get("episode_id") or "").strip()
    mapping: Dict[str, str] = {}
    unresolved: List[str] = []
    if not episode_id:
        return mapping, unresolved
    for entry in payload.get("identities") or []:
        if not isinstance(entry, dict):
            continue
        nid = entry.get("id")
        if not isinstance(nid, str) or not is_bare_person_id(nid):
            continue
        target = scoped_person_id(nid, episode_id)
        if target in graph_ids:
            mapping[nid] = target
        else:
            # The graph does not contain the id this SHOULD have become. Do not invent it.
            unresolved.append(nid)
    return mapping, unresolved


def apply_to_payload(payload: dict, mapping: Dict[str, str]) -> Tuple[dict, int]:
    """Substitute ids in ``identities`` and ``fuzzy_merges``. Returns ``(new, changes)``.

    Identities are keyed by id, so a substitution can collide with an entry that already exists —
    the same merge hazard `rewrite_ids` handles for the graph layers. Merge rather than emit two
    entries sharing one id, which would be a corrupt bridge.
    """
    if not mapping:
        return payload, 0
    out = copy.deepcopy(payload)
    changes = 0

    merged: Dict[str, dict] = {}
    order: List[str] = []
    for entry in out.get("identities") or []:
        if not isinstance(entry, dict):
            continue
        nid = entry.get("id")
        new_id = mapping.get(nid, nid) if isinstance(nid, str) else nid
        if isinstance(nid, str) and new_id != nid:
            changes += 1
            entry = {**entry, "id": new_id}
        key = str(new_id)
        if key in merged:
            prev = merged[key]
            aliases = sorted({*(prev.get("aliases") or []), *(entry.get("aliases") or [])})
            sources = {**(prev.get("sources") or {}), **(entry.get("sources") or {})}
            merged[key] = {**prev, "aliases": aliases, "sources": sources}
        else:
            merged[key] = entry
            order.append(key)
    if order:
        out["identities"] = [merged[k] for k in order]

    fuzzy = out.get("fuzzy_merges")
    if isinstance(fuzzy, list):
        for row in fuzzy:
            if not isinstance(row, dict):
                continue
            for key in ("gi_id", "kg_id"):
                val = row.get(key)
                if isinstance(val, str) and val in mapping:
                    row[key] = mapping[val]
                    changes += 1
    return out, changes


def _write_atomic(path: Path, payload: dict) -> None:
    """tmp + os.replace, so a crash cannot leave a truncated bridge."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(root: Path, *, dry_run: bool) -> Dict[str, object]:
    """Walk every bridge; rewrite the ones the migration invalidated."""
    scanned = changed = rewrites = unparsable = 0
    unresolved_total: List[str] = []
    for bridge in _iter_bridges(root):
        scanned += 1
        try:
            payload = json.loads(bridge.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unparsable += 1
            continue
        if not isinstance(payload, dict):
            unparsable += 1
            continue
        mapping, unresolved = plan_bridge(payload, _graph_person_ids(bridge))
        unresolved_total.extend(unresolved)
        if not mapping:
            continue
        new_payload, count = apply_to_payload(payload, mapping)
        if not count:
            continue
        changed += 1
        rewrites += count
        if not dry_run:
            _write_atomic(bridge, new_payload)
    return {
        "scanned": scanned,
        "changed": changed,
        "rewrites": rewrites,
        "unparsable": unparsable,
        "unresolved": sorted(set(unresolved_total)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Non-zero when the corpus is missing or the pass raises."""
    parser = argparse.ArgumentParser(
        prog="python -m podcast_scraper.upgrade.rewrite_bridges_m0007",
        description="Re-point *.bridge.json at the ids m0007 wrote.",
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("plan", "apply"), default="plan")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.corpus_root
    if not root.is_dir():
        print(f"ERROR: corpus root does not exist: {root}", file=sys.stderr)
        return 1

    dry_run = args.mode == "plan"
    print(f"bridge re-point — mode={args.mode} corpus={root}")
    if not dry_run:
        print("WRITING IN PLACE.")
    try:
        r = run(root, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 — surface it, never a tidy zero
        print(f"ERROR: bridge rewrite failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    verb = "would re-point" if dry_run else "re-pointed"
    print(
        f"{verb} {r['changed']} bridge(s), {r['rewrites']} id substitution(s); "
        f"{r['scanned']} scanned, {r['unparsable']} unparsable"
    )
    unresolved = r["unresolved"]
    if isinstance(unresolved, list) and unresolved:
        # Not fatal, but never silent: the bridge holds a bare id whose scoped form is absent
        # from that episode's graph, so the two disagree for a reason this pass cannot explain.
        print(
            f"WARNING: {len(unresolved)} bare id(s) left untouched — no scoped form in the "
            f"episode's graph: {', '.join(unresolved[:10])}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
