"""0036 — every published GI / KG person has its bridge identity (text only).

An episode's ``bridge.json`` joins its GI and KG persons; the cross-layer read surfaces read it
(``server/cil_queries.py``). On snapshot-prod-20261025, 115 persons in 45 episodes had no bridge
identity although GI or KG holds them: identities older rewrites dropped (most last held in 0025's
backups) and persons later rewrites added to GI / KG with no bridge entry.

The rule reads no judgement, so there is no frozen list: for each served episode, every GI / KG
Person id that no bridge identity carries (by its id or an alias), whose name the pipeline would
publish (``is_publishable_speaker_name`` with the episode's language), gets the identity the
pipeline's own builder makes (``bridge_builder._collect_identity`` over the GI node, then the KG
node). Unpublishable names are left out: those are junk labels 0015 removed from the bridge that GI
or KG kept, and restoring them would put the junk on the person surfaces. No fuzzy merge (it needs
an embedding model). Nothing but the bridge is written. UNDO restores from
``.podcast_scraper/upgrade-backups/0036/`` through ``bridge_identities_restored.jsonl``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...builders.bridge_builder import _collect_identity, _strip_layer_prefixes, build_bridge
from ...speaker_detectors.hosts import is_publishable_speaker_name
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0036_bridge_identities_restored"
RECEIPTS_FILE = "bridge_identities_restored.jsonl"
BACKUP_TAG = "0036"
#: ``emitted_at`` of a rebuilt bridge: fixed (the day this repair was specified), so a rebuild is
#: deterministic and a second apply or an undo cycle is byte-identical. The builder's default is
#: the wall clock.
REBUILT_EMITTED_AT = datetime(2026, 10, 26, tzinfo=timezone.utc)


def _json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _persons(doc: Any) -> Dict[str, dict]:
    """``{person id: node}`` of a GI / KG artifact's Person nodes, keyed by the id the builder
    gives the identity (layer prefixes such as ``g:`` / ``k:`` stripped). Persons only."""
    nodes = doc.get("nodes") if isinstance(doc, dict) else None
    out: Dict[str, dict] = {}
    for n in nodes or []:
        if isinstance(n, dict) and n.get("type") == "Person" and isinstance(n.get("id"), str):
            pid = _strip_layer_prefixes(n["id"])
            if pid.startswith("person:"):
                out[pid] = n
    return out


def _display(node: dict) -> str:
    """The name the builder displays: ``name``, else ``label`` (``_collect_identity``)."""
    props = node.get("properties") or {}
    for key in ("name", "label"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _language(meta: dict) -> Optional[str]:
    return (meta.get("feed") or {}).get("language") or (meta.get("episode") or {}).get("language")


def _carried(bridge: dict) -> set:
    out: set = set()
    for ident in bridge.get("identities") or []:
        if isinstance(ident, dict):
            out.add(ident.get("id"))
            out.update(a for a in ident.get("aliases") or [] if isinstance(a, str))
    return out


#: Why an episode was skipped, counted in the apply message (never a crash, never a write).
SKIP_REASONS = (
    "metadata unreadable",
    "no bridge or bridge unreadable",
    "bridge has no identities list",
    "empty bridge without a readable GI and KG",
)


Gap = Tuple[Optional[Path], Optional[dict], List[dict], List[str], bool]


def _publishable(name: str, lang: Optional[str]) -> bool:
    return bool(name) and is_publishable_speaker_name(name, language=lang)


def _rebuilt(meta: dict, gi_doc: Any, kg_doc: Any, lang: Optional[str]) -> Tuple[dict, List[str]]:
    """The bridge the builder makes from GI and KG (no fuzzy pass), unpublishable persons out."""
    eid = str((meta.get("episode") or {}).get("episode_id") or "")
    bridge = build_bridge(
        eid,
        gi_doc if isinstance(gi_doc, dict) else None,
        kg_doc if isinstance(kg_doc, dict) else None,
        emitted_at=REBUILT_EMITTED_AT,
        fuzzy_reconcile=False,
    )
    left_out = [
        i["id"]
        for i in bridge["identities"]
        if i.get("type") == "person" and not _publishable(str(i.get("display_name") or ""), lang)
    ]
    bridge["identities"] = [i for i in bridge["identities"] if i["id"] not in set(left_out)]
    return bridge, left_out


def episode_gaps(meta_path: Path, skipped: Optional[Dict[str, int]] = None) -> Gap:
    """``(bridge path, bridge to write, person identities added, unpublishable ids left out,
    rebuilt)``.

    An EMPTY bridge (``{}``: no ``identities`` key) is replaced whole by the builder's bridge;
    any other bridge gets the missing publishable persons appended. Persons follow the GI node
    when both layers hold them (read GI first, as the builder does)."""

    def skip(reason: str) -> Gap:
        if skipped is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
        return None, None, [], [], False

    meta = _json(meta_path)
    if not isinstance(meta, dict):
        return skip(SKIP_REASONS[0])
    stem = meta_path.name[: -len(".metadata.json")]
    bridge_path = meta_path.with_name(f"{stem}.bridge.json")
    bridge = _json(bridge_path)
    if not isinstance(bridge, dict):
        return skip(SKIP_REASONS[1])
    gi_doc = _json(meta_path.with_name(f"{stem}.gi.json"))
    kg_doc = _json(meta_path.with_name(f"{stem}.kg.json"))
    lang = _language(meta)
    # Damaged: no episode_id (`{}` from 0025, `{"identities": []}` from a later rewrite of one).
    if "episode_id" not in bridge:
        # Rebuilt only from both layers: a bridge rebuilt from one would silently lack the other's
        # persons and topics, and a later run would append only the persons back.
        if not isinstance(gi_doc, dict) or not isinstance(kg_doc, dict):
            return skip(SKIP_REASONS[3])
        new, out = _rebuilt(meta, gi_doc, kg_doc, lang)
        persons = [i for i in new["identities"] if i.get("type") == "person"]
        return bridge_path, new, persons, out, True
    if not isinstance(bridge.get("identities"), list):
        return skip(SKIP_REASONS[2])
    gi = _persons(gi_doc)
    kg = _persons(kg_doc)
    carried = _carried(bridge)
    add: List[dict] = []
    left_out: List[str] = []
    for pid in sorted((set(gi) | set(kg)) - carried):
        name = _display(gi[pid]) if pid in gi else _display(kg[pid])
        if not _publishable(name, lang):
            left_out.append(pid)
            continue
        built: Dict[str, Dict[str, Any]] = {}
        if pid in gi:
            _collect_identity(built, gi[pid], source="gi")
        if pid in kg:
            _collect_identity(built, kg[pid], source="kg")
        add.extend(built.values())
    return bridge_path, bridge, add, left_out, False


def _with(existing: List[Any], add: List[dict]) -> List[Any]:
    """``existing`` plus ``add``: in id order when ``existing`` was in id order (the builder's
    order), else appended; existing identities keep their relative order either way."""
    ids = [str(i.get("id")) for i in existing if isinstance(i, dict)]
    if len(ids) == len(existing) and ids == sorted(ids):
        return sorted(list(existing) + add, key=lambda i: str(i.get("id")))
    return list(existing) + add


def find_gaps(
    root: Path, skipped: Optional[Dict[str, int]] = None
) -> Iterable[Tuple[Path, Path, dict, List[dict], List[str], bool]]:
    """``(metadata, bridge path, bridge, to add, left out)`` per SERVED episode with a gap;
    superseded runs of an episode are not read (the served run is the one readers join)."""
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        bridge_path, bridge, add, left_out, rebuilt = episode_gaps(meta_path, skipped)
        if bridge_path is not None and bridge is not None and (add or left_out or rebuilt):
            yield meta_path, bridge_path, bridge, add, left_out, rebuilt


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each bridge this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class BridgeIdentitiesRestoredMigration(Migration):
    """A bridge identity for every published GI / KG person that has none."""

    id = MIGRATION_ID
    to_version = "2.7.30"
    description = (
        "every GI / KG person the pipeline would publish gets the bridge identity its own builder "
        "makes, where the bridge carries none (text only; unpublishable names left out)"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would restore and rebuild: pure read, no writes."""
        n = eps = rebuilt = 0
        for _m, _p, _b, add, _l, whole in find_gaps(ctx.corpus_root):
            if add or whole:
                n += len(add)
                eps += 1
                rebuilt += whole
        return (
            f"{self.id}: {n} person identit(ies) to restore in {eps} episode(s), "
            f"{rebuilt} empty bridge(s) rebuilt"
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write each bridge that lacks an identity or is damaged; back up each, receipt it."""
        root = Path(ctx.corpus_root)
        receipts: List[Dict[str, str]] = []
        added = left_out = rebuilt = rebuilt_identities = 0
        episodes: List[str] = []
        skipped: Dict[str, int] = {}
        for meta_path, bridge_path, bridge, add, out, whole in find_gaps(root, skipped):
            left_out += len(out)
            if not add and not whole:
                continue
            added += len(add)
            rebuilt += whole
            rebuilt_identities += len(bridge["identities"]) if whole else 0
            episodes.append(str(meta_path.relative_to(root)))
            if ctx.dry_run:
                continue
            if not whole:
                # Appended, as the shared rewrite adds identities: existing ones keep their place.
                bridge["identities"] = _with(bridge["identities"], add)
            receipts.append(write_with_backup(root, BACKUP_TAG, bridge_path, bridge))
        if not ctx.dry_run:
            append_receipts(root, RECEIPTS_FILE, {"migration": self.id}, receipts)
        message = (
            f"restored {added} bridge identit(ies) in {len(episodes)} episode(s); "
            f"identities_added={added}, bridges_rebuilt={rebuilt}, "
            f"rebuilt_identities={rebuilt_identities}, unpublishable_left_out={left_out}"
            + "".join(f", skipped[{k}]={v}" for k, v in sorted(skipped.items()))
        )
        return MigrationResult(
            migration_id=self.id,
            applied=not ctx.dry_run,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "episodes": episodes,
                "identities_added": added,
                "bridges_rebuilt": rebuilt,
                "rebuilt_identities": rebuilt_identities,
                "left_out": left_out,
                "skipped": skipped,
            },
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No published GI / KG person left without a bridge identity, and no damaged bridge."""
        gaps = list(find_gaps(ctx.corpus_root))
        missing = sum(len(add) for _m, _p, _b, add, _l, _w in gaps)
        empty = sum(1 for g in gaps if g[5])
        if missing or empty:
            return False, (
                f"{missing} published GI / KG person(s) still lack a bridge identity; "
                f"{empty} empty bridge(s)"
            )
        return True, "every published GI / KG person has a bridge identity"


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write, with its counters and input
    sha256s (the outcome hook's plan diff)."""
    from ..read_recorder import recording

    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        with recording(Path(root)) as rec:
            bridge_path, bridge, add, _out, whole = episode_gaps(meta_path)
        if bridge_path is None or not (add or whole):
            continue
        counts = {"identities_added": len(add)}
        if whole and bridge is not None:
            counts["rebuilt_identities"] = len(bridge["identities"])
        yield {
            "meta": str(meta_path.relative_to(root)),
            "counts": counts,
            "inputs": rec.inputs(),
        }
