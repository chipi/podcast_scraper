"""0012 — an organisation the corpus itself calls an organisation is nobody's voice (#2220).

KG extraction labels every entity Person / Organization, and the speaker path never read it, so
names the corpus overwhelmingly calls an organisation were seated on voices. Measured on prod
(2,002 served episodes, 2026-10-01): 144 roster host/guest entries, 16 names — Andreessen Horowitz
53, The Brazilian Report 40, Americas Online 31, Carnegie India 8, then one-off guests (White House,
World Bank, Supreme Court, Apple, …) — credited on 2,087 insight links, 869 of them surfaced.

The pipeline now refuses these names at candidate time and at both readers of the stored segment
label (#2220). This removes what is already on disk — no LLM, no reprocess.

WHAT EACH SURFACE BECOMES — what the pipeline writes for a voice it failed to name, on all five, or
on none (a half-rewritten episode would disagree with itself):

* ``.metadata.json`` — the ``content.speakers`` entry is dropped (an org is not "a person a source
  named", so not ``placed: false`` either) and the name leaves ``detected_hosts/guests``.
* ``.segments.json`` + ``.adfree.segments.json`` — ``speaker_label`` is removed (the raw
  ``SPEAKER_NN`` stays: a real, unnamed voice) and ``voice_type`` becomes ``unknown`` (a person we
  FAILED to name — it is naming work, and ``scripts/audit/unbound_name_causes.py`` counts it so).
  These are the durable record every re-derive reads; leaving them re-poisons the repair.
* ``.kg.json`` — the host/guest Person node and every edge touching it are DELETED. Not demoted to
  ``mentioned``: that node would then vote Person and flip the very verdict this applies (54 : 0
  becomes 54 : 53). Extraction's own ``mentioned`` nodes are left alone — they are the evidence.
* ``.gi.json`` — the Person node and its ``SPOKEN_BY`` edges go; quotes lose ``speaker_id`` and
  carry ``speaker_voice_type: unknown``; insights it was credited with lose ``speaker``, become
  ``surfaceable: false`` and get ``routing_tag`` / ``salience`` recomputed by the pipeline's own
  ``_apply_route_and_tag``. A wrong name is worse than no name (#876).
* ``.bridge.json`` — the identity row is removed, or CIL serves a dangling person.

THE SET IS FROZEN. The organisation names are decided once, at apply time, from the served KGs, and
written to the receipts. ``verify`` judges against that set, never live votes, which drift as the
corpus grows.

UNDO. Every file is copied, as it was, under ``.podcast_scraper/upgrade-backups/0012/`` before it is
written; ``undo`` restores a file only while it is still exactly what this migration wrote.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...speaker_detectors.entity_kind_votes import kind_key, votes_from_kg_payloads
from ..corpus_selection import select_served_artifacts
from ..migration import Migration, MigrationContext, MigrationResult
from ..role_ledger import file_sha

MIGRATION_ID = "0012_org_speakers_removed"
RECEIPTS_FILE = "org_speakers_removed.jsonl"
BACKUP_DIR = Path(".podcast_scraper") / "upgrade-backups" / "0012"
_SPEAKING = ("host", "guest")


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _sibling(meta: Path, suffix: str) -> Path:
    return meta.with_name(meta.name[: -len(".metadata.json")] + suffix)


def _segment_files(meta: Path, content: dict) -> List[Path]:
    rel = str((content or {}).get("transcript_file_path") or "")
    if not rel:
        return []
    run_root = meta.parent.parent
    out = []
    for suffix in (".segments.json", ".adfree.segments.json"):
        candidate = run_root / rel.replace(".txt", suffix)
        if candidate.is_file():
            out.append(candidate)
    return out


def org_names(root: Path) -> Dict[str, Tuple[int, int]]:
    """``{kind_key: (org, person)}`` votes for each roster host/guest the votes call an org."""
    served_kg, _ = select_served_artifacts(root, ".kg.json")
    votes = votes_from_kg_payloads(p for p in (_load(k) for k in served_kg) if isinstance(p, dict))
    out: Dict[str, Tuple[int, int]] = {}
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        payload = _load(meta)
        if not isinstance(payload, dict):
            continue
        for entry in (payload.get("content") or {}).get("speakers") or []:
            name = (entry or {}).get("name")
            if (entry or {}).get("role") in _SPEAKING and name and votes.calls_organisation(name):
                out[kind_key(name)] = votes.counts[kind_key(name)]
    return out


class _Episode:
    """One episode's five surfaces, rewritten in memory; nothing touches disk here."""

    def __init__(self, meta: Path, orgs: Set[str]) -> None:
        self.meta = meta
        self.orgs = orgs
        self.files: Dict[Path, Any] = {}
        self.changed: Set[Path] = set()
        self.counts: Dict[str, int] = {}
        self.person_ids: Set[str] = set()

    def _bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def _is_org(self, name: Any) -> bool:
        return isinstance(name, str) and bool(name.strip()) and kind_key(name) in self.orgs

    def plan(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        self.files[self.meta] = meta
        content = meta.setdefault("content", {})
        speakers = content.get("speakers") or []
        kept = [s for s in speakers if not self._is_org((s or {}).get("name"))]
        if len(kept) != len(speakers):
            content["speakers"] = kept
            self._bump("roster_entries", len(speakers) - len(kept))
            self.changed.add(self.meta)
        for field in ("detected_hosts", "detected_guests"):
            names = content.get(field)
            if isinstance(names, list):
                left = [n for n in names if not self._is_org(n)]
                if len(left) != len(names):
                    content[field] = left
                    self.changed.add(self.meta)

        for seg_path in _segment_files(self.meta, content):
            raw = _load(seg_path)
            rows = raw if isinstance(raw, list) else (raw or {}).get("segments")
            if not isinstance(rows, list):
                continue
            hit = 0
            for row in rows:
                if isinstance(row, dict) and self._is_org(row.get("speaker_label")):
                    row.pop("speaker_label", None)
                    row["voice_type"] = "unknown"
                    hit += 1
            if hit:
                self.files[seg_path] = raw
                self.changed.add(seg_path)
                self._bump("segments_relabelled", hit)

        kg_path = _sibling(self.meta, ".kg.json")
        kg = _load(kg_path)
        if isinstance(kg, dict):
            gone = {
                str(n.get("id"))
                for n in kg.get("nodes") or []
                if isinstance(n, dict)
                and n.get("type") == "Person"
                and (n.get("properties") or {}).get("role") in _SPEAKING
                and self._is_org((n.get("properties") or {}).get("name"))
            }
            if gone:
                kg["nodes"] = [n for n in kg.get("nodes") or [] if str(n.get("id")) not in gone]
                edges = kg.get("edges") or []
                kg["edges"] = [
                    e for e in edges if e.get("from") not in gone and e.get("to") not in gone
                ]
                self._bump("kg_nodes_deleted", len(gone))
                self._bump("kg_edges_deleted", len(edges) - len(kg["edges"]))
                self.files[kg_path] = kg
                self.changed.add(kg_path)
                self.person_ids |= gone

        gi_path = _sibling(self.meta, ".gi.json")
        gi = _load(gi_path)
        if isinstance(gi, dict) and self._rewrite_gi(gi):
            self.files[gi_path] = gi
            self.changed.add(gi_path)

        bridge_path = _sibling(self.meta, ".bridge.json")
        bridge = _load(bridge_path)
        if isinstance(bridge, dict):
            ids = bridge.get("identities") or []
            left = [
                i
                for i in ids
                if not (
                    str(i.get("id") or "").startswith("person:")
                    and (str(i.get("id")) in self.person_ids or self._is_org(i.get("display_name")))
                )
            ]
            if len(left) != len(ids):
                bridge["identities"] = left
                self._bump("bridge_identities_removed", len(ids) - len(left))
                self.files[bridge_path] = bridge
                self.changed.add(bridge_path)
        return bool(self.changed)

    def _rewrite_gi(self, gi: dict) -> bool:
        from ...gi.pipeline import _apply_route_and_tag

        nodes = gi.get("nodes") or []
        gone = {
            str(n.get("id"))
            for n in nodes
            if isinstance(n, dict)
            and n.get("type") == "Person"
            and self._is_org((n.get("properties") or {}).get("name"))
        }
        if not gone:
            return False
        self.person_ids |= gone
        edges = gi.get("edges") or []
        spoken = {
            e.get("from") for e in edges if e.get("type") == "SPOKEN_BY" and e.get("to") in gone
        }
        gi["edges"] = [e for e in edges if e.get("from") not in gone and e.get("to") not in gone]
        supported: Set[str] = {
            str(e.get("from"))
            for e in gi["edges"]
            if e.get("type") == "SUPPORTED_BY" and e.get("to") in spoken
        }
        kept_nodes = []
        for node in nodes:
            nid = str(node.get("id"))
            if nid in gone:
                continue
            props = node.setdefault("properties", {})
            if node.get("type") == "Quote" and (nid in spoken or props.get("speaker_id") in gone):
                props["speaker_id"] = None
                props["speaker_voice_type"] = "unknown"
                props.pop("speaker_name", None)
                self._bump("quotes_unattributed")
            elif node.get("type") == "Insight" and (
                self._is_org(props.get("speaker"))
                or (not props.get("speaker") and nid in supported)
            ):
                was_surface = props.get("routing_tag") == "surface"
                props.pop("speaker", None)
                props["speaker_voice_type"] = "unknown"
                props["surfaceable"] = False
                if isinstance(props.get("tier"), int):
                    _apply_route_and_tag(props, props["tier"])
                self._bump("insights_unattributed")
                if was_surface:
                    self._bump("insights_surface_to_connect")
            kept_nodes.append(node)
        gi["nodes"] = kept_nodes
        self._bump("gi_persons_deleted", len(gone))
        return True


def _receipts_path(root: Path) -> Path:
    return root / RECEIPTS_FILE


def _read_receipts(root: Path) -> Tuple[Dict[str, Any], List[dict]]:
    """``(header, rows)``; the header carries the frozen org set."""
    header: Dict[str, Any] = {}
    rows: List[dict] = []
    try:
        lines = _receipts_path(root).read_text(encoding="utf-8").splitlines()
    except OSError:
        return header, rows
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "header":
            header = row
        else:
            rows.append(row)
    return header, rows


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    root = Path(root)
    _header, rows = _read_receipts(root)
    restored, refused = 0, []
    for row in rows:
        target = root / row["relpath"]
        backup = root / BACKUP_DIR / row["relpath"]
        if file_sha(target) != row["sha_after"]:
            refused.append(f"{row['relpath']}: changed since this migration wrote it")
            continue
        if not backup.is_file():
            refused.append(f"{row['relpath']}: no backup")
            continue
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(backup, tmp)
        os.replace(tmp, target)
        restored += 1
    if restored:
        try:
            from ..state import FilesystemStateStore

            FilesystemStateStore(root).record_reverted(MIGRATION_ID)
        except Exception:  # noqa: BLE001 — the files are restored; never undo the undo
            pass
    return restored, refused


class OrgSpeakersRemovedMigration(Migration):
    """Remove names the corpus's extraction calls organisations from every speaker surface."""

    id = MIGRATION_ID
    to_version = "2.7.6"
    description = (
        "#2220: a name the corpus's KG extraction decisively calls an organisation (>=3 votes, "
        ">=4x Person) is nobody's voice. Remove it from the roster, segment labels, KG host/guest "
        "nodes, GI Person/SPOKEN_BY/quote/insight attribution and bridge identities"
    )

    def _episodes(self, root: Path, orgs: Set[str]) -> Iterable[_Episode]:
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, orgs)
            if ep.plan():
                yield ep

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        orgs = org_names(ctx.corpus_root)
        if not orgs:
            return "no roster speaker is an organisation by corpus vote — nothing to remove"
        totals: Dict[str, int] = {}
        episodes = 0
        for ep in self._episodes(ctx.corpus_root, set(orgs)):
            episodes += 1
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return (
            f"org speakers plan: {len(orgs)} name(s) {sorted(orgs)}; {episodes} episode(s); "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface still names a FROZEN org as a speaker. ``(ok, message)``."""
        header, _rows = _read_receipts(ctx.corpus_root)
        orgs = set(header.get("orgs") or {})
        if not orgs:
            return True, "no receipts — nothing was removed, nothing to verify"
        left = [
            f"{ep.meta.name}: {sorted(ep.counts)}" for ep in self._episodes(ctx.corpus_root, orgs)
        ]
        if left:
            return False, f"{len(left)} episode(s) still name an org as a speaker: {left[:5]}"
        return True, f"no served surface names any of {len(orgs)} org(s) as a speaker"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        root = ctx.corpus_root
        orgs = org_names(root)
        totals: Dict[str, int] = {}
        touched: List[str] = []
        receipts: List[dict] = []
        for ep in self._episodes(root, set(orgs)):
            touched.append(str(ep.meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if ctx.dry_run:
                continue
            for path in sorted(ep.changed):
                rel = str(path.relative_to(root))
                backup = root / BACKUP_DIR / rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copyfile(path, backup)
                sha_before = file_sha(path)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(_dump(ep.files[path]), encoding="utf-8")
                os.replace(tmp, path)
                receipts.append(
                    {"relpath": rel, "sha_before": sha_before, "sha_after": file_sha(path)}
                )
        if not ctx.dry_run and receipts:
            with _receipts_path(root).open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"kind": "header", "orgs": {k: list(v) for k, v in orgs.items()}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for row in receipts:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        message = f"{verb} {len(touched)} episode(s) for {len(orgs)} org name(s): " + ", ".join(
            f"{k}={v}" for k, v in sorted(totals.items())
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "orgs": {k: list(v) for k, v in sorted(orgs.items())},
                "episodes": len(touched),
                "totals": totals,
                "files_written": len(receipts),
                "episodes_sample": touched[:20],
            },
        )
