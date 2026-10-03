"""0017 — a published speaker name renamed to the clean form today's pipeline publishes.

Two defects, one mechanism (a rename whose id moves with the name, as m0010 does):

* PREFIX REPAIRS — "Your Host Luisa Leni", "Deputy Editor Eilish Hart", "Planet Money's Kenny
  Malone", "Celestin Ntawirema CEO": the job or the show was kept with the person. The roster now
  keeps the person (``strip_role_prefix`` on a placed name, ``_clean_stated_name`` on a stated
  one); this repairs what is on disk. The DISPLAY name changes, and so does the id (ids derive from
  the name).
* TITLED SPLIT IDENTITIES — "Professor Hannah Fry" minted ``person:professor-hannah-fry`` beside
  ``person:hannah-fry``. ``identity.slugify.person_identity_name`` now drops a leading title from
  the ID only. The DISPLAY name is unchanged ("Dr. Adam Rodman" is published as stated, the
  roster's golden fixture pins it), so only the id merges or is re-minted.

THE RENAME MAP IS NOT A LIST. ``display_target`` is m0015's ``rename_target`` (leading role words,
then ``hosts._clean_stated_name``, gated by ``is_publishable_speaker_name``) plus
``canonical_person_name``, and ids come from ``identity.slugify.person_id``. One deliberate
exception: ``_clean_stated_name`` also strips a leading honorific, which is right for a name
STATED in prose and wrong for the published display, so a change that is nothing but a title is
not a rename here (that is C5's id-only half).

SCOPE. A display rename applies only to names published AS A SPEAKER (roster, ``detected_*``,
segment labels, KG host/guest nodes, GI ``SPOKEN_BY`` targets / quote credits, the bridge rows of
those ids) and only when the result keeps two or more words: ``rename_target`` is a stated-speaker
cleaner, and on a KG ``mentioned`` entity it turns "Moore's Law" into "Law" and "Cosimo de' Medici"
into "Medici". The title-only id merge/re-mint (``title_only``) may apply to any Person node.

THE ID RULE IS m0010's: a node is re-minted only when its id demonstrably came from its own name
(``slugify(name)`` or ``slugify(canonical_person_name(name))``) and ``person_id`` of the target name
differs. ``person:joe-weisenthal`` carrying the name "Joe" is left alone. Episode-scoped ids are
left alone (0007).

A rename whose target id already exists in the same episode is a MERGE: ``rewrite_ids`` (as in
m0010) re-points edges and quote ``speaker_id`` and applies the role precedence; the EXISTING node
is placed first so its properties survive; edges that become identical are collapsed.

FIVE SURFACES, all or none per episode (as m0012): metadata ``content.speakers`` +
``detected_hosts/guests``; ``.segments.json`` + ``.adfree.segments.json`` ``speaker_label``;
``.kg.json`` Person id/name + edges; ``.gi.json`` Person id/name, ``SPOKEN_BY``, quote
``speaker_id``/``speaker_name``, insight ``speaker``; ``.bridge.json`` identity rows. A rename is
not a removal, so insights keep their surfaceability.

THE MAPS ARE FROZEN at apply time into the receipt header and ``verify`` judges against them. Before
any apply there is no header, so verify computes the live map and fails while it is non-empty.

NOT TOUCHED: enrichment files. ``details.person_ids_changed`` lists the old -> new ids so
person_web can be targeted; it drops an old id's bio on its next run and fetches the new ids.

UNDO: every file is copied under ``.podcast_scraper/upgrade-backups/0017/`` before it is written;
``undo`` restores a file only while it is still exactly what this migration wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ...identity.bare_name_scope import is_scoped_person_id, rewrite_ids
from ...identity.slugify import canonical_person_name, person_id, person_identity_name, slugify
from ...speaker_detectors.hosts import HONORIFIC_TITLES, is_publishable_speaker_name
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, read_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from ..rewrite_bridges_m0007 import apply_to_payload as _rewrite_bridge_ids
from .m0015_unpublishable_speaker_names_removed import rename_target as _stated_rename_target

MIGRATION_ID = "0017_speaker_names_canonicalised"
RECEIPTS_FILE = "speaker_names_canonicalised.jsonl"
BACKUP_TAG = "0017"
_PERSON = "person:"


def _without_leading_honorifics(name: str) -> str:
    toks = name.split()
    while len(toks) > 2 and toks[0].lower().strip(".,") in HONORIFIC_TITLES:
        toks = toks[1:]
    return " ".join(toks)


def display_target(name: Any) -> Optional[str]:
    """The clean form the pipeline publishes for *name*, or ``None`` when it publishes it as is.

    m0015's ``rename_target`` (shared, so "removed" and "renamed" never overlap) minus one case: it
    strips a leading title, right for a name STATED in prose, wrong for the display name. A change
    that is nothing but a title is not a rename — the title stays and only the id forgets it
    (``identity.slugify.person_identity_name``).
    """
    if not isinstance(name, str) or not name.strip():
        return None
    clean = _stated_rename_target(name)
    if clean is None or clean == _without_leading_honorifics(name):
        return None
    target = canonical_person_name(clean) or clean
    # A person keeps a first and a last name. "Cosimo de' Medici" -> "Medici" is the stated-name
    # cleaner mistaking "de'" for a possessive; a name reduced to one word is never a repair.
    if len(target.split()) < 2:
        return None
    return None if target == name else target


def title_only(name: Any) -> bool:
    """The id of *name* differs from its canonical spelling ONLY by a leading title.

    ``person_identity_name`` already drops the title; this also demands the remainder be two or more
    words and publishable, so "Cosimo de' Medici" and "Elon Musk's mother" can never qualify.
    """
    if not isinstance(name, str):
        return False
    full = (canonical_person_name(name) or "").split()
    rest = person_identity_name(name).split()
    return (
        len(rest) >= 2
        and len(rest) < len(full)
        and full[len(full) - len(rest) :] == rest
        and is_publishable_speaker_name(" ".join(rest))
    )


def _is_renamed(value: Any, names: Dict[str, str]) -> bool:
    return isinstance(value, str) and value in names


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def _segment_rows(raw: Any) -> List[Any]:
    rows = raw if isinstance(raw, list) else (raw or {}).get("segments") if raw else None
    return rows if isinstance(rows, list) else []


def _person_nodes(payload: Any) -> List[dict]:
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        return []
    return [
        n
        for n in nodes
        if isinstance(n, dict) and isinstance(n.get("id"), str) and n["id"].startswith(_PERSON)
    ]


def _survivor_first(items: List[Any], id_map: Dict[str, str]) -> List[Any]:
    """Put each item about to be renamed onto an id that already exists right after that item.

    ``rewrite_ids`` and the bridge merge keep the FIRST item's properties; the existing one is the
    survivor, so it has to be first.
    """
    present = {x.get("id") for x in items if isinstance(x, dict)}
    movers = [
        x
        for x in items
        if isinstance(x, dict)
        and x.get("id") in id_map
        and id_map[x["id"]] != x["id"]
        and id_map[x["id"]] in present
    ]
    if not movers:
        return items
    out = [x for x in items if not any(x is m for m in movers)]
    for mover in movers:
        target = id_map[mover["id"]]
        at = max(i for i, x in enumerate(out) if isinstance(x, dict) and x.get("id") == target)
        out.insert(at + 1, mover)
    return out


class _Episode:
    """One episode's five surfaces; rewritten in memory, nothing touches disk here."""

    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.files: Dict[Path, Any] = {}
        self._before: Dict[Path, str] = {}
        self.counts: Dict[str, int] = {}
        self.person_ids: Set[str] = set()

    def _bump(self, key: str, n: int = 1) -> None:
        if n:
            self.counts[key] = self.counts.get(key, 0) + n

    def load(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        self.files[self.meta] = meta
        content = meta.get("content") or {}
        paths = _segment_files(self.meta, content)
        paths += [_sibling(self.meta, s) for s in (".kg.json", ".gi.json", ".bridge.json")]
        for path in paths:
            payload = _load(path)
            if isinstance(payload, (dict, list)):
                self.files[path] = payload
        self._before = {p: json.dumps(v, sort_keys=True) for p, v in self.files.items()}
        return True

    @property
    def changed(self) -> Set[Path]:
        return {
            p for p, v in self.files.items() if json.dumps(v, sort_keys=True) != self._before[p]
        }

    def speaker_ids(self) -> Set[str]:
        """Person ids published as a speaker: a KG host/guest, or what a GI quote is credited to.

        Never a KG ``mentioned`` entity: a name's display is only cleaned where it is a voice.
        """
        out: Set[str] = set()
        for path, payload in self.files.items():
            if not isinstance(payload, dict):
                continue
            if path.name.endswith(".kg.json"):
                for node in _person_nodes(payload):
                    if (node.get("properties") or {}).get("role") in ("host", "guest"):
                        out.add(node["id"])
            elif path.name.endswith(".gi.json"):
                for edge in payload.get("edges") or []:
                    if isinstance(edge, dict) and edge.get("type") == "SPOKEN_BY":
                        if isinstance(edge.get("to"), str):
                            out.add(edge["to"])
                for node in payload.get("nodes") or []:
                    props = node.get("properties") if isinstance(node, dict) else None
                    if isinstance(props, dict) and isinstance(props.get("speaker_id"), str):
                        out.add(props["speaker_id"])
        return {i for i in out if i.startswith(_PERSON)}

    def names(self) -> Iterator[str]:
        """Every name published AS A SPEAKER on the five surfaces."""
        sids = self.speaker_ids()
        content = (self.files[self.meta].get("content")) or {}
        for entry in content.get("speakers") or []:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                yield entry["name"]
        for field in ("detected_hosts", "detected_guests"):
            for n in content.get(field) or []:
                if isinstance(n, str):
                    yield n
        for path, payload in self.files.items():
            if path.name.endswith(".segments.json"):
                for row in _segment_rows(payload):
                    if isinstance(row, dict) and isinstance(row.get("speaker_label"), str):
                        yield row["speaker_label"]
            elif path.name.endswith((".kg.json", ".gi.json")):
                for node in (payload.get("nodes") or []) if isinstance(payload, dict) else []:
                    props = node.get("properties") if isinstance(node, dict) else None
                    if not isinstance(props, dict):
                        continue
                    for key in ("speaker_name", "speaker"):
                        if isinstance(props.get(key), str):
                            yield props[key]
                    if node.get("id") in sids and isinstance(props.get("name"), str):
                        yield props["name"]
            elif path.name.endswith(".bridge.json") and isinstance(payload, dict):
                for ident in payload.get("identities") or []:
                    if (
                        isinstance(ident, dict)
                        and ident.get("id") in sids
                        and isinstance(ident.get("display_name"), str)
                    ):
                        yield ident["display_name"]

    def person_pairs(self) -> Iterator[Tuple[str, str, bool]]:
        """``(person id, name, published as a speaker)`` for every Person node and bridge row."""
        sids = self.speaker_ids()
        for path, payload in self.files.items():
            if path.name.endswith((".kg.json", ".gi.json")):
                for node in _person_nodes(payload):
                    name = (node.get("properties") or {}).get("name")
                    if isinstance(name, str):
                        yield node["id"], name, node["id"] in sids
            elif path.name.endswith(".bridge.json") and isinstance(payload, dict):
                for ident in payload.get("identities") or []:
                    if not isinstance(ident, dict):
                        continue
                    nid, name = ident.get("id"), ident.get("display_name")
                    if isinstance(nid, str) and nid.startswith(_PERSON) and isinstance(name, str):
                        yield nid, name, nid in sids

    def rewrite(self, names: Dict[str, str], ids: Dict[str, str]) -> None:
        sids = self.speaker_ids()
        self._rewrite_meta(names)
        for path in list(self.files):
            if path.name.endswith(".segments.json"):
                self._rewrite_segments(path, names)
            elif path.name.endswith((".kg.json", ".gi.json")):
                self._rewrite_graph(path, names, ids, sids)
            elif path.name.endswith(".bridge.json"):
                self._rewrite_bridge(path, names, ids, sids)

    def _rewrite_meta(self, names: Dict[str, str]) -> None:
        content = self.files[self.meta].get("content") or {}
        for entry in content.get("speakers") or []:
            if isinstance(entry, dict) and _is_renamed(entry.get("name"), names):
                entry["name"] = names[entry["name"]]
                self._bump("roster_names")
        for field in ("detected_hosts", "detected_guests"):
            listed = content.get(field)
            if isinstance(listed, list):
                content[field] = [names.get(n, n) if isinstance(n, str) else n for n in listed]
                self._bump(
                    "detected_names", sum(1 for n in listed if isinstance(n, str) and n in names)
                )

    def _rewrite_segments(self, path: Path, names: Dict[str, str]) -> None:
        for row in _segment_rows(self.files[path]):
            if isinstance(row, dict) and _is_renamed(row.get("speaker_label"), names):
                row["speaker_label"] = names[row["speaker_label"]]
                self._bump("segments_relabelled")

    def _rewrite_graph(
        self, path: Path, names: Dict[str, str], ids: Dict[str, str], sids: Set[str]
    ) -> None:
        payload = self.files[path]
        if not isinstance(payload, dict):
            return
        layer = "kg" if path.name.endswith(".kg.json") else "gi"
        present = {n["id"] for n in _person_nodes(payload)}
        self.person_ids |= present & set(ids)
        for node in payload.get("nodes") or []:
            props = node.get("properties") if isinstance(node, dict) else None
            if not isinstance(props, dict):
                continue
            keys = ["speaker_name", "speaker"]
            if node.get("id") in sids:
                keys.append("name")
            for key in keys:
                if _is_renamed(props.get(key), names):
                    props[key] = names[props[key]]
                    self._bump(f"{layer}_names_rewritten")
        if isinstance(payload.get("nodes"), list):
            payload["nodes"] = _survivor_first(payload["nodes"], ids)
        before_nodes = len(_person_nodes(payload))
        out, changes = rewrite_ids(payload, ids)
        self._bump(f"{layer}_ids_rewritten", changes)
        self._bump(f"{layer}_person_nodes_merged", before_nodes - len(_person_nodes(out)))
        touched = set(ids.values())
        seen: List[str] = []
        edges: List[Any] = []
        for edge in out.get("edges") or []:
            if isinstance(edge, dict) and (
                edge.get("from") in touched or edge.get("to") in touched
            ):
                key = json.dumps(edge, sort_keys=True)
                if key in seen:
                    self._bump(f"{layer}_edges_deduped")
                    continue
                seen.append(key)
            edges.append(edge)
        if "edges" in out:
            out["edges"] = edges
        payload.clear()
        payload.update(out)

    def _rewrite_bridge(
        self, path: Path, names: Dict[str, str], ids: Dict[str, str], sids: Set[str]
    ) -> None:
        bridge = self.files[path]
        if not isinstance(bridge, dict):
            return
        identities = bridge.get("identities")
        if isinstance(identities, list):
            bridge["identities"] = _survivor_first(identities, ids)
        before = len(bridge.get("identities") or [])
        for ident in bridge.get("identities") or []:
            if not isinstance(ident, dict) or ident.get("id") not in sids:
                continue
            if _is_renamed(ident.get("display_name"), names):
                ident["display_name"] = names[ident["display_name"]]
                self._bump("bridge_names_rewritten")
            if isinstance(ident.get("aliases"), list):
                ident["aliases"] = [
                    names.get(a, a) if isinstance(a, str) else a for a in ident["aliases"]
                ]
        # The shared bridge merge overwrites `sources` with the later entry's; two identities for
        # one person seen in different layers must OR, or the survivor loses a layer.
        seen_in: Dict[str, Dict[str, bool]] = {}
        for ident in bridge.get("identities") or []:
            if isinstance(ident, dict) and isinstance(ident.get("sources"), dict):
                old = str(ident.get("id"))
                acc = seen_in.setdefault(ids.get(old, old), {})
                for layer, on in ident["sources"].items():
                    acc[layer] = acc.get(layer, False) or bool(on)
        out, _changes = _rewrite_bridge_ids(bridge, ids)
        for ident in out.get("identities") or []:
            if isinstance(ident, dict) and str(ident.get("id")) in seen_in:
                ident["sources"] = seen_in[str(ident["id"])]
        self._bump("bridge_identities_merged", before - len(out.get("identities") or []))
        bridge.clear()
        bridge.update(out)


def plan_maps(root: Path) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """``(names, ids, conflicts)`` over the served corpus: what to rename, and what each id becomes.

    ``names`` is ``{published name: clean name}``; ``ids`` is ``{old person id: new person id}``.
    An old id that two names would send to different targets is dropped and reported.
    """
    names: Dict[str, str] = {}
    ids: Dict[str, str] = {}
    conflicts: Set[str] = set()
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta)
        if not ep.load():
            continue
        for name in ep.names():
            if name not in names:
                target = display_target(name)
                if target:
                    names[name] = target
        for nid, name, is_speaker in ep.person_pairs():
            if is_scoped_person_id(nid):
                continue
            target = display_target(name) if is_speaker else None
            if target is None and not title_only(name):
                continue
            try:
                minted = {
                    f"{_PERSON}{slugify(name)}",
                    f"{_PERSON}{slugify(canonical_person_name(name))}",
                }
                want = person_id(target or name)
            except ValueError:
                continue
            if nid not in minted or want == nid:
                continue
            if ids.setdefault(nid, want) != want:
                conflicts.add(nid)
    for nid in conflicts:
        ids.pop(nid, None)
    return names, ids, sorted(conflicts)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class SpeakerNamesCanonicalisedMigration(Migration):
    """Rename published speaker names to their clean form, and move their person ids with them."""

    id = MIGRATION_ID
    to_version = "2.7.11"
    description = (
        "Published speaker names renamed to the clean form the pipeline now publishes: a job or "
        "show prefix dropped (Your Host X, Planet Money's X) and a titled id (person:professor-x) "
        "merged into or re-minted as the untitled one, across metadata, segments, KG, GI and "
        "bridge"
    )

    def _episodes(
        self, root: Path, names: Dict[str, str], ids: Dict[str, str]
    ) -> Iterable[_Episode]:
        if not names and not ids:
            return
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta)
            if not ep.load():
                continue
            ep.rewrite(names, ids)
            if ep.changed:
                yield ep

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        names, ids, conflicts = plan_maps(ctx.corpus_root)
        if not names and not ids:
            return "no published speaker name or person id needs renaming"
        episodes = sum(1 for _ in self._episodes(ctx.corpus_root, names, ids))
        return (
            f"speaker names plan: {len(names)} name(s) renamed, {len(ids)} person id(s) moved, "
            f"{episodes} episode(s), {len(conflicts)} conflicting id(s) skipped"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface still carries a frozen old name or id. ``(ok, message)``.

        Before any apply there is no frozen map, so the live one is computed: a corpus with
        something left to rename does not verify.
        """
        header, _rows = read_receipts(ctx.corpus_root, RECEIPTS_FILE)
        if header:
            names, ids = dict(header.get("names") or {}), dict(header.get("ids") or {})
        else:
            names, ids, _conflicts = plan_maps(ctx.corpus_root)
        left = [ep.meta.name for ep in self._episodes(ctx.corpus_root, names, ids)]
        if left:
            return False, f"{len(left)} episode(s) still carry an old name or id: {left[:5]}"
        return True, f"no served surface carries any of {len(names)} name(s) / {len(ids)} id(s)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Freeze the maps, then rewrite each episode's surfaces together; back up and receipt."""
        root = ctx.corpus_root
        names, ids, conflicts = plan_maps(root)
        totals: Dict[str, int] = {}
        touched: List[str] = []
        receipts: List[dict] = []
        for ep in self._episodes(root, names, ids):
            touched.append(str(ep.meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if ctx.dry_run:
                continue
            for path in sorted(ep.changed):
                receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
        if not ctx.dry_run:
            append_receipts(
                root,
                RECEIPTS_FILE,
                {"names": names, "ids": ids, "conflicts": conflicts},
                receipts,
            )
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        message = (
            f"{verb} {len(touched)} episode(s): {len(names)} name(s), {len(ids)} id(s); "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "names": dict(sorted(names.items())),
                "person_ids_changed": dict(sorted(ids.items())),
                "conflicts_skipped": conflicts,
                "episodes": len(touched),
                "totals": totals,
                "files_written": len(receipts),
                "episodes_sample": touched[:20],
            },
        )


def dry_run_report(root: Path) -> Dict[str, Any]:
    """Read-only: the frozen-to-be maps and what applying them would touch. Writes nothing."""
    root = Path(root)
    names, ids, conflicts = plan_maps(root)
    migration = SpeakerNamesCanonicalisedMigration()
    totals: Dict[str, int] = {}
    episodes: List[str] = []
    for ep in migration._episodes(root, names, ids):
        episodes.append(str(ep.meta.relative_to(root)))
        for k, v in ep.counts.items():
            totals[k] = totals.get(k, 0) + v
    return {
        "names": dict(sorted(names.items())),
        "ids": dict(sorted(ids.items())),
        "conflicts": conflicts,
        "episodes": len(episodes),
        "episodes_sample": episodes[:20],
        "totals": totals,
    }
