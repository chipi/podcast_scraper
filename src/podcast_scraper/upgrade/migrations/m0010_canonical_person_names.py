"""0010 — one spelling, one id, for a person already on disk (#2130 step 1).

``canonical_person_name`` is now applied at the LOWEST layer that mints a person id
(``identity.slugify.person_id``) and at the other one (``graph_id_utils.entity_node_id``), so
``Peter Attia, MD`` mints ``person:peter-attia`` and publishes the name ``Peter Attia``. A
mint-time change only reaches episodes processed after it; production's artifacts were written
before, so without this step the corpus holds both spellings and both ids indefinitely.

MEASURED by dry-running this migration over the 2,257-episode production snapshot (2026-09-21),
and the measurement is why it does TWO things rather than one::

    2257 episodes scanned, 91 would change
    (39 id(s) remapped, of which 33 merge into an id already in the same episode;
     195 published name(s) rewritten), 0 unparsable

* ids that change: exactly one pair, ``person:peter-attia-md`` -> ``person:peter-attia``, across
  39 episodes — and in 33 of them the target id is ALREADY PRESENT in the same episode. So the
  common case is a MERGE, not a rename, which is precisely what a naive rewrite corrupts;
* names that change with NO id change: 195 occurrences of 24 spellings — the ``Name)`` close-paren
  family the extractor cut at a bracket (``Sophia Dew)``, ``Aaron Levie)``), the trailing-period
  generationals (``Robert F. Kennedy Jr.``, ``Donald Trump Jr.``) and doubled internal spaces
  (``Kevin  Murphy``).

  ``slugify`` already drops all of that, so every one of these mints the SAME id it always did —
  nothing about the id tells you the name is wrong. An id-only migration would have called itself
  finished and left ``Sophia Dew)`` on the person rail: 195 occurrences against the 39 the id fix
  covers.

THREE SURFACES, because the name is published from all of them: the ``.gi.json`` and
``.kg.json`` person nodes (``properties.name``), and the episode's ``.metadata.json``
``content.speakers[].name``, which is what the player and the API read. Leaving any one of
them produces exactly the inconsistency this exists to remove.

THE ID REMAP REUSES ``rewrite_ids`` VERBATIM (as 0007 does). It already merges two nodes landing
on one id rather than emitting a duplicate, rewrites every edge endpoint, rewrites quote nodes'
``properties.speaker_id``, and applies the role precedence a merge needs (a stated
``host``/``guest`` beats ``mentioned``). Re-implementing any of that here would be a second
answer to a question the corpus must only have one answer to.

SCOPED IDS ARE LEFT ALONE. ``person:unresolved-alex-ep42`` is 0007's deliberate episode-scoping of
a bare first name; re-minting from its ``name`` property would resolve it back to ``person:alex``
and silently undo that migration. ``is_scoped_person_id`` gates it.

Idempotent: a migrated artifact's names are already canonical and its ids already agree with them,
so the second run plans an empty map, finds no name to rewrite, and writes nothing. Unparsable
files are recorded and skipped rather than failing the run (mirrors 0003/0005/0006/0007).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ...identity.bare_name_scope import is_scoped_person_id, rewrite_ids
from ...identity.slugify import canonical_person_name, person_id, slugify
from ..migration import Migration, MigrationContext, MigrationResult

_PERSON = "person:"


def _iter_gi_files(root: Path) -> Iterable[Path]:
    """All ``*.gi.json`` files under *root* (recursive). Stable order."""
    return sorted(root.rglob("*.gi.json"))


def _sibling(gi_path: Path, suffix: str) -> Path:
    """The ``.kg.json`` / ``.metadata.json`` beside a ``.gi.json`` — same stem, same episode."""
    return gi_path.with_name(gi_path.name[: -len(".gi.json")] + suffix)


def _load(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, exc.__class__.__name__


def _write_atomic(path: Path, payload: dict) -> None:
    """tmp + os.replace — a kill mid-write must not leave a truncated, unparsable artifact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _person_nodes(payload: Optional[dict]) -> Iterable[dict]:
    """Every ``person:`` node in a GI/KG artifact, scoped placeholders included.

    Callers decide what to do with a scoped id; this only finds them, because a scoped node's
    NAME still needs canonicalising even though its id must not be re-minted.
    """
    nodes = (payload or {}).get("nodes")
    if not isinstance(nodes, list):
        return []
    return [
        n
        for n in nodes
        if isinstance(n, dict) and isinstance(n.get("id"), str) and n["id"].startswith(_PERSON)
    ]


def _plan_ids(*payloads: Optional[dict]) -> Dict[str, str]:
    """``{old_id: new_id}`` for the person nodes across one episode's layers.

    THE RULE IS NOT "RE-MINT FROM THE NAME", and the difference is the whole safety of this
    migration. A node is remapped only when the OLD rule, applied to this node's OWN published
    name, produces the id it already has — i.e. the id demonstrably came from this name — and the
    NEW rule produces a different one. Anything else is left alone.

    WHY, measured on the 2,257-episode production snapshot. The naive rule ("re-mint from the
    name") plans 83 remaps, and only the 39 ``person:peter-attia-md`` episodes are real. The other
    44 are nodes whose id is a full name while ``properties.name`` holds something narrower::

        person:joe-weisenthal   name "Joe"        -> person:joe
        person:casey-newton     name "Casey"      -> person:casey
        person:kashmir-hill     name "Kashmir"    -> person:kashmir
        person:elad-gil         name "Gil"        -> person:gil
        person:speaker-substackpost213603862-06   -> person:speaker-06

    Every one of those would demote a fully-resolved person to a global bare token — precisely
    what 0007 exists to prevent — or collapse an episode-scoped speaker placeholder into a shared
    one. Under the rule below they are all skipped, because ``slugify("Joe")`` is ``joe``, which is
    not the id the node carries, so nothing says that id was ever minted from that name.
    """
    id_map: Dict[str, str] = {}
    for payload in payloads:
        for node in _person_nodes(payload):
            nid = str(node["id"])
            if is_scoped_person_id(nid):
                continue  # 0007's episode scoping — re-minting would undo it
            name = (node.get("properties") or {}).get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            try:
                minted_by_the_old_rule = f"{_PERSON}{slugify(name)}"
            except ValueError:
                continue  # a name that slugifies to nothing tells us nothing about the id
            if minted_by_the_old_rule != nid:
                continue  # this id did not come from this name — not ours to touch
            want = person_id(name)
            if want != nid:
                id_map[nid] = want
    return id_map


def _duplicate_person_ids(*payloads: Optional[dict]) -> Dict[str, str]:
    """Person ids carried by MORE THAN ONE node in the same layer, as a self-map.

    TWO NODES, ONE ID IS A CORRUPT GRAPH, and the close-paren defect produced it directly: a
    person written both ``Jen Kha`` and ``Jen Kha)`` is two different name strings that
    ``slugify`` reduces to the same ``person:jen-kha``, so both nodes were emitted under it.
    Found in the production snapshot (``The State of AI: Macro, Apps and...``, kg layer) — one
    node ``mentioned``, the other ``host``.

    Canonicalising the NAME does not fix this: the ids already agree, so the remap never runs and
    the artifact ends with two identical-looking nodes instead of two different-looking ones.

    Returned as ``{id: id}`` because that is what ``rewrite_ids`` needs to take its merge path —
    the merge is keyed on the resulting id and folds duplicates with the role precedence a merge
    needs (a stated ``host``/``guest`` beats ``mentioned``, which is exactly the Jen Kha case).
    A self-map changes no id, so it adds nothing to the remap count.
    """
    dupes: Dict[str, str] = {}
    for payload in payloads:
        seen: set = set()
        for node in _person_nodes(payload):
            nid = str(node["id"])
            if nid in seen:
                dupes[nid] = nid
            seen.add(nid)
    return dupes


def _canonicalize_names(payload: Optional[dict]) -> Tuple[Optional[dict], int]:
    """Rewrite every person node's published name to its canonical spelling. ``(payload, count)``.

    Runs on a copy the caller owns. Applies to scoped nodes too — the SCOPE is 0007's business,
    the SPELLING is this migration's.
    """
    if payload is None:
        return None, 0
    changed = 0
    for node in _person_nodes(payload):
        props = node.get("properties")
        if not isinstance(props, dict):
            continue
        name = props.get("name")
        if not isinstance(name, str):
            continue
        canonical = canonical_person_name(name)
        if canonical and canonical != name:
            props["name"] = canonical
            changed += 1
    return payload, changed


def _canonicalize_speakers(payload: Optional[dict]) -> Tuple[Optional[dict], int]:
    """Same, for ``content.speakers[].name`` in an episode's ``.metadata.json``.

    This is the surface the player and the API read, so a corpus whose graphs were fixed and whose
    metadata was not still SHOWS the old spelling to every listener.
    """
    if payload is None:
        return None, 0
    speakers = (payload.get("content") or {}).get("speakers")
    if not isinstance(speakers, list):
        return payload, 0
    changed = 0
    for speaker in speakers:
        if not isinstance(speaker, dict):
            continue
        name = speaker.get("name")
        if not isinstance(name, str):
            continue
        canonical = canonical_person_name(name)
        if canonical and canonical != name:
            speaker["name"] = canonical
            changed += 1
    return payload, changed


class _EpisodePlan:
    """What one episode's three artifacts would become. Pure data; nothing is written here."""

    __slots__ = ("gi", "kg", "meta", "id_map", "dupe_ids", "name_changes", "error")

    def __init__(self) -> None:
        self.gi: Optional[dict] = None
        self.kg: Optional[dict] = None
        self.meta: Optional[dict] = None
        self.id_map: Dict[str, str] = {}
        self.dupe_ids: Dict[str, str] = {}
        self.name_changes = 0
        self.error: Optional[str] = None


def _plan_for_episode(gi_path: Path) -> _EpisodePlan:
    """Read one episode's ``.gi``/``.kg``/``.metadata`` trio and compute the whole rewrite.

    PAIRED, NOT PER FILE, for the same reason 0007 is: the id map is computed from the UNION of
    the two graph layers, because a person can be a node in one and only an edge endpoint in the
    other. Rewriting them with different maps would leave the episode's graphs disagreeing about
    who somebody is — worse than not migrating.
    """
    plan = _EpisodePlan()
    gi, err = _load(gi_path)
    if gi is None:
        plan.error = f"{gi_path}: {err}"
        return plan
    plan.gi = gi

    kg_path = _sibling(gi_path, ".kg.json")
    if kg_path.is_file():
        kg, kg_err = _load(kg_path)
        if kg is None:
            plan.error = f"{kg_path}: {kg_err}"
            return plan
        plan.kg = kg

    meta_path = _sibling(gi_path, ".metadata.json")
    if meta_path.is_file():
        meta, meta_err = _load(meta_path)
        if meta is None:
            plan.error = f"{meta_path}: {meta_err}"
            return plan
        plan.meta = meta

    plan.id_map = _plan_ids(plan.gi, plan.kg)
    plan.dupe_ids = _duplicate_person_ids(plan.gi, plan.kg)
    for payload in (plan.gi, plan.kg):
        for node in _person_nodes(payload):
            name = (node.get("properties") or {}).get("name")
            if isinstance(name, str):
                canonical = canonical_person_name(name)
                if canonical and canonical != name:
                    plan.name_changes += 1
    speakers = ((plan.meta or {}).get("content") or {}).get("speakers")
    if isinstance(speakers, list):
        for speaker in speakers:
            if isinstance(speaker, dict) and isinstance(speaker.get("name"), str):
                canonical = canonical_person_name(speaker["name"])
                if canonical and canonical != speaker["name"]:
                    plan.name_changes += 1
    return plan


class CanonicalPersonNamesMigration(Migration):
    """One spelling and one id per person, across an existing corpus."""

    id = "0010_canonical_person_names"
    to_version = "2.7.4"
    description = (
        "#2130: `canonical_person_name` is applied where person ids are minted and where names "
        "are published, so `Peter Attia, MD` becomes `person:peter-attia` named `Peter Attia`. "
        "Remap (and MERGE) the ids already on disk, and rewrite the published name on all three "
        "surfaces — .gi.json, .kg.json and .metadata.json content.speakers"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            return "no .gi.json files under corpus — nothing to migrate"
        episodes = remapped = merged = renamed = deduped = unparsable = 0
        for gi_path in files:
            plan = _plan_for_episode(gi_path)
            if plan.error:
                unparsable += 1
                continue
            if not (plan.id_map or plan.dupe_ids or plan.name_changes):
                continue
            episodes += 1
            remapped += len(plan.id_map)
            deduped += len(plan.dupe_ids)
            renamed += plan.name_changes
            # A merge, not a rename: the target id already has a node in this episode.
            present = {str(n["id"]) for n in _person_nodes(plan.gi)} | {
                str(n["id"]) for n in _person_nodes(plan.kg)
            }
            merged += sum(1 for old, new in plan.id_map.items() if new in present - {old})
        return (
            f"canonical person names plan: {len(files)} episodes scanned, {episodes} would "
            f"change ({remapped} id(s) remapped, of which {merged} merge into an id already in "
            f"the same episode; {deduped} duplicate node id(s) folded; "
            f"{renamed} published name(s) rewritten), "
            f"{unparsable} unparsable (will be skipped)"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Is the migration's effect actually PRESENT in the corpus? ``(ok, message)``.

        The default is ``True, "no verification defined"``, and this repo has already paid for
        that once: ``upgrade verify`` answered "no verification defined" while the ledger claimed
        a version the data did not have, and nothing in the system could tell (recorded in
        ``upgrade/state.py`` and ``role_ledger.py``). A migration that runs unattended after a
        deploy and cannot be checked afterwards is a migration you have to trust.

        Both halves are checked, because both are things this migration promised:

        * no person node publishes a non-canonical name — the 195-occurrence half;
        * no person id disagrees with the name that minted it — the 39-episode half, expressed as
          the same narrow rule ``_plan_ids`` uses, so verify cannot demand something apply never
          intended to do.

        Deliberately NOT a re-run of ``plan``: the ledger excludes an already-applied migration,
        so plan-returns-zero would be vacuous. This reads the corpus directly.
        """
        bad_names: List[str] = []
        bad_ids: List[str] = []
        for gi_path in _iter_gi_files(ctx.corpus_root):
            plan = _plan_for_episode(gi_path)
            if plan.error:
                continue
            for payload in (plan.gi, plan.kg, None):
                for node in _person_nodes(payload):
                    name = (node.get("properties") or {}).get("name")
                    if isinstance(name, str) and canonical_person_name(name) not in (name, ""):
                        bad_names.append(f"{gi_path.name}: {name!r}")
            speakers = ((plan.meta or {}).get("content") or {}).get("speakers")
            if isinstance(speakers, list):
                for speaker in speakers:
                    if not isinstance(speaker, dict):
                        continue
                    name = speaker.get("name")
                    if isinstance(name, str) and canonical_person_name(name) not in (name, ""):
                        bad_names.append(f"{gi_path.name} (speakers): {name!r}")
            for old, new in plan.id_map.items():
                bad_ids.append(f"{gi_path.name}: {old} should be {new}")
            for dup in plan.dupe_ids:
                bad_ids.append(f"{gi_path.name}: {dup} is carried by two nodes at once")

        if bad_names or bad_ids:
            head = "; ".join((bad_ids + bad_names)[:5])
            return False, (
                f"{len(bad_ids)} person id(s) and {len(bad_names)} published name(s) are still "
                f"pre-canonical: {head}"
            )
        return (
            True,
            "every person name is canonical and every id agrees with the name that minted it",
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite each episode's three artifacts together, or none of them.

        They share one id map and one spelling; writing some and not others is the inconsistent
        state this migration exists to remove.
        """
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            ctx.log(f"no .gi.json files under {ctx.corpus_root}")
            return MigrationResult(
                self.id,
                applied=True,
                dry_run=ctx.dry_run,
                message="no .gi.json files found",
                details={"episodes_scanned": 0},
            )

        changed: List[str] = []
        remapped = merged = renamed = deduped = unchanged = 0
        unparsable: List[str] = []

        for gi_path in files:
            plan = _plan_for_episode(gi_path)
            if plan.error:
                unparsable.append(plan.error)
                continue
            if plan.gi is None or not (plan.id_map or plan.dupe_ids or plan.name_changes):
                unchanged += 1
                continue

            present = {str(n["id"]) for n in _person_nodes(plan.gi)} | {
                str(n["id"]) for n in _person_nodes(plan.kg)
            }
            merged += sum(1 for old, new in plan.id_map.items() if new in present - {old})
            remapped += len(plan.id_map)

            # IDS FIRST, THEN NAMES. `rewrite_ids` may MERGE two nodes into one, and the survivor
            # keeps whichever name it carried; canonicalising afterwards fixes the survivor
            # whatever it turned out to be. The other order would tidy a name that is then
            # discarded with its node, and leave the survivor's untouched.
            # The self-mapped duplicates ride along in the SAME map: `rewrite_ids` keys its merge
            # on the resulting id, so including them is what makes it fold two nodes sharing one
            # id into one. They change no id, so they add nothing to `remapped`.
            _full_map = {**plan.id_map, **plan.dupe_ids}
            new_gi, gi_id_changes = rewrite_ids(copy.deepcopy(plan.gi), _full_map)
            new_kg, kg_id_changes = (
                rewrite_ids(copy.deepcopy(plan.kg), _full_map) if plan.kg is not None else ({}, 0)
            )
            deduped += len(plan.dupe_ids)
            out_gi, gi_name_changes = _canonicalize_names(new_gi)
            out_kg, kg_name_changes = _canonicalize_names(new_kg if plan.kg is not None else None)
            out_meta, meta_changes = _canonicalize_speakers(copy.deepcopy(plan.meta))
            renamed += gi_name_changes + kg_name_changes + meta_changes

            if not any(
                (
                    gi_id_changes,
                    kg_id_changes,
                    gi_name_changes,
                    kg_name_changes,
                    meta_changes,
                    plan.dupe_ids,
                )
            ):
                unchanged += 1
                continue

            changed.append(str(gi_path.relative_to(ctx.corpus_root)))
            if ctx.dry_run:
                continue
            if (gi_id_changes or gi_name_changes or plan.dupe_ids) and out_gi is not None:
                _write_atomic(gi_path, out_gi)
            if (kg_id_changes or kg_name_changes or plan.dupe_ids) and out_kg:
                _write_atomic(_sibling(gi_path, ".kg.json"), out_kg)
            if meta_changes and out_meta is not None:
                _write_atomic(_sibling(gi_path, ".metadata.json"), out_meta)

        message = (
            f"{'would rewrite' if ctx.dry_run else 'rewrote'} {len(changed)} episode(s): "
            f"{remapped} id(s) remapped ({merged} merged into an existing id), "
            f"{deduped} duplicate node id(s) folded, "
            f"{renamed} published name(s) canonicalised; {unchanged} already-current, "
            f"{len(unparsable)} unparsable"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "episodes_scanned": len(files),
                "changed": len(changed),
                "ids_remapped": remapped,
                "ids_merged": merged,
                "duplicate_ids_folded": deduped,
                "names_canonicalised": renamed,
                "unparsable": unparsable[:20],
            },
        )
