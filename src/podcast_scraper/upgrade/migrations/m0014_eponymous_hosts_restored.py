"""0014 — restore the host a show is named after, where their own voice said who they are.

The show-name rule (#2064) reads a title PREFIX as the show, so "Peter Attia" was "The Peter Attia
Drive" and lost his host seat on every episode: on prod (2026-10-01), 40 of 40 episodes with no
placed roster host, KG ``person:peter-attia`` only ``mentioned``, and his GI quotes episode-scoped
as ``person:unresolved-peter-attia-<episode>`` named "Unidentified speaker". The pipeline now
exempts a SELF-INTRODUCED name from that rule; this repairs what it already wrote. Everything it
needs is on disk — the diagnostics record that the voice introduced itself (``source:
self_intro``) and the segments keep its label — so no LLM, no reprocess.

Only names that pass BOTH tests are touched: the roster's diagnostics say a voice introduced itself
with it, and ``names_the_show`` matches it. Measured: Peter Attia 40 episodes; every real show-name
label (Machine Learning Street, Trivium China, Africa Tech Summit, Turkey Book) had 0 self-intros.

Per episode, all surfaces or none — what the fixed pipeline writes:

* ``.metadata.json`` — a placed roster entry (role from the diagnostics, ``source: self_intro``,
  its voices); the duplicate ``placed: false`` entry for the same person goes.
* ``.kg.json`` — ``person:<slug>`` gets the speaking role (added if missing) and the per-show
  ``HOSTS`` / ``GUESTS_ON`` edge.
* ``.gi.json`` — the episode-scoped id is merged into ``person:<slug>`` (``rewrite_ids``), the node
  is named, and quotes drop the "Unidentified speaker" label.
* ``.bridge.json`` — the scoped identity is re-pointed at the person.

Backed up, receipted and undoable through ``upgrade.file_rewrite``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...graph_id_utils import person_node_id
from ...identity.bare_name_scope import rewrite_ids, scoped_person_id
from ...kg.speaker_coherence import same_person
from ...speaker_detectors.hosts import names_the_show
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0014_eponymous_hosts_restored"
RECEIPTS_FILE = "eponymous_hosts_restored.jsonl"
BACKUP_TAG = "0014"
SELF_INTRO = "self_intro"
_EDGE_FOR_ROLE = {"host": "HOSTS", "guest": "GUESTS_ON"}


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sibling(meta: Path, suffix: str) -> Path:
    return meta.with_name(meta.name[: -len(".metadata.json")] + suffix)


def _diagnostics(meta: Path, content: dict) -> dict:
    rel = str((content or {}).get("transcript_file_path") or "")
    if not rel:
        return {}
    path = meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json")
    payload = _load(path)
    return payload if isinstance(payload, dict) else {}


def _eponymous(meta_payload: dict, diagnostics: dict) -> Dict[str, Tuple[str, List[str]]]:
    """``{name: (role, voices)}`` for self-introduced names the show-name rule would refuse."""
    feed_title = str((meta_payload.get("feed") or {}).get("title") or "")
    if not feed_title:
        return {}
    out: Dict[str, Tuple[str, List[str]]] = {}
    for voice in diagnostics.get("voices") or []:
        if not isinstance(voice, dict) or voice.get("source") != SELF_INTRO:
            continue
        name = str(voice.get("resolved_name") or "")
        if not name or not names_the_show(name, feed_title):
            continue
        role = str(voice.get("role") or "host").lower()
        role = role if role in _EDGE_FOR_ROLE else "host"
        prior_role, voices = out.get(name, (role, []))
        out[name] = (prior_role, voices + [str(voice.get("voice"))])
    return out


class _Episode:
    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.files: Dict[Path, Any] = {}
        self.changed: set = set()
        self.restored: List[str] = []

    def plan(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        content = meta.setdefault("content", {})
        names = _eponymous(meta, _diagnostics(self.meta, content))
        if not names:
            return False
        episode_id = str((meta.get("episode") or {}).get("episode_id") or "")
        self.files[self.meta] = meta
        for name, (role, voices) in names.items():
            if self._roster(content, name, role, voices):
                self.changed.add(self.meta)
            self._kg(name, role, episode_id)
            self._gi(name, episode_id)
            self._bridge(name, episode_id)
            if self.changed:
                self.restored.append(name)
        return bool(self.changed)

    def _roster(self, content: dict, name: str, role: str, voices: List[str]) -> bool:
        speakers = list(content.get("speakers") or [])
        if any(
            s.get("placed") is True and same_person(str(s.get("name") or ""), name)
            for s in speakers
        ):
            return False
        kept = [
            s
            for s in speakers
            if not (s.get("placed") is False and same_person(str(s.get("name") or ""), name))
        ]
        entry = {
            "id": role,
            "name": name,
            "role": role,
            "placed": True,
            "voices": voices,
            "source": SELF_INTRO,
        }
        content["speakers"] = [entry] + kept
        return True

    def _kg(self, name: str, role: str, episode_id: str) -> None:
        from ...kg.pipeline import _typed_person_org_node

        path = _sibling(self.meta, ".kg.json")
        kg = self.files.get(path) or _load(path)
        if not isinstance(kg, dict):
            return
        before = copy.deepcopy(kg)
        pid = person_node_id(name)
        nodes = kg.setdefault("nodes", [])
        edges = kg.setdefault("edges", [])
        node = next((n for n in nodes if n.get("id") == pid), None)
        if node is None:
            nodes.append(
                _typed_person_org_node(
                    name=name, entity_kind="person", role=role, episode_id=episode_id
                )
            )
            ep_node = next((n.get("id") for n in nodes if n.get("type") == "Episode"), None)
            if ep_node:
                edges.append({"from": pid, "to": ep_node, "type": "MENTIONS", "properties": {}})
        elif (node.get("properties") or {}).get("role") == "mentioned":
            node["properties"]["role"] = role
        podcast = next((n.get("id") for n in nodes if n.get("type") == "Podcast"), None)
        edge_type = _EDGE_FOR_ROLE[role]
        if podcast and not any(
            e.get("from") == pid and e.get("to") == podcast and e.get("type") == edge_type
            for e in edges
        ):
            edges.append({"from": pid, "to": podcast, "type": edge_type, "properties": {}})
        if kg != before:
            self.files[path] = kg
            self.changed.add(path)

    def _gi(self, name: str, episode_id: str) -> None:
        path = _sibling(self.meta, ".gi.json")
        gi = self.files.get(path) or _load(path)
        if not isinstance(gi, dict):
            return
        pid = person_node_id(name)
        scoped = scoped_person_id(pid, episode_id or "unknown")
        if not any(n.get("id") == scoped for n in gi.get("nodes") or []):
            return
        new, _changes = rewrite_ids(gi, {scoped: pid})
        for node in new.get("nodes") or []:
            props = node.setdefault("properties", {})
            if node.get("id") == pid and node.get("type") == "Person":
                props["name"] = name
            if node.get("type") == "Quote" and props.get("speaker_id") == pid:
                props.pop("speaker_name", None)
        self.files[path] = new
        self.changed.add(path)

    def _bridge(self, name: str, episode_id: str) -> None:
        path = _sibling(self.meta, ".bridge.json")
        bridge = self.files.get(path) or _load(path)
        if not isinstance(bridge, dict):
            return
        pid = person_node_id(name)
        scoped = scoped_person_id(pid, episode_id or "unknown")
        ids = bridge.get("identities") or []
        if not any(i.get("id") == scoped for i in ids):
            return
        has_pid = any(i.get("id") == pid for i in ids)
        out = []
        for ident in ids:
            if ident.get("id") == scoped:
                if has_pid:
                    continue
                ident = {**ident, "id": pid, "display_name": name}
            out.append(ident)
        bridge["identities"] = out
        self.files[path] = bridge
        self.changed.add(path)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class EponymousHostsRestoredMigration(Migration):
    """Give a self-introduced host back the seat the show-name rule took."""

    id = MIGRATION_ID
    to_version = "2.7.8"
    description = (
        'A host the show is named after ("The Peter Attia Drive") lost their seat to the '
        "show-name rule although their own voice introduced them. Restore roster, KG role, GI "
        "speaker and bridge identity from the stored self-introduction"
    )

    def _episodes(self, root: Path) -> List[_Episode]:
        out = []
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta)
            if ep.plan():
                out.append(ep)
        return out

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        eps = self._episodes(ctx.corpus_root)
        names = sorted({n for ep in eps for n in ep.restored})
        return f"eponymous hosts plan: {len(eps)} episode(s) for {names}"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served episode still has a self-introduced host the show-name rule removed."""
        left = [ep.meta.name for ep in self._episodes(ctx.corpus_root)]
        if left:
            return (
                False,
                f"{len(left)} episode(s) still lack their self-introduced host: {left[:5]}",
            )
        return True, "every self-introduced eponymous host holds their seat"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Restore a host the show-name rule unseated, where their own voice introduced them.

        Reads the STORED self-introduction rather than re-deriving: the evidence that this
        person is the host is already in the corpus, and re-running detection would make
        the repair depend on whatever the detector does today.
        """
        root = ctx.corpus_root
        eps = self._episodes(root)
        receipts = []
        if not ctx.dry_run:
            for ep in eps:
                for path in sorted(ep.changed):
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        names = sorted({n for ep in eps for n in ep.restored})
        verb = "would restore" if ctx.dry_run else "restored"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {names} on {len(eps)} episode(s)",
            details={"episodes": len(eps), "names": names, "files_written": len(receipts)},
        )
