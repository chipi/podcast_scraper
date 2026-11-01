"""0041 — a joined person's name in the episode files is the group's one spelling (text only).

RFC-131 plan V3, D2 items 9 and 8. Record: ``docs/wip/changes/0041_joined_names_respelled.md``.

After S6a-S6d every served surface shows a joined person under the ledger's ``display_name``, at
read time; the transcript keeps each episode's own spelling. This writes the group spelling into
the episode files, so the transcript and every other surface agree. It is also the code the ledger
step runs after each build (``identity.respell``).

**The decision, per episode** (:func:`episode_renames`): a Person node name (KG and GI together) is
respelled when every node of that name is joined and all of them take one group spelling that
differs from the name. A name with an unjoined node, or two group spellings, is refused and
counted. Ids never move with a spelling (ADR-166).

**Item 8's fold:** a roster entry that 0025's ``same_person`` matches to a respelled person under
another full-name spelling is that person listed twice. It takes the group spelling, its node
folds into the joined id, and 0025's ``_record`` merges the two entries: the one case where an id
moves, an identity decision on the episode's own evidence (0025's precedent). Refused when both
ids are in ledger groups, when the variant has episodes elsewhere, when either name is one word,
or when the variant is a scoped id.

**The rewrite** is 0025's name-map path through 0031's derived surfaces (ad maps, metadata counts):
metadata, segments, the three transcripts in place with every GI quote and ad-free offset carried,
GI, KG (speakers and mentioned people), bridge, node labels, diagnostics voices, turns (only when a
speaker label changed), context. Kept as evidence: ``detected_hosts`` / ``detected_guests`` (the
feed's and show notes' hint), the diagnostics' publisher text, decision traces, and every spoken
word. UNDO restores from ``.podcast_scraper/upgrade-backups/0041/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...identity.bare_name_scope import is_scoped_person_id
from ...identity.display import PersonDisplay
from ..file_rewrite import undo_from_receipts
from ..migration import MigrationContext
from . import m0029_naming_arc_voices_repaired as m29
from .m0025_one_person_one_entry import _load, _run_relative, same_person
from .voice_rewrite_derived import DerivedSurfacesEpisode

MIGRATION_ID = "0041_joined_names_respelled"
RECEIPTS_FILE = "joined_names_respelled.jsonl"
BACKUP_TAG = "0041"
_PUBLISHER_HINT = ("detected_hosts", "detected_guests")


class LedgerView:
    """What the ledger says, as this migration reads it: each joined id's group spelling, each
    id's canonical, and each id's episode count."""

    def __init__(
        self,
        display: PersonDisplay,
        id_map: Dict[str, str],
        episodes_of: Optional[Dict[str, int]] = None,
    ) -> None:
        self.display, self.id_map = display, id_map
        self.episodes_of = episodes_of or {}

    def joined(self, person_id: str) -> bool:
        """True when the ledger joins ``person_id`` into a group with a spelling."""
        return self.display.name_of(person_id) is not None


def served_view(root: Path) -> Optional[LedgerView]:
    """The served ledger's view; None when there is no valid ledger (then nothing is respelled)."""
    from ...identity.ledger import read_ledger

    ledger, _state = read_ledger(root)
    if ledger is None:
        return None
    return view_of(ledger)


def view_of(ledger: Dict[str, Any]) -> LedgerView:
    """The view one ledger gives (the migration's served one, or the ledger step's new build)."""
    from ...identity.display import display_from_ledger
    from ...identity.ledger import ledger_id_map

    episodes: Dict[str, int] = {}
    for cluster in (ledger.get("clusters_payload") or {}).get("clusters") or []:
        for m in cluster.get("members") or []:
            if m.get("id"):
                episodes[str(m["id"])] = int(m.get("episode_count") or 0)
    return LedgerView(display_from_ledger(ledger), ledger_id_map(ledger), episodes)


def _person_nodes_by_name(meta: Path) -> Dict[str, Set[str]]:
    """``{name: ids}`` over the episode's KG and GI Person nodes."""
    stem = str(meta)[: -len(".metadata.json")]
    out: Dict[str, Set[str]] = {}
    for suffix in (".kg.json", ".gi.json"):
        doc = _load(Path(stem + suffix))
        for n in (doc or {}).get("nodes") or [] if isinstance(doc, dict) else []:
            if isinstance(n, dict) and n.get("type") == "Person" and n.get("id"):
                name = str((n.get("properties") or {}).get("name") or "").strip()
                if name:
                    out.setdefault(name, set()).add(str(n["id"]))
    return out


def _placeholder(person_id: str) -> bool:
    """An episode-scoped bare name, or an unresolved-speaker placeholder of any shape: kept as
    the episode wrote it. A group's full spelling written over a bare scoped name makes it a full
    name the ledger's join test then matches against other people (record 4f); read time still
    shows the group spelling."""
    from ...graph_id_utils import is_unresolved_speaker_placeholder

    return is_scoped_person_id(person_id) or is_unresolved_speaker_placeholder(person_id)


def episode_renames(
    meta: Path, view: LedgerView
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """``(renames, folds, refused)`` for one episode: ``{own name: group spelling}``, ``{variant id:
    surviving id}`` (item 8), and ``{name: why}`` for each name left as it is."""
    by_name = _person_nodes_by_name(meta)
    renames: Dict[str, str] = {}
    refused: Dict[str, str] = {}
    settled: Set[str] = set()
    for name, ids in by_name.items():
        if any(_placeholder(i) for i in ids):
            if any(view.joined(i) for i in ids):
                refused[name] = "an episode-scoped id"
            continue
        spellings = {view.display.name_of(i) for i in ids}
        if spellings == {name}:
            settled.add(name)
            continue
        if spellings == {None}:
            continue
        if None in spellings:
            refused[name] = "a node of the name is not joined"
        elif len(spellings) > 1:
            refused[name] = "the name's nodes take two group spellings"
        else:
            new = next(iter(spellings))
            if new and new != name:
                renames[name] = new
    folds = _folds(meta, view, by_name, renames, settled, refused)
    return renames, folds, refused


def _folds(
    meta: Path,
    view: LedgerView,
    by_name: Dict[str, Set[str]],
    renames: Dict[str, str],
    settled: Set[str],
    refused: Dict[str, str],
) -> Dict[str, str]:
    """Item 8: a roster entry that is a respelled person under another spelling, folded in."""
    doc = _load(meta)
    content = (doc or {}).get("content") or {} if isinstance(doc, dict) else {}
    joined_names = dict(renames)
    joined_names.update({new: new for new in renames.values()})
    joined_names.update({name: name for name in settled})
    folds: Dict[str, str] = {}
    for entry in content.get("speakers") or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or name in joined_names:
            continue
        twin = next((k for k in joined_names if same_person(name, k)), None)
        if twin is None:
            continue
        new = joined_names[twin]
        if len(name.split()) < 2 or len(twin.split()) < 2:
            refused[name] = "a fold needs two full names"
            continue
        variant_ids = by_name.get(name, set())
        # The survivor sits under its file name, which may be a spelling being replaced.
        spelled_new = {old for old, spelled in renames.items() if spelled == new} | {twin, new}
        survivors = {i for nm in spelled_new for i in by_name.get(nm, set()) if view.joined(i)}
        if len(survivors) != 1:
            refused[name] = "the fold has no single joined id to survive"
            continue
        survivor = next(iter(survivors))
        bad = [i for i in variant_ids if view.joined(i) or is_scoped_person_id(i)]
        if bad:
            refused[name] = "the variant id is joined or scoped"
            continue
        if any(view.episodes_of.get(i, 1) > 1 for i in variant_ids):
            refused[name] = "the variant id has episodes elsewhere"
            continue
        renames[name] = new
        for i in variant_ids:
            if i != survivor:
                folds[i] = survivor
    return folds


class _Episode(DerivedSurfacesEpisode):
    """0025's name-map rewrite through 0031's derived surfaces, decided by the ledger."""

    def __init__(self, meta: Path, root: Path, view: Optional[LedgerView] = None) -> None:
        super().__init__(meta, root, voices={})
        self._view = view
        self.folds: Dict[str, str] = {}
        self.carried: Dict[str, str] = {}
        self.names_refused: Dict[str, str] = {}
        self._speaker_renamed = False

    def decide(self) -> bool:
        if self._view is None:
            return False
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        renames, self.folds, self.names_refused = episode_renames(self.meta, self._view)
        for _name in self.names_refused:
            self._bump("names_refused")
        self._carry_0025(renames)
        self.renames = renames
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), self.meta.parent.parent
        )
        if rel.endswith(".txt"):
            seg = _load(self.meta.parent.parent / rel.replace(".txt", ".segments.json"))
            rows = seg if isinstance(seg, list) else (seg or {}).get("segments") or []
            self._speaker_renamed = any(
                isinstance(r, dict) and r.get("speaker_label") in renames for r in rows
            )
        return bool(self.renames or self.dropped)

    def _carry_0025(self, renames: Dict[str, str]) -> None:
        """0025's pending decision on this episode, applied whole or not at all (record 4c, 4d).

        0025 is applied; a later naming migration (0034, 0035) wrote a second spelling of a person
        into the episode, so 0025's verify fails there. Its decision is the feed's stated host
        spelling, the publisher's text. Carried only when it is renames, dropped twins and host
        promotions alone, every target is in ``known_hosts`` or ``metadata_named``, and no carried
        name is held by a ledger group."""
        from . import m0025_one_person_one_entry as m25

        e25 = m25._Episode(self.meta, self.root)
        if not e25.decide() or e25.refused or e25.voice_names or e25.left:
            return
        view = self._view
        assert view is not None
        by_name = _person_nodes_by_name(self.meta)
        carry = {
            old: new
            for old, new in e25.renames.items()
            if old not in renames and not any(view.joined(i) for i in by_name.get(old, ()))
        }
        if not (carry or e25.dropped):
            return
        stated = self._stated_names()
        if any(new not in stated for new in carry.values()):
            self._bump("carry_0025_skipped_target_not_stated")
            return
        renames.update(carry)
        self.carried = carry
        self.dropped = list(e25.dropped)
        self.promote = set(e25.promote)
        self._bump("carry_0025_renames", len(carry))

    def _stated_names(self) -> Set[str]:
        """The episode's ``known_hosts`` and ``metadata_named``: the publisher's own statements."""
        doc = _load(self.meta) or {}
        rel = _run_relative(
            (doc.get("content") or {}).get("transcript_file_path"), self.meta.parent.parent
        )
        if not rel.endswith(".txt"):
            return set()
        diag = _load(self.meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
        tried = (diag or {}).get("tried") or {} if isinstance(diag, dict) else {}
        return {str(n) for k in ("known_hosts", "metadata_named") for n in tried.get(k) or [] if n}

    def _id_map(self) -> Dict[str, str]:
        """Item 8's folds, and for a carried 0025 rename 0025's own move (old name's id to the new
        name's, never a scoped id). A respelling of a joined person never moves an id (ADR-166)."""
        from ...identity.slugify import person_id

        out = dict(self.folds)
        for old, new in self.carried.items():
            try:
                a, b = person_id(old), person_id(new)
            except ValueError:
                continue
            if a != b and not is_scoped_person_id(a):
                out[a] = b
        return out

    def _record(self, meta: dict, promoted: Set[str]) -> None:
        """0025's merge and renumbering only where an entry folds; elsewhere the roster keeps its
        order and entry ids (m0017 has already respelled the names)."""
        if self.folds or self.carried or self.dropped or self._merges_by_name(meta):
            super()._record(meta, promoted)

    @staticmethod
    def _merges_by_name(meta: dict) -> bool:
        names = [
            s.get("name")
            for s in (meta.get("content") or {}).get("speakers") or []
            if isinstance(s, dict) and isinstance(s.get("name"), str)
        ]
        return len(names) != len(set(names))

    def _node_labels(self, ep: Any) -> None:
        """Every joined Person node takes the spelling, not only a speaker's (m0017 renames a
        speaker's ``name``; 0025 every ``label``), and so does every bridge identity of the name.
        Then the publisher hint m0017 respells is put back: it is evidence."""
        for path, payload in ep.files.items():
            if path.name.endswith((".kg.json", ".gi.json")) and isinstance(payload, dict):
                for node in payload.get("nodes") or []:
                    props = node.get("properties") if isinstance(node, dict) else None
                    if node.get("type") != "Person" or not isinstance(props, dict):
                        continue
                    for key in ("name", "label"):
                        if props.get(key) in self.renames:
                            props[key] = self.renames[props[key]]
                            self._bump(f"person_{key}s_respelled")
            elif path.name.endswith(".bridge.json") and isinstance(payload, dict):
                for ident in payload.get("identities") or []:
                    if isinstance(ident, dict) and ident.get("display_name") in self.renames:
                        ident["display_name"] = self.renames[ident["display_name"]]
                        self._bump("bridge_names_respelled")
        on_disk = _load(self.meta)
        content = (
            ep.files[self.meta].get("content")
            if isinstance(ep.files.get(self.meta), dict)
            else None
        )
        if isinstance(on_disk, dict) and isinstance(content, dict):
            for field in _PUBLISHER_HINT:
                if field in (on_disk.get("content") or {}):
                    content[field] = (on_disk.get("content") or {})[field]
        self._keep_own_spellings(ep, on_disk, content)

    def _keep_own_spellings(self, ep: Any, on_disk: Any, content: Any) -> None:
        """Each respelled Person node and roster entry keeps the episode's own spelling in
        ``respelled_from`` (record 4g): the ledger reads that as evidence, every surface shows the
        group spelling. The first original is kept. A carried 0025 rename is a correction of the
        episode's evidence, not a respelling, and records nothing."""
        presentation = {old for old in self.renames if old not in self.carried}
        for path, payload in ep.files.items():
            if not path.name.endswith((".kg.json", ".gi.json")) or not isinstance(payload, dict):
                continue
            before = _load(path)
            own = {
                n.get("id"): (n.get("properties") or {}).get("name")
                for n in (before or {}).get("nodes") or []
                if isinstance(n, dict) and n.get("type") == "Person"
            }
            for node in payload.get("nodes") or []:
                props = node.get("properties") if isinstance(node, dict) else None
                if node.get("type") != "Person" or not isinstance(props, dict):
                    continue
                was = own.get(node.get("id"))
                if was in presentation and props.get("name") != was:
                    props.setdefault("respelled_from", was)
        if not isinstance(content, dict) or not isinstance(on_disk, dict):
            return
        old_entries = [
            s for s in (on_disk.get("content") or {}).get("speakers") or [] if isinstance(s, dict)
        ]
        for entry in content.get("speakers") or []:
            if not isinstance(entry, dict) or "respelled_from" in entry:
                continue
            voices = tuple(entry.get("voices") or [])
            match = [s for s in old_entries if voices and tuple(s.get("voices") or []) == voices]
            if not match:
                olds = [o for o in presentation if self.renames[o] == entry.get("name")]
                match = [s for s in old_entries if s.get("name") in olds]
            if len(match) == 1 and match[0].get("name") in presentation:
                if match[0].get("name") != entry.get("name"):
                    entry["respelled_from"] = match[0]["name"]

    def _turns(self, files: Dict[Path, Any], changed: Set[Path], rel: str) -> None:
        """Rebuilt only when a speaker label changed; a mention-only respelling leaves them."""
        if self._speaker_renamed:
            super()._turns(files, changed, rel)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class JoinedNamesRespelledMigration(m29.NamingArcVoicesRepairedMigration):
    """Joined people's names written in the group's one spelling, as the served ledger gives it."""

    id = MIGRATION_ID
    to_version = "2.7.35"
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "joined names respelled"
    description = (
        "joined people's names in the episode files take the ledger's group spelling (RFC-131 "
        "D2 item 9; item 8's duplicates folded): 0025's name-map rewrite, ids kept"
    )
    #: Set per scan from the served ledger; a test passes its own.
    _view_override: Optional[LedgerView] = None

    def _episode_kwargs(self) -> Dict[str, Any]:
        return {"view": self._view}

    def _scan(self, root: Path) -> Tuple[List[Any], List[Any], Dict[str, int]]:
        self._view = self._view_override or served_view(root)
        return super()._scan(root)

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when nothing is left to respell and nothing was refused. No valid ledger: fail,
        since the migration cannot have run against one."""
        if (self._view_override or served_view(ctx.corpus_root)) is None:
            return False, "no valid identity ledger: nothing to respell against"
        write, refused, totals = self._scan(ctx.corpus_root)
        if write or refused:
            names = [ep.meta.name for ep in write + refused]
            return (
                False,
                f"{len(write)} episode(s) still to respell, {len(refused)} refused: {names[:5]}",
            )
        return True, "every joined person's name in the episode files is the group's spelling"


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses, with its counters and
    the sha256 of every file its plan read: the per-episode plan the outcome hook diffs prod's
    against the clone's (as 0034's ``plan_rows``)."""
    from ..corpus_selection import select_served_artifacts
    from ..read_recorder import recording

    root = Path(root)
    mig = JoinedNamesRespelledMigration()
    view = mig._view_override or served_view(root)
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta, root, view=view)
        with recording(root) as rec:
            planned = ep.plan()
        if not planned and not ep.refused:
            continue
        if ep.refused:
            ep._bump("refused")  # as ``_scan`` counts it, so rows sum to plan()'s totals
        yield {
            "meta": str(meta.relative_to(root)),
            "renames": len(ep.renames),
            "folds": dict(ep.folds),
            "refused": ep.refused,
            "counts": dict(sorted(ep.counts.items())),
            "inputs": rec.inputs(),
        }
