"""0025 — one person is one entry on an episode, whatever their title or spelling (c05cc0273).

The roster compared a leading title as the given name, and the record de-duplicated its unplaced
people with an exact-surname check only. So one person was published twice: a voice under the
title or spelling it said, and the feed's or the show notes' spelling beside it. Prod 2026-10-08:
22 served episodes ("Professor Hannah Frye" placed + "Hannah Fry" unplaced, "Traci Alloway" guest
+ "Tracy Alloway" host, "Bernard Leong" + "Bernard Leung", "Ali Ghodsi" twice, ...). The pipeline
no longer does this; this repairs what is on disk, with the pipeline's own predicates.

Two steps per episode, decided from its stored record and its feed's stated hosts (the speaker
diagnostics' ``tried.known_hosts``):

1. A PLACED name takes the stated host's spelling, and the host role, where the roster now snaps
   it (``roster._snap_near_identical_host``; for a voice the record calls a host, also
   ``_canonicalize_to_known_host``) — and only where this fix is the reason it did not before:
   the name carries a leading title, or the record also lists that host unplaced. Every DeepMind
   episode's "Professor Hannah Fry" becomes the feed's "Hannah Fry" (the person id does not move);
   Odd Lots' "Traci Alloway" guest becomes Tracy Alloway, host.
2. An UNPLACED entry the record already lists as a kept person
   (``_same_person`` or ``_same_person_on_one_episode``, as ``_unplaced_speakers`` now decides) is
   dropped; two placed entries that end with one name become one entry with both voices.

A rename reaches every surface that carries the name, all or none per episode: metadata
(``content.speakers``, ``detected_*``), ``.segments.json`` / ``.adfree.segments.json`` labels and
``speaker_role``, ``.kg.json`` / ``.gi.json`` / ``.bridge.json`` names and ids (m0017's rewrite,
ids move with the name) plus the Person node ``label``, the KG Person role, the speaker
diagnostics, the context digest's speaker fields, the ``<name>: `` prefixes of ``.txt`` /
``.adfree.txt`` / ``.cleaned.txt`` with every GI quote and ad-free segment offset shifted (m0023's
in-place path and its guard), and each variant's ``turns.json``, rebuilt by the pipeline's own
builder from the new text (or the episode is refused).

REFUSED, untouched and listed: an episode where a rename would give a voice a name another voice
holds in a different role (DeepMind 0014: "Professor Hannah Fry" sits on a guest seat while
"Hannah Fry" is on the host's; the right outcome moves Paige Bailey onto the other voice, which only
the roster can decide), one stated host claimed by two voices, or a transcript m0023's guard
refuses. LEFT and listed: two placed voices whose different names are one person by the predicate
("Andy Ratcliffe" / "Andy Rachleff") — which voice is whom is the roster's call, not this one's.

NOT DONE: a guest voice whose spelling differs from a stated name by more than the roster's host
rule allows keeps its own spelling; only the duplicate entry goes ("Anita Arnond" stays, the
unplaced "Anita Anand" is dropped), exactly as the pipeline now publishes.

``verify`` fails while any served episode still has something this migration would write OR
refuses, so a refusal never reads as done. UNDO restores from ``.podcast_scraper/upgrade-backups/
0025/`` while each file is still exactly what this migration wrote.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...builders.context_digest_builder import build_context_digest
from ...identity.bare_name_scope import is_scoped_person_id
from ...identity.slugify import person_id
from ...providers.ml.diarization import roster as R
from ...workflow.turns_artifact import build_turns_document, TURNS_SUFFIX
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, dump_json, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0017_speaker_names_canonicalised import _Episode as _RenameEpisode, _person_nodes
from .m0023_transcript_speaker_prefixes_resynced import (
    _Episode as _TranscriptEpisode,
    _write_text_with_backup,
    Refused,
)

MIGRATION_ID = "0025_one_person_one_entry"
RECEIPTS_FILE = "one_person_one_entry.jsonl"
BACKUP_TAG = "0025"
_CONTEXT_FIELDS = (("basic", "hosts"), ("basic", "guests"), (None, "people"))


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def same_person(a: str, b: str) -> bool:
    """One person on one episode's record — the predicate ``_unplaced_speakers`` now uses."""
    return a.lower() == b.lower() or R._same_person(a, b) or R._same_person_on_one_episode(a, b)


def _has_title(name: str) -> bool:
    toks = name.split()
    return len(toks) > 2 and toks[0].lower().strip(".,") in R.HONORIFIC_TITLES


def host_spelling(
    name: str, role: Optional[str], hosts: List[str], taken: Set[str]
) -> Optional[str]:
    """The stated host *name* is, by the roster's own host rules, or ``None``."""
    snapped = R._snap_near_identical_host(name, hosts, frozenset(taken))
    if snapped != name:
        return snapped
    if role == "host":
        canon = R._canonicalize_to_known_host(name, hosts)
        if canon != name:
            return canon
    return None


class _Transcripts(_TranscriptEpisode):
    """m0023's in-place prefix rename, driven by this episode's rename map."""

    def __init__(self, meta: Path, root: Optional[Path], renames: Dict[str, str]) -> None:
        super().__init__(meta, root)
        self._renames = renames

    def _old_names(self, run: Path, rel: str) -> Dict[str, str]:
        return dict(self._renames)


class _Episode:
    """One episode's decision and rewrite, in memory; nothing touches disk here."""

    def __init__(self, meta: Path, root: Path) -> None:
        self.meta = meta
        self.root = root
        self.json_files: Dict[Path, Any] = {}
        self.text_files: Dict[Path, str] = {}
        self.counts: Dict[str, int] = {}
        self.refused: Optional[str] = None
        self.left: List[Tuple[str, str]] = []
        self.renames: Dict[str, str] = {}
        self.dropped: List[str] = []

    def _bump(self, key: str, n: int = 1) -> None:
        if n:
            self.counts[key] = self.counts.get(key, 0) + n

    # ---- the decision -------------------------------------------------------------------------

    def decide(self) -> bool:
        """Fill ``renames`` / ``dropped`` / ``left`` / ``refused`` from the stored record."""
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        content = meta.get("content") or {}
        speakers = [s for s in content.get("speakers") or [] if isinstance(s, dict)]
        rel = str(content.get("transcript_file_path") or "")
        diag = (
            _load(self.meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
            if rel.endswith(".txt")
            else None
        )
        hosts = [str(h) for h in ((diag or {}).get("tried") or {}).get("known_hosts") or [] if h]
        placed = [s for s in speakers if s.get("voices") and isinstance(s.get("name"), str)]
        unplaced_lower = {
            str(s.get("name")).lower()
            for s in speakers
            if not s.get("voices") and isinstance(s.get("name"), str)
        }
        role_of = {str(s["name"]): s.get("role") for s in placed}
        claimed: Dict[str, str] = {}
        for s in placed:
            name = str(s["name"])
            if not hosts or name in hosts:
                continue
            taken = {str(o["name"]) for o in placed if o is not s}
            target = host_spelling(name, s.get("role"), hosts, taken)
            if target is None:
                continue
            # Only what this fix changes: a title, or the host listed beside the voice.
            if not (_has_title(name) or target.lower() in unplaced_lower):
                continue
            if target in role_of and role_of[target] != s.get("role") and role_of[target]:
                self.refused = f"{name!r} would take {target!r}, held in another role"
                return True
            if claimed.setdefault(target, name) != name:
                self.refused = f"{target!r} would be claimed by two voices"
                return True
            self.renames[name] = target
        kept: List[str] = []
        for s in speakers:
            raw = s.get("name")
            if not isinstance(raw, str):
                continue
            final = self.renames.get(raw, raw)
            match = next((k for k in kept if same_person(final, k)), None)
            if match is None:
                kept.append(final)
            elif not s.get("voices"):
                self.dropped.append(raw)
            elif final != match:
                self.left.append((match, final))
        return bool(self.renames or self.dropped or self.left)

    # ---- the rewrite --------------------------------------------------------------------------

    def plan(self) -> bool:
        """Decide, then build every rewritten file in memory. False when nothing is written."""
        if not self.decide() or self.refused:
            return False
        if not (self.renames or self.dropped):
            return False
        meta = _load(self.meta) or {}
        rel = str((meta.get("content") or {}).get("transcript_file_path") or "")
        transcripts = _Transcripts(self.meta, self.root, self.renames)
        if self.renames and rel.endswith(".txt"):
            try:
                transcripts._in_place(self.meta.parent.parent, rel)
            except Refused as exc:
                self.refused = f"transcripts: {exc}"
                return False
        ids = self._id_map()
        ep = _RenameEpisode(self.meta)
        if not ep.load():
            return False
        # The transcript step moved offsets in the GI and ad-free segments; rename on top of those.
        for path, payload in transcripts.json_files.items():
            ep.files[path] = payload
        ep.rewrite(self.renames, ids)
        self._node_labels(ep)
        promoted = self._hosts_promoted(meta)
        self._record(ep.files[self.meta], promoted)
        self._segment_roles(ep, promoted)
        self._kg_roles(ep, promoted, ids)
        for k, v in ep.counts.items():
            self._bump(k, v)
        for path in ep.changed:
            self.json_files[path] = ep.files[path]
        for path, text in transcripts.text_files.items():
            self.text_files[path] = text
        for k, v in transcripts.counts.items():
            self._bump(k, v)
        if self.renames and rel.endswith(".txt"):
            try:
                self._turns(ep, rel)
            except Refused as exc:
                self.json_files, self.text_files = {}, {}
                self.refused = f"turns: {exc}"
                return False
        self._diagnostics(rel, promoted)
        self._context(ep)
        return bool(self.json_files or self.text_files)

    def _node_labels(self, ep: _RenameEpisode) -> None:
        """A Person node's ``label`` is its display name too; m0017 renames ``name`` only."""
        for path, payload in ep.files.items():
            if not path.name.endswith((".kg.json", ".gi.json")):
                continue
            for node in _person_nodes(payload):
                props = node.get("properties")
                if isinstance(props, dict) and props.get("label") in self.renames:
                    props["label"] = self.renames[props["label"]]
                    self._bump("node_labels_rewritten")

    def _turns(self, ep: _RenameEpisode, rel: str) -> None:
        """Rebuild each variant's ``turns.json`` (RFC-123) from the rewritten text and segments.

        Its rows hold the speaker label AND character offsets into the transcript, so a rename
        that changes a prefix's length makes the old file wrong twice. The pipeline's own builder
        is used, which returns nothing unless the text is exactly the render of the segments:
        then the episode is refused rather than left with turns that index the old text.
        """
        run = self.meta.parent.parent
        for suffix in (".txt", ".adfree.txt"):
            text_path = run / rel.replace(".txt", suffix)
            turns_path = Path(str(text_path)[: -len(".txt")] + TURNS_SUFFIX)
            old = _load(turns_path)
            if not isinstance(old, dict):
                continue
            seg_path = Path(str(text_path)[: -len(".txt")] + ".segments.json")
            segs = ep.files.get(seg_path)
            rows = segs if isinstance(segs, list) else (segs or {}).get("segments")
            text = self.text_files.get(text_path)
            if text is None:
                text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
            raw_source = old.get("source")
            source: Dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
            sha = source.get("segments_sha256")
            if seg_path in ep.changed:
                sha = hashlib.sha256(dump_json(segs).encode("utf-8")).hexdigest()
            doc = build_turns_document(
                text,
                [r for r in rows or [] if isinstance(r, dict)],
                rel_transcript_path=str(source.get("transcript_ref") or ""),
                episode_slug=old.get("episode_slug"),
                language=old.get("language"),
                segments_sha256=sha,
            )
            if doc is None:
                raise Refused(f"{turns_path.name} no longer rebuilds from its text")
            if len(doc.get("turns") or []) != len(old.get("turns") or []):
                raise Refused(f"{turns_path.name} would change its number of turns")
            if doc != old:
                # The pipeline's own serialisation (turns_artifact.write_turns_artifact).
                self.text_files[turns_path] = json.dumps(doc, indent=0, allow_nan=False)
                self._bump("turns_rebuilt")

    def _id_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for old, new in self.renames.items():
            try:
                a, b = person_id(old), person_id(new)
            except ValueError:
                continue
            if a != b and not is_scoped_person_id(a):
                out[a] = b
        return out

    def _hosts_promoted(self, meta: dict) -> Set[str]:
        """New names that become the host where the record had the voice as a guest."""
        out: Set[str] = set()
        for s in (meta.get("content") or {}).get("speakers") or []:
            if isinstance(s, dict) and s.get("name") in self.renames and s.get("role") != "host":
                out.add(self.renames[s["name"]])
        return out

    def _record(self, meta: dict, promoted: Set[str]) -> None:
        """``content.speakers``: renamed (by m0017 already), merged, de-duplicated, renumbered."""
        content = meta.get("content") or {}
        entries = [s for s in content.get("speakers") or [] if isinstance(s, dict)]
        out: List[dict] = []
        for s in entries:
            name = s.get("name")
            if not isinstance(name, str):
                out.append(s)
                continue
            if name in promoted and s.get("voices"):
                s["role"] = "host"
                self._bump("roles_promoted")
            twin = next(
                (
                    o
                    for o in out
                    if isinstance(o.get("name"), str) and same_person(name, str(o["name"]))
                ),
                None,
            )
            if twin is None:
                out.append(s)
            elif not s.get("voices"):
                self._bump("unplaced_dropped")
            elif twin.get("voices") and twin.get("name") == name:
                twin["voices"] = list(dict.fromkeys(list(twin["voices"]) + list(s["voices"])))
                self._bump("placed_merged")
            else:
                out.append(s)  # two placed voices, two names: left to the roster
        hosts = [s for s in out if s.get("voices") and s.get("role") == "host"]
        guests = [s for s in out if s.get("voices") and s.get("role") != "host"]
        unplaced = [s for s in out if not s.get("voices")]
        for group, stem in ((hosts, "host"), (guests, "guest")):
            for i, s in enumerate(group):
                s["id"] = stem if len(group) == 1 else f"{stem}_{i + 1}"
        for i, s in enumerate(unplaced):
            s["id"] = f"unplaced_{i + 1}"
        content["speakers"] = hosts + guests + unplaced

    def _voices_of(self, meta: dict, names: Set[str]) -> Set[str]:
        return {
            str(v)
            for s in (meta.get("content") or {}).get("speakers") or []
            if isinstance(s, dict) and s.get("name") in names
            for v in s.get("voices") or []
        }

    def _segment_roles(self, ep: _RenameEpisode, promoted: Set[str]) -> None:
        if not promoted:
            return
        for path, payload in ep.files.items():
            if not path.name.endswith(".segments.json"):
                continue
            rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
            for row in rows or []:
                if (
                    isinstance(row, dict)
                    and row.get("speaker_label") in promoted
                    and row.get("speaker_role") not in (None, "host")
                ):
                    row["speaker_role"] = "host"
                    self._bump("segment_roles_promoted")

    def _kg_roles(self, ep: _RenameEpisode, promoted: Set[str], ids: Dict[str, str]) -> None:
        if not promoted:
            return
        wanted = set()
        for name in promoted:
            try:
                wanted.add(person_id(name))
            except ValueError:
                continue
        for path, payload in ep.files.items():
            if not path.name.endswith(".kg.json"):
                continue
            for node in _person_nodes(payload):
                props = node.get("properties")
                if (
                    node["id"] in wanted
                    and isinstance(props, dict)
                    and props.get("role") == "guest"
                ):
                    props["role"] = "host"
                    self._bump("kg_roles_promoted")

    def _diagnostics(self, rel: str, promoted: Set[str]) -> None:
        if not rel.endswith(".txt") or not self.renames:
            return
        path = self.meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json")
        diag = _load(path)
        if not isinstance(diag, dict):
            return
        hit = 0
        for v in diag.get("voices") or []:
            if isinstance(v, dict) and v.get("named") and v.get("resolved_name") in self.renames:
                v["resolved_name"] = self.renames[v["resolved_name"]]
                if v["resolved_name"] in promoted:
                    v["role"] = "host"
                hit += 1
        if hit:
            self.json_files[path] = diag
            self._bump("diagnostics_voices_renamed", hit)

    def _context(self, ep: _RenameEpisode) -> None:
        base = str(self.meta)[: -len(".metadata.json")]
        path = Path(base + ".context.json")
        ctx = _load(path)
        if not isinstance(ctx, dict):
            return
        rebuilt = build_context_digest(
            str(ctx.get("episode_id") or ""),
            gi_artifact=ep.files.get(Path(base + ".gi.json")),
            kg_artifact=ep.files.get(Path(base + ".kg.json")),
            metadata=ep.files[self.meta],
        )
        hit = False
        for parent, key in _CONTEXT_FIELDS:
            old = ctx.get(parent) if parent else ctx
            new = rebuilt.get(parent) if parent else rebuilt
            if isinstance(old, dict) and isinstance(new, dict) and key in new:
                if old.get(key) != new[key]:
                    old[key] = new[key]
                    hit = True
        if hit:
            self.json_files[path] = ctx
            self._bump("context_rebuilt")


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class OnePersonOneEntryMigration(Migration):
    """One person, one entry: titled and respelt names take the stated host's; duplicates go."""

    id = MIGRATION_ID
    to_version = "2.7.19"
    description = (
        "one person is one entry per episode: a titled or respelt voice name takes the stated "
        "host's spelling and role, and a duplicate unplaced entry is dropped, across metadata, "
        "segments, KG, GI, bridge, diagnostics, context and transcripts (offsets shifted)"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], List[_Episode], Dict[str, int]]:
        """``(episodes to write, episodes refused or left, totals)``."""
        write: List[_Episode] = []
        other: List[_Episode] = []
        totals: Dict[str, int] = {}
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, root)
            if ep.plan():
                write.append(ep)
            if ep.refused or ep.left:
                other.append(ep)
            if ep.refused:
                ep._bump("refused")
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return write, other, totals

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        write, other, totals = self._scan(ctx.corpus_root)
        return (
            f"one person one entry plan: {len(write)} episode(s), {len(other)} refused/left; "
            + (", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Nothing left to write and nothing refused. Two placed voices left are reported only."""
        write, other, _totals = self._scan(ctx.corpus_root)
        refused = [ep for ep in other if ep.refused]
        if write or refused:
            names = [ep.meta.name for ep in write + refused]
            return False, (
                f"{len(write)} episode(s) still list one person twice, {len(refused)} refused: "
                f"{names[:5]}"
            )
        left = [ep for ep in other if ep.left]
        return True, (
            "no served episode lists one person twice"
            + (f" (except {len(left)} with two placed voices left to the roster)" if left else "")
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write every rewritten file of every episode; back up each, receipt it."""
        root = ctx.corpus_root
        write, other, totals = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in write:
                for path, payload in sorted(ep.json_files.items()):
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                for path, text in sorted(ep.text_files.items()):
                    receipts.append(_write_text_with_backup(root, path, text, BACKUP_TAG))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(write)} episode(s); "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "episodes": [
                    {
                        "meta": str(ep.meta.relative_to(root)),
                        "renames": ep.renames,
                        "dropped": ep.dropped,
                    }
                    for ep in write
                ],
                "refused": [
                    {"meta": str(ep.meta.relative_to(root)), "why": ep.refused}
                    for ep in other
                    if ep.refused
                ],
                "left": [
                    {"meta": str(ep.meta.relative_to(root)), "pairs": ep.left}
                    for ep in other
                    if ep.left and not ep.refused
                ],
                "totals": totals,
                "files_written": len(receipts),
            },
        )


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write, refuse or leave."""
    m = OnePersonOneEntryMigration()
    write, other, _totals = m._scan(Path(root))
    for ep in write + [o for o in other if o not in write]:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "renames": ep.renames,
            "dropped": ep.dropped,
            "left": ep.left,
            "refused": ep.refused,
            "files": sorted(
                str(p.relative_to(root)) for p in list(ep.json_files) + list(ep.text_files)
            ),
            "counts": ep.counts,
        }
