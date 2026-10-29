"""0030 — served episodes carry today's ad cut, every offset into the ad-free text carried across.

#2360, and the ad-cut changes before it (#2168 post- and mid-rolls, the cross-promo cue), changed
what ``build_adfree_artifacts`` cuts. New episodes get today's cut; served ones keep the ad-free
text whichever older code wrote it: a cold open cut as a pre-roll that today's code keeps, or a
sponsor read today's code removes. Measured on the 2026-10-15 prod snapshot, of 2,685 served
episodes: 421 recut, 304 whose ad-map alone differs (``source_length``, after the speaker
migrations re-rendered the raw text), the rest already today's cut. Every episode, not a list: a
served transcript that still carries a cut today's code would not make is the same defect
whichever change it predates.

The cut is a pure function of the stored raw ``<base>.txt`` and ``<base>.segments.json`` (prod's
``feeds.spec.yaml`` sets no ``crosspromo_cue_patterns``, so the built-ins are the pipeline's), so
the three ad-free sidecars are rebuilt with the pipeline's own builder and serialisation. What
locates text in the ad-free file BY CHARACTER OFFSET is carried across, never re-derived:

* GI ``Quote`` nodes whose ``transcript_ref`` names the ad-free file (ids kept: other files
  reference them);
* ``.adfree.segments.json`` rows (rebuilt, so exact);
* ``.adfree.turns.json`` (rebuilt with the pipeline's builder).

Each segment that survives both cuts is matched old -> new by its time and text, and a quote moves
with the segment it sits in (m0023's map and mover). A quote in text today's cut removes is dropped
with its edges, as the pipeline drops an ad-span quote at build. What the LLM stages built from
that text is cut out by hand, never regenerated: an insight every quote of which is now ad text,
and a named entity (organisation, object, mentioned person) whose name occurs in the cut text and
nowhere in today's ad-free text, go from GI and KG with their edges; the bridge drops identities
left in neither, and the metadata counts and context digest follow. Read on the snapshot: 52 of
the 54 insights removed restate a sponsor read (Felix by Rogo, Ramp's "small advantages
compound", Buildkite), and of the 174 entity nodes removed all but 3 are sponsors or show promos.
The 2 insights and 3 nodes that are real talk lost their only text to a segment shared with an ad.
A host's or guest's node is never removed. Text today's cut
keeps that an older cut removed has no insights; adding them needs the LLM, so it is left.

A quote already off its own text is placed on the one occurrence of it, as enrich-edges does at
read time; one whose text is not there exactly once is moved with its place and counted, no worse
than it was. Refuses an episode rather than guess: a kept segment whose stored values would change,
or a quote that would not slice to the same speech. Search follows on the next pipeline run, with no
rebuild: the incremental index upsert re-embeds an episode whose chunk text changed and prunes its
superseded rows (the dropped quotes), and enrichment re-derives its edges because GI and KG changed.

``verify`` fails while anything is left to write or refused. Undo restores from
``.podcast_scraper/upgrade-backups/0030/`` while each file is still what this migration wrote.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...builders.context_digest_builder import build_context_digest
from ...gi.pipeline import _char_range_to_ms
from ...gi.speakers import _reanchor_quote
from ...providers.ml.diarization.turns import TurnInvariantError
from ...workflow.adfree_transcript import build_adfree_artifacts
from ...workflow.turns_artifact import build_turns_document, TURNS_SUFFIX
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0023_transcript_speaker_prefixes_resynced import (
    _load,
    _move_quote,
    _OffsetMap,
    _rows,
    _write_text_with_backup,
    Refused,
)
from .m0025_one_person_one_entry import _CONTEXT_FIELDS, _run_relative

MIGRATION_ID = "0030_adfree_cut_resynced"
RECEIPTS_FILE = "adfree_cut_resynced.jsonl"
BACKUP_TAG = "0030"
_OFFSETS = ("char_start", "char_end")
#: Named entities whose only source can be a sponsor read; a Topic is the LLM's paraphrase, so its
#: label occurring in the text or not says nothing.
_NAMED = ("Organization", "Object", "Person")
_SPEAKING_ROLES = ("host", "guest")
#: The context digest's fields this migration can change: m0025's people fields, and the
#: organisations it removes as sponsors.
_CONTEXT_SYNCED = _CONTEXT_FIELDS + ((None, "companies"),)


def _key(row: Dict[str, Any]) -> Tuple[float, float, str]:
    return (
        round(float(row.get("start") or 0.0), 3),
        round(float(row.get("end") or 0.0), 3),
        str(row.get("text") or ""),
    )


def _spans(rows: List[dict]) -> List[Tuple[int, int]]:
    return [(int(r["char_start"]), int(r["char_end"])) for r in rows]


class _Episode:
    """One served episode's ad-free sidecars and GI quotes, rebuilt in memory."""

    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.json_files: Dict[Path, Any] = {}
        self.text_files: Dict[Path, str] = {}
        self.counts: Dict[str, int] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def plan(self) -> bool:
        meta = _load(self.meta)
        run = self.meta.parent.parent
        # Absolute ("/app/output/feeds/…") on older runs: joined as-is it names prod's path, not
        # this corpus's, and every such episode would be skipped off prod.
        rel = _run_relative(((meta or {}).get("content") or {}).get("transcript_file_path"), run)
        if not rel.endswith(".txt"):
            return False
        try:
            self._plan(run, rel)
        except Refused as exc:
            self.json_files, self.text_files = {}, {}
            self.counts = {"refused": 1, f"refused: {exc}": 1}
        return bool(self.json_files or self.text_files)

    def _plan(self, run: Path, rel: str) -> None:
        base = run / rel[: -len(".txt")]
        adfree_txt = Path(f"{base}.adfree.txt")
        raw_txt = Path(f"{base}.txt")
        if not adfree_txt.is_file() or not raw_txt.is_file():
            return
        art = build_adfree_artifacts(
            raw_txt.read_text(encoding="utf-8"),
            _rows(_load(Path(f"{base}.segments.json"))) or None,
        )
        if art is None:
            return
        old = adfree_txt.read_text(encoding="utf-8")
        admap_path = Path(f"{base}.adfree.admap.json")
        if art.text == old:
            if _load(admap_path) != art.ad_map:
                self.text_files[admap_path] = json.dumps(art.ad_map, indent=2)
                self._bump("admaps_rewritten")
            return
        seg_path = Path(f"{base}.adfree.segments.json")
        old_rows = _rows(_load(seg_path))
        new_rows = art.segments
        new_by_key = {_key(r): r for r in new_rows}
        kept_old: List[dict] = []
        kept_new: List[dict] = []
        dropped: List[Tuple[int, int]] = []
        for row in old_rows:
            twin = new_by_key.get(_key(row))
            if twin is None:
                dropped.append((int(row["char_start"]), int(row["char_end"])))
                continue
            # What the stored row says must survive; a value it never had (a plain transcript's
            # label, filled from the raw row today) loses nothing.
            if any(v is not None and twin.get(k) != v for k, v in row.items() if k not in _OFFSETS):
                raise Refused("a kept segment changed beyond its place")
            kept_old.append(row)
            kept_new.append(twin)
        if not kept_old:
            raise Refused("no segment survives both cuts")
        new_spans = _spans(kept_new)
        if new_spans != sorted(new_spans):
            raise Refused("the kept segments changed order")
        fmap = _OffsetMap(_spans(kept_old), new_spans)
        gi = self._quotes(adfree_txt.name, old, art.text, fmap, dropped, new_rows)
        cut_text = " ".join(str(r.get("text") or "") for r in old_rows if _key(r) not in new_by_key)
        if cut_text:
            self._surgery(gi, cut_text, art.text)
        seg_text = json.dumps(new_rows, indent=0, allow_nan=False)
        self.text_files[adfree_txt] = art.text
        self.text_files[seg_path] = seg_text
        self.text_files[admap_path] = json.dumps(art.ad_map, indent=2)
        self._turns(base, art.text, new_rows, seg_text)
        self._bump("episodes_recut")
        self._bump("segments_restored", len(new_rows) - len(kept_new))
        self._bump("segments_cut", len(dropped))

    def _quotes(
        self,
        name: str,
        old: str,
        new: str,
        fmap: _OffsetMap,
        dropped: List[Tuple[int, int]],
        new_rows: List[dict],
    ) -> Dict[str, Any]:
        gi_path = Path(str(self.meta)[: -len(".metadata.json")] + ".gi.json")
        gi: Dict[str, Any] = _load(gi_path) or {}
        self._supported_before = {
            e.get("from") for e in gi.get("edges") or [] if e.get("type") == "SUPPORTED_BY"
        }
        moved = 0
        cut: set = set()
        for node in gi.get("nodes") or []:
            props = (node.get("properties") or {}) if isinstance(node, dict) else {}
            ref = props.get("transcript_ref")
            # By file NAME: refs are run-relative on some episodes, absolute on others (m0023).
            if node.get("type") != "Quote" or not isinstance(ref, str) or Path(ref).name != name:
                continue
            cs, ce = props.get("char_start"), props.get("char_end")
            if not isinstance(cs, int) or not isinstance(ce, int):
                continue
            if props.get("text") != old[cs:ce]:
                # Already off before this cut (prod: all 73 quotes of one episode one character
                # early), so its offsets say nothing — checked before the ad test below, which
                # reads them. Placed on the one occurrence of its text, as enrich-edges does at
                # read time, and its times recomputed from there (they came from the same wrong
                # offsets: one segment early, ~10 s).
                text = str(props.get("text") or "")
                if _reanchor_quote(props, text, new) is not None and (
                    new[props["char_start"] : props["char_end"]] == text
                ):
                    if "timestamp_start_ms" in props:
                        ts = _char_range_to_ms(
                            new, props["char_start"], props["char_end"], new_rows
                        )
                        props["timestamp_start_ms"], props["timestamp_end_ms"] = ts
                    moved += 1
                    self._bump("quotes_reanchored")
                    continue
                # Not there exactly once ("the companies that", 6 Odd Lots episodes): no worse than
                # it was — moved with its place, counted — rather than holding the episode's cut.
                self._bump("quotes_already_off_left")
                if not any(cs < b and a < ce for a, b in dropped):
                    props["char_start"] = fmap(cs, is_end=False)
                    props["char_end"] = fmap(ce, is_end=True)
                    moved += 1
                    continue
            if any(cs < b and a < ce for a, b in dropped):
                # Ad text now: the pipeline drops such a quote, node and edges (gi/pipeline.py
                # `_is_advertisement_span`), and leaves its insight as it is.
                cut.add(node.get("id"))
                continue
            did_move, retexted = _move_quote(
                props, old, new, fmap(cs, is_end=False), fmap(ce, is_end=True)
            )
            moved += did_move
            if retexted:
                self._bump("quotes_retexted")
        if cut:
            gi["nodes"] = [
                n for n in gi["nodes"] if not (isinstance(n, dict) and n.get("id") in cut)
            ]
            gi["edges"] = [
                e
                for e in gi.get("edges") or []
                if not (isinstance(e, dict) and (e.get("from") in cut or e.get("to") in cut))
            ]
            self._bump("quotes_dropped_as_ad_text", len(cut))
        if moved or cut:
            self.json_files[gi_path] = gi
            self._bump("quotes_moved", moved)
        return gi

    def _surgery(self, gi: Dict[str, Any], cut_text: str, new: str) -> None:
        """Cut out what the LLM stages built from text today's cut removes (see the module doc)."""
        stem = str(self.meta)[: -len(".metadata.json")]
        gi_path, kg_path = Path(stem + ".gi.json"), Path(stem + ".kg.json")
        kg: Dict[str, Any] = _load(kg_path) or {}
        cut_l, new_l = cut_text.lower(), new.lower()

        def only_in_cut(node: Any) -> bool:
            props = (node.get("properties") or {}) if isinstance(node, dict) else {}
            name = str(props.get("name") or props.get("label") or "")
            if node.get("type") not in _NAMED or not name or props.get("role") in _SPEAKING_ROLES:
                return False
            rx = re.compile(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)")
            return bool(rx.search(cut_l)) and not rx.search(new_l)

        supported = {
            e.get("from") for e in gi.get("edges") or [] if e.get("type") == "SUPPORTED_BY"
        }
        gone = {
            n.get("id")
            for n in gi.get("nodes") or []
            if n.get("type") == "Insight"
            and n.get("id") in self._supported_before
            and n.get("id") not in supported
        }
        gone_entities = {
            n.get("id") for n in (gi.get("nodes") or []) + (kg.get("nodes") or []) if only_in_cut(n)
        }
        for path, doc, ids in ((gi_path, gi, gone | gone_entities), (kg_path, kg, gone_entities)):
            if not ids or not doc.get("nodes") or not any(n.get("id") in ids for n in doc["nodes"]):
                continue
            doc["nodes"] = [n for n in doc["nodes"] if n.get("id") not in ids]
            doc["edges"] = [
                e
                for e in doc.get("edges") or []
                if e.get("from") not in ids and e.get("to") not in ids
            ]
            self.json_files[path] = doc
        if not gone and not gone_entities:
            return
        self._bump("insights_removed_as_ad_text", len(gone))
        self._bump("entities_removed_as_ad_text", len(gone_entities))
        self._bridge(stem, gi, kg, gone_entities)
        self._metadata_and_context(stem, gi, kg)

    def _bridge(self, stem: str, gi: Dict[str, Any], kg: Dict[str, Any], gone: set) -> None:
        path = Path(stem + ".bridge.json")
        bridge = _load(path)
        if not isinstance(bridge, dict) or not gone:
            return
        in_gi = {n.get("id") for n in gi.get("nodes") or []}
        in_kg = {n.get("id") for n in kg.get("nodes") or []}
        out = []
        for ident in bridge.get("identities") or []:
            iid = ident.get("id") if isinstance(ident, dict) else None
            if iid in gone:
                if iid not in in_gi and iid not in in_kg:
                    self._bump("bridge_identities_removed")
                    continue
                src = ident.setdefault("sources", {})
                if isinstance(src, dict):
                    src["gi"], src["kg"] = iid in in_gi, iid in in_kg
            out.append(ident)
        bridge["identities"] = out
        self.json_files[path] = bridge

    def _metadata_and_context(self, stem: str, gi: Dict[str, Any], kg: Dict[str, Any]) -> None:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return
        gi_block, kg_block = meta.get("grounded_insights"), meta.get("knowledge_graph")
        if isinstance(gi_block, dict) and "insight_count" in gi_block:
            gi_block["insight_count"] = sum(
                1 for n in gi.get("nodes") or [] if n.get("type") == "Insight"
            )
        if isinstance(kg_block, dict) and kg.get("nodes") is not None:
            kg_block["node_count"] = len(kg.get("nodes") or [])
            kg_block["edge_count"] = len(kg.get("edges") or [])
        self.json_files[self.meta] = meta
        path = Path(stem + ".context.json")
        ctx = _load(path)
        if not isinstance(ctx, dict):
            return
        rebuilt = build_context_digest(
            str(ctx.get("episode_id") or ""), gi_artifact=gi, kg_artifact=kg, metadata=meta
        )
        for parent, key in _CONTEXT_SYNCED:
            old = ctx.get(parent) if parent else ctx
            fresh = rebuilt.get(parent) if parent else rebuilt
            if isinstance(old, dict) and isinstance(fresh, dict) and key in fresh:
                if old.get(key) != fresh[key]:
                    old[key] = fresh[key]
                    self.json_files[path] = ctx

    def _turns(self, base: Path, text: str, rows: List[dict], seg_text: str) -> None:
        turns_path = Path(f"{base}.adfree{TURNS_SUFFIX}")
        old = _load(turns_path)
        if not isinstance(old, dict):
            return
        raw_source = old.get("source")
        source: Dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        try:
            doc = build_turns_document(
                text,
                rows,
                rel_transcript_path=str(source.get("transcript_ref") or ""),
                episode_slug=old.get("episode_slug"),
                language=old.get("language"),
                segments_sha256=hashlib.sha256(seg_text.encode("utf-8")).hexdigest(),
            )
        except TurnInvariantError as exc:
            # The builder raises rather than return a bad document; one episode must not stop
            # the scan of every other one.
            raise Refused(f"{turns_path.name} breaks a turn invariant: {exc}") from exc
        if doc is None:
            raise Refused(f"{turns_path.name} does not rebuild from the new text")
        # The pipeline's own serialisation (turns_artifact.write_turns_artifact).
        self.text_files[turns_path] = json.dumps(doc, indent=0, allow_nan=False)
        self._bump("turns_rebuilt")


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class AdfreeCutResyncedMigration(Migration):
    """Rebuild served ad-free sidecars with today's cut, carrying every offset into them."""

    id = MIGRATION_ID
    to_version = "2.7.24"
    description = (
        "rebuild .adfree.txt/.segments/.admap/.turns with today's ad cut (#2360) and move GI "
        "quote offsets with the segments they sit in"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], Dict[str, int], List[str]]:
        eps: List[_Episode] = []
        totals: Dict[str, int] = {}
        refused: List[str] = []
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta)
            if ep.plan():
                eps.append(ep)
            if ep.counts.get("refused"):
                refused.append(str(meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return eps, totals, refused

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        eps, totals, _refused = self._scan(ctx.corpus_root)
        return f"ad-free cut plan: {len(eps)} episode(s); " + ", ".join(
            f"{k}={v}" for k, v in sorted(totals.items())
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Every served ad-free base is today's cut. ``(ok, message)``."""
        eps, _totals, refused = self._scan(ctx.corpus_root)
        if eps or refused:
            return False, f"{len(eps)} episode(s) left to rewrite, {len(refused)} refused"
        return True, "every served ad-free base is today's cut"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write every rebuilt sidecar and moved quote; back up each file, receipt it."""
        root = ctx.corpus_root
        eps, totals, refused = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in eps:
                for path, payload in ep.json_files.items():
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                for path, text in ep.text_files.items():
                    receipts.append(_write_text_with_backup(root, path, text, BACKUP_TAG))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(eps)} episode(s), refused {len(refused)}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "episodes": [str(ep.meta.relative_to(root)) for ep in eps[:60]],
                "refused": refused,
                "totals": totals,
                "files_written": len(receipts),
            },
        )
