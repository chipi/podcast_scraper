"""0016 — a name published on a voice the corpus's ad signatures call an ad is nobody's voice.

A diarized voice that reads an ad was seated with a person's name: Daniel Atkinson (Wirecutter) on
dozens of episodes, Jonathan Knight (NYT Games), Shannon Maldonado (Shopify), the "Gemini" /
"Claude" promos, Vox's promo voice, a host's name on a German ad. The pipeline now classifies such
a voice ``commercial`` from ``search/ad_signatures.json`` (``AdSignatures.is_ad_voice``); this
removes what is already on disk — no LLM, no reprocess.

A HIT is ``(episode metadata relpath, voice id, name)``: a voice whose text (``.segments.json`` rows
grouped by the raw ``speaker`` id) the BUILT signatures classify as an ad, carrying a name — a
placed ``content.speakers`` entry listing the voice in ``voices``, or segment rows of that voice
with a ``speaker_label``.

WHAT EACH SURFACE BECOMES — the same five as m0012, with ``voice_type: commercial`` where m0012
writes ``unknown`` (this voice is not a person we failed to name; it is an ad):

* VOICE level, always: segment rows of the hit voice lose ``speaker_label`` (raw ``SPEAKER_NN``
  stays) and become ``voice_type: commercial`` in ``.segments.json`` and ``.adfree.segments.json``;
  the voice id leaves the name's roster entry ``voices``.
* NAME level, only when the hit voices were EVERY voice carrying that name in the episode (a host
  whose name also sits on their real voice keeps it — only the ad voice loses it): the roster entry
  and ``detected_hosts`` / ``detected_guests`` entry, the KG Person node + edges, the GI Person
  node, ``SPOKEN_BY``, quote ``speaker_id`` / insight ``speaker`` (m0012's rewrite, then
  ``unknown`` becomes ``commercial``) and the bridge identity row. GI/KG carry no voice id, so a
  name that survives on another voice keeps its quotes, including any the ad voice contributed.

THE SET IS FROZEN. The hits are decided once, at apply time, from the built signatures and written
to the receipt header with the file's ``built_at`` and SHA-256 (it is rebuilt every 6h; a later
rebuild must not change what was applied). ``verify`` judges against the frozen hits only; before
any apply there is none, so it judges the live hits.

STATED NAMES ARE NEVER HITS. A host reads the same promo and outro every week, so a host's own
voice text recurs across shows. A name is skipped when a source stated it for the episode (feed
hosts in the diagnostics' ``tried.known_hosts``, ``detected_hosts`` / ``detected_guests``,
``metadata_named``, unbound names, a roster entry that is unplaced or sourced from a statement): the
target is a name NOBODY stated sitting on an ad voice, an outside person reading the ad. A label
that is a raw voice id (``SPEAKER_03``) or a role word is not a name at all.

LANGUAGE-ONLY hits — flagged by the language rule (``lang != episode_language and lang in
ad_languages``), not by recurrence — are the ones that can be content (a Spanish clip on a Latin
America show). They are EXCLUDED by default, listed under ``language_only`` for hand review, and
applied only with ``ctx.options["include_language_only"]`` or ``M0016_INCLUDE_LANGUAGE_ONLY=1``.

No ``search/ad_signatures.json`` → a clean no-op: nothing is written.

UNDO. Same as m0012: every file is copied under ``.podcast_scraper/upgrade-backups/0016/`` first;
``undo`` restores a file only while it is still exactly what this migration wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...graph_id_utils import is_bare_speaker_label
from ...kg.speaker_coherence import same_person
from ...providers.ml.diarization.ad_signatures import (
    AdSignatures,
    episode_language,
    FILENAME as SIGNATURES_FILENAME,
    SCHEMA_VERSION,
    VOICE_RECUR_FRACTION,
    words,
)
from ...speaker_detectors.entity_kind_votes import kind_key
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, read_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0012_org_speakers_removed import _Episode, _load, _segment_files, _SPEAKING

MIGRATION_ID = "0016_ad_reader_names_removed"
RECEIPTS_FILE = "ad_reader_names_removed.jsonl"
BACKUP_TAG = "0016"
INCLUDE_LANGUAGE_ONLY_ENV = "M0016_INCLUDE_LANGUAGE_ONLY"
NO_SIGNATURES_MESSAGE = "ad_signatures.json not built yet — run after the first finalize / A4"
COMMERCIAL = "commercial"

Hit = Tuple[str, str, str]  # (metadata relpath, voice id, name)


def _include_language_only(ctx: MigrationContext) -> bool:
    return bool(ctx.options.get("include_language_only")) or (
        os.environ.get(INCLUDE_LANGUAGE_ONLY_ENV) == "1"
    )


def _load_signatures(root: Path) -> Optional[Tuple[AdSignatures, Dict[str, str]]]:
    """The corpus's own built signatures + their identity, or ``None``.

    Not ``load_near``: that walks UP from the directory and could return another corpus's file.
    The bytes are read once so the recorded SHA is of exactly what was parsed.
    """
    path = Path(root) / "search" / SIGNATURES_FILENAME
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        return None
    sigs = AdSignatures(
        recurring=frozenset(int(h) for h in doc.get("recurring") or []),
        ad_languages=frozenset(doc.get("ad_languages") or []),
    )
    identity = {
        "built_at": str(doc.get("built_at") or ""),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return sigs, identity


def _rows(raw: Any) -> List[dict]:
    rows = raw if isinstance(raw, list) else (raw or {}).get("segments")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _voice_texts(rows: Iterable[dict]) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = {}
    for row in rows:
        if row.get("speaker"):
            grouped.setdefault(str(row["speaker"]), []).append(str(row.get("text") or ""))
    return {voice: " ".join(parts) for voice, parts in grouped.items()}


#: Roster sources that are NOT the voice's own doing: a name the feed, the show notes or the
#: pre-listening hint stated, matched to a voice afterwards. ``self_intro`` / ``llm_resolution`` /
#: ``raw`` are the voice (or a model reading it) naming itself — those stay candidates.
_STATED_SOURCES = frozenset(
    {"known_hosts", "feed", "guest", "feed_statement", "episode_metadata", "hint", "metadata"}
)


def _strings(values: Any) -> List[str]:
    return [str(v).strip() for v in values or [] if isinstance(v, str) and v.strip()]


def _stated_names(meta: Path, content: dict) -> List[str]:
    """Every person this episode's sources STATED, as opposed to a name a voice earned.

    A host reads the same promo and outro every week, so their own voice text recurs across shows;
    recurrence alone cannot tell them from an outside ad reader. What can: nobody stated the
    reader. Sources, from what the metadata really carries: the diagnostics' ``tried.known_hosts``
    / ``detected_guests`` / ``metadata_named`` and ``summary.unbound_names``, the metadata's own
    ``detected_hosts`` / ``detected_guests``, and ``content.speakers`` entries that are unplaced or
    whose ``source`` is a stated one.
    """
    stated = _strings(content.get("detected_hosts")) + _strings(content.get("detected_guests"))
    for entry in content.get("speakers") or []:
        if isinstance(entry, dict) and (
            entry.get("placed") is False or entry.get("source") in _STATED_SOURCES
        ):
            stated += _strings([entry.get("name")])
    rel_tx = str(content.get("transcript_file_path") or "")
    diagnostics = (
        _load(meta.parent.parent / rel_tx.replace(".txt", ".speakers.diagnostics.json"))
        if rel_tx
        else None
    )
    if isinstance(diagnostics, dict):
        tried = diagnostics.get("tried") or {}
        for key in ("known_hosts", "detected_guests", "metadata_named"):
            stated += _strings(tried.get(key))
        stated += _strings((diagnostics.get("summary") or {}).get("unbound_names"))
    return stated


def _episode_hits(
    meta: Path, rel: str, sigs: AdSignatures
) -> Tuple[List[Hit], List[Hit], List[Hit]]:
    """``(recurrence hits, language-only hits, excluded because stated)`` for one episode."""
    payload = _load(meta)
    if not isinstance(payload, dict):
        return [], [], []
    content = payload.get("content") or {}
    rel_tx = str(content.get("transcript_file_path") or "")
    if not rel_tx:
        return [], [], []
    raw = _load(meta.parent.parent / rel_tx.replace(".txt", ".segments.json"))
    rows = _rows(raw)
    texts = _voice_texts(rows)
    if not texts:
        return [], [], []
    language = episode_language(texts)

    names: Dict[str, Set[str]] = {}
    for entry in content.get("speakers") or []:
        if not isinstance(entry, dict) or entry.get("placed") is False:
            continue
        name = str(entry.get("name") or "").strip()
        if name and entry.get("role") in _SPEAKING:
            for voice in entry.get("voices") or []:
                names.setdefault(str(voice), set()).add(name)
    for row in rows:
        label = str(row.get("speaker_label") or "").strip()
        if label and row.get("speaker"):
            names.setdefault(str(row["speaker"]), set()).add(label)

    stated: Optional[List[str]] = None
    recurrence: List[Hit] = []
    language_only: List[Hit] = []
    excluded: List[Hit] = []
    for voice, text in sorted(texts.items()):
        if voice not in names or not sigs.is_ad_voice(text, language):
            continue
        by_recurrence = sigs.recurring_fraction(words(text)) >= VOICE_RECUR_FRACTION
        for name in sorted(names[voice]):
            if is_bare_speaker_label(name):
                continue
            if stated is None:
                stated = _stated_names(meta, content)
            if any(same_person(name, s) for s in stated):
                excluded.append((rel, voice, name))
                continue
            (recurrence if by_recurrence else language_only).append((rel, voice, name))
    return recurrence, language_only, excluded


def find_hits(root: Path, sigs: AdSignatures) -> Tuple[List[Hit], List[Hit], List[Hit]]:
    """``(recurrence, language-only, stated-and-excluded)`` hits over every served episode."""
    root = Path(root)
    recurrence: List[Hit] = []
    language_only: List[Hit] = []
    excluded: List[Hit] = []
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        r, lo, ex = _episode_hits(meta, str(meta.relative_to(root)), sigs)
        recurrence += r
        language_only += lo
        excluded += ex
    return recurrence, language_only, excluded


def _count_by_name(hits: Iterable[Hit]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _rel, _voice, name in hits:
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def dry_run_report(root: Path, sigs: AdSignatures) -> Dict[str, Any]:
    """Read-only: what m0016 would act on, by name, for ``root`` under ``sigs``.

    ``hits_by_name`` is what apply would remove; ``language_only_by_name`` is held for hand review;
    ``excluded_stated_by_name`` are names on ad-classified voices that a source stated (hosts, named
    guests), kept. Counts are voice hits (one per episode and voice).
    """
    recurrence, language_only, excluded = find_hits(Path(root), sigs)
    return {
        "hits": len(recurrence),
        "episodes": len({rel for rel, _v, _n in recurrence}),
        "hits_by_name": _count_by_name(recurrence),
        "language_only": len(language_only),
        "language_only_by_name": _count_by_name(language_only),
        "excluded_stated": len(excluded),
        "excluded_stated_by_name": _count_by_name(excluded),
    }


class _VoiceEpisode(_Episode):
    """m0012's five-surface rewrite, narrowed to the voices of the hits and typed ``commercial``."""

    def __init__(self, meta: Path, hits: Set[Tuple[str, str]]) -> None:
        super().__init__(meta, set())
        self.hits = hits  # {(voice id, name)}

    def _carriers(self, content: dict, seg_rows: List[List[dict]]) -> Dict[str, Set[str]]:
        """``{kind_key(name): voices carrying it}`` for each hit name, before any rewrite."""
        out: Dict[str, Set[str]] = {kind_key(n): set() for _v, n in self.hits}
        for entry in content.get("speakers") or []:
            key = kind_key(str((entry or {}).get("name") or ""))
            if key in out:
                out[key] |= {str(v) for v in (entry.get("voices") or [])}
        for rows in seg_rows:
            for row in rows:
                key = kind_key(str(row.get("speaker_label") or ""))
                if key in out and row.get("speaker"):
                    out[key].add(str(row["speaker"]))
        return out

    def plan(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        content = meta.get("content") or {}
        seg_paths = _segment_files(self.meta, content)
        before_rows = [_rows(_load(p)) for p in seg_paths]
        carriers = self._carriers(content, before_rows)
        hit_voices: Dict[str, Set[str]] = {}
        for voice, name in self.hits:
            hit_voices.setdefault(kind_key(name), set()).add(voice)
        whole = {k for k, vs in carriers.items() if vs <= hit_voices[k]}
        partial = set(hit_voices) - whole

        gi_before = _node_voice_types(_load(_sibling_gi(self.meta)))
        marked = [
            [i for i, r in enumerate(rows) if kind_key(str(r.get("speaker_label") or "")) in whole]
            for rows in before_rows
        ]
        self.orgs = whole
        if whole:
            super().plan()
            self._commercial_after_whole(seg_paths, marked, gi_before)
        if partial:
            self._partial(seg_paths, partial, hit_voices)
        return bool(self.changed)

    def _commercial_after_whole(
        self, seg_paths: List[Path], marked: List[List[int]], gi_before: Dict[str, Any]
    ) -> None:
        for path, idxs in zip(seg_paths, marked):
            if path in self.files and idxs:
                rows = _rows(self.files[path])
                for i in idxs:
                    rows[i]["voice_type"] = COMMERCIAL
        gi_path = _sibling_gi(self.meta)
        gi = self.files.get(gi_path)
        for node in (gi or {}).get("nodes") or []:
            props = node.get("properties") or {}
            if (
                node.get("type") in ("Quote", "Insight")
                and props.get("speaker_voice_type") == "unknown"
                and gi_before.get(str(node.get("id"))) != "unknown"
            ):
                props["speaker_voice_type"] = COMMERCIAL

    def _partial(
        self, seg_paths: List[Path], partial: Set[str], hit_voices: Dict[str, Set[str]]
    ) -> None:
        """A name that survives on another voice loses only the hit voices, not the person."""
        for path in seg_paths:
            raw = self.files.get(path) or _load(path)
            hit = 0
            for row in _rows(raw):
                key = kind_key(str(row.get("speaker_label") or ""))
                if key in partial and str(row.get("speaker")) in hit_voices[key]:
                    row.pop("speaker_label", None)
                    row["voice_type"] = COMMERCIAL
                    hit += 1
            if hit:
                self.files[path] = raw
                self.changed.add(path)
                self._bump("segments_relabelled", hit)
        meta = self.files.get(self.meta) or _load(self.meta)
        trimmed = 0
        for entry in ((meta or {}).get("content") or {}).get("speakers") or []:
            key = kind_key(str((entry or {}).get("name") or ""))
            voices = entry.get("voices")
            if key in partial and isinstance(voices, list):
                left = [v for v in voices if str(v) not in hit_voices[key]]
                if len(left) != len(voices):
                    entry["voices"] = left
                    trimmed += len(voices) - len(left)
        if trimmed:
            self.files[self.meta] = meta
            self.changed.add(self.meta)
            self._bump("roster_voices_trimmed", trimmed)


def _sibling_gi(meta: Path) -> Path:
    return meta.with_name(meta.name[: -len(".metadata.json")] + ".gi.json")


def _node_voice_types(gi: Any) -> Dict[str, Any]:
    return {
        str(n.get("id")): (n.get("properties") or {}).get("speaker_voice_type")
        for n in (gi or {}).get("nodes") or []
        if isinstance(n, dict)
    }


def _by_episode(hits: Iterable[Iterable[str]]) -> Dict[str, Set[Tuple[str, str]]]:
    out: Dict[str, Set[Tuple[str, str]]] = {}
    for rel, voice, name in hits:
        out.setdefault(rel, set()).add((voice, name))
    return out


def _episodes(root: Path, hits: Iterable[Iterable[str]]) -> Iterable[_VoiceEpisode]:
    for rel, voice_names in sorted(_by_episode(hits).items()):
        ep = _VoiceEpisode(Path(root) / rel, voice_names)
        if ep.plan():
            yield ep


def _read_receipts(root: Path) -> Tuple[Dict[str, Any], List[dict]]:
    """``(header, rows)``; the header carries the frozen hits."""
    return read_receipts(root, RECEIPTS_FILE)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


def _totals(episodes: Iterable[_VoiceEpisode]) -> Tuple[int, Dict[str, int]]:
    totals: Dict[str, int] = {}
    n = 0
    for ep in episodes:
        n += 1
        for k, v in ep.counts.items():
            totals[k] = totals.get(k, 0) + v
    return n, totals


class AdReaderNamesRemovedMigration(Migration):
    """Remove names published on voices the corpus's ad signatures classify as ads."""

    id = MIGRATION_ID
    to_version = "2.7.10"
    description = (
        "A published speaker name on a voice the corpus's built ad signatures call an ad "
        "(cross-show recurring script) is nobody's voice. Remove it from the roster, segment "
        "labels (voice_type commercial), KG host/guest nodes, GI Person/SPOKEN_BY/quote/insight "
        "attribution and bridge identities; language-only hits are listed, not applied"
    )

    def _chosen(
        self, ctx: MigrationContext
    ) -> Optional[Tuple[List[Hit], List[Hit], Dict[str, str]]]:
        loaded = _load_signatures(ctx.corpus_root)
        if loaded is None:
            return None
        sigs, identity = loaded
        recurrence, language_only, _stated = find_hits(ctx.corpus_root, sigs)
        chosen = recurrence + (language_only if _include_language_only(ctx) else [])
        return chosen, language_only, identity

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        found = self._chosen(ctx)
        if found is None:
            return NO_SIGNATURES_MESSAGE
        chosen, language_only, _identity = found
        n, totals = _totals(_episodes(ctx.corpus_root, chosen))
        held = "" if _include_language_only(ctx) else f"; {len(language_only)} language_only hit(s)"
        return (
            f"ad reader plan: {len(chosen)} hit(s) {_names(chosen)}; {n} episode(s); "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
            + held
            + (f" held for hand review: {_names(language_only)}" if held else "")
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface still names a FROZEN hit's name on its voice. ``(ok, message)``."""
        header, _rows = _read_receipts(ctx.corpus_root)
        frozen = header.get("hits")
        if not frozen:
            found = self._chosen(ctx)
            if found is None:
                return True, NO_SIGNATURES_MESSAGE
            chosen = found[0]
            if chosen:
                return False, f"{len(chosen)} ad-reader hit(s) not applied yet: {_names(chosen)}"
            return True, "no receipts and no ad-voice hit — nothing to remove"
        left = [f"{ep.meta.name}: {sorted(ep.counts)}" for ep in _episodes(ctx.corpus_root, frozen)]
        if left:
            return False, f"{len(left)} episode(s) still name an ad voice: {left[:5]}"
        return True, f"no served surface names an ad voice for any of {len(frozen)} frozen hit(s)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Remove the frozen hits from every surface; back up, receipt, own."""
        root = ctx.corpus_root
        found = self._chosen(ctx)
        if found is None:
            return MigrationResult(
                self.id, applied=True, dry_run=ctx.dry_run, message=NO_SIGNATURES_MESSAGE
            )
        chosen, language_only, identity = found
        touched: List[str] = []
        receipts: List[dict] = []
        totals: Dict[str, int] = {}
        for ep in _episodes(root, chosen):
            touched.append(str(ep.meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if ctx.dry_run:
                continue
            for path in sorted(ep.changed):
                receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
        if not ctx.dry_run:
            prior, _ = _read_receipts(root)
            hits = [list(h) for h in prior.get("hits") or []]
            hits += [list(h) for h in chosen if list(h) not in hits]
            append_receipts(
                root,
                RECEIPTS_FILE,
                {
                    "signatures": identity,
                    "hits": hits,
                    "language_only": [list(h) for h in language_only],
                    "language_only_applied": _include_language_only(ctx),
                },
                receipts,
            )
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        message = (
            f"{verb} {len(touched)} episode(s) for {len(chosen)} ad-voice hit(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
            + f"; {len(language_only)} language_only hit(s) "
            + ("included" if _include_language_only(ctx) else "EXCLUDED — hand review")
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "signatures": identity,
                "hits": [list(h) for h in chosen],
                "language_only": [list(h) for h in language_only],
                "episodes": len(touched),
                "totals": totals,
                "files_written": len(receipts),
                "episodes_sample": touched[:20],
            },
        )


def _names(hits: Iterable[Hit]) -> List[str]:
    return sorted({name for _rel, _voice, name in hits})
