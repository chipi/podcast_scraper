"""Corpus-wide episode identity helpers (GitHub #505).

Composite ``(feed_id, episode_id)`` scope keys avoid fingerprint and vector row
collisions when multiple feeds share a corpus parent.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

from podcast_scraper.utils import filesystem
from podcast_scraper.utils.path_validation import safe_resolve_directory

logger = logging.getLogger(__name__)


def normalize_feed_id(feed_id: Any) -> Optional[str]:
    """Return stripped feed id string, or None if missing."""
    if isinstance(feed_id, str) and feed_id.strip():
        return feed_id.strip()
    return None


def index_fingerprint_scope_key(feed_id: Optional[str], episode_id: str) -> str:
    """Stable key for ``episode_fingerprints.json`` (one row per scoped episode)."""
    fn = normalize_feed_id(feed_id)
    if fn:
        return f"{fn}\x1f{episode_id}"
    return episode_id


def vector_doc_scope_tag(feed_id: Optional[str], episode_id: str) -> str:
    """Segment embedded in the index doc ids for uniqueness across feeds."""
    if not feed_id:
        return episode_id
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in feed_id)[:120]
    return f"{safe}__{episode_id}"


def gi_map_lookup_key_from_vector_meta(meta: dict) -> str:
    """Key into the GI path map built from metadata (matches fingerprint scope)."""
    ep = meta.get("episode_id")
    if not isinstance(ep, str) or not ep:
        return ""
    return index_fingerprint_scope_key(normalize_feed_id(meta.get("feed_id")), ep)


def episode_root_from_metadata_path(metadata_path: Path) -> Path:
    """Episode workspace root: parent of ``metadata/``."""
    return metadata_path.parent.parent.resolve()


def feed_dir_and_run_segment_from_relpath(rel_posix: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``feeds/<feedDir>/run_<tag>/metadata/...`` into ``(feedDir, run_segment)``.

    Returns ``(None, None)`` when *rel_posix* is not under that layout (flat corpus, other
    trees, or ``feeds/...`` without a ``run_*`` segment before ``metadata/``).
    """
    rel = rel_posix.replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) < 5 or parts[0] != "feeds":
        return None, None
    if parts[2].startswith("run_") and parts[3] == "metadata":
        return parts[1], parts[2]
    return None, None


# Grouping key for corpora that are NOT under ``feeds/`` — see
# :func:`run_segment_from_flat_relpath`. Not a real feed dir, and deliberately a string that
# cannot collide with one (a feed dir is always a non-empty slug).
FLAT_CORPUS_FEED_KEY = ""


def run_segment_from_flat_relpath(rel_posix: str) -> Optional[str]:
    """Return the ``run_*`` segment of a FLAT ``run_<tag>/metadata/...`` path, else ``None``.

    Deliberately separate from :func:`feed_dir_and_run_segment_from_relpath` rather than folded
    into it. That function's other callers (``latest_run_segment_by_feed_dir``,
    ``latest_feed_run_allowed_relpaths``, ``corpus_catalog``) rank runs *lexicographically*, a
    weaker rule than the timestamp ordering the dedupe uses; teaching the shared parser about a
    new layout would silently change their behaviour too. This exists so the central
    membership rule can cover the flat layout without that blast radius.

    Why it is needed at all: ``feeds/<dir>/run_*/metadata/`` is what production writes, but
    ``run_*/metadata/`` is what a single-feed output dir looks like — and that is the layout
    ``skip-existing`` reads during a run. Without this the flat layout fell through the dedupe
    untouched and the caller's ``if guid in out: continue`` kept the FIRST match over
    ascending-sorted globs, i.e. the OLDEST run, every time.
    """
    parts = [p for p in rel_posix.replace("\\", "/").split("/") if p]
    if len(parts) >= 3 and parts[0].startswith("run_") and parts[1] == "metadata":
        return parts[0]
    return None


def corpus_relpath_is_excluded(rel_posix: str) -> bool:
    """True when a corpus-relative path lies under a RETIRED / non-content directory (#2161).

    Corpus cleanup moves superseded artifacts to ``.trash/<ts>/feeds/<feed>/run_*/metadata/``.
    Those files are real metadata for a real episode id, so every stage that walks the corpus for
    membership used to collect them as if they were live — and because the trash path parses as
    NEITHER corpus layout it was kept unconditionally and never compared against the live copy. The
    episode was then collected twice, its id-keyed rows collided, and the index prune deleted the
    LIVE copy's rows as "superseded".

    The rule is by leading dot rather than a ``.trash`` literal: every dot-directory here
    (``.trash``, ``.viewer`` job logs, caches) is bookkeeping, not corpus content, and a future
    retirement directory should be excluded on the day it is introduced rather than after it
    deletes data. Judged on the path RELATIVE to the corpus root, so a root that itself sits under
    a dotted directory is unaffected.
    """
    return any(
        seg.startswith(".") and seg not in (".", "..")
        for seg in rel_posix.replace("\\", "/").split("/")
        if seg
    )


#: Corpus-relative layouts this module knows how to reconcile. Anything else is unrecognised and
#: must not become canonical silently — see :func:`classify_metadata_relpath`.
LAYOUT_FEED_RUN = "feeds/<feed>/run_*/metadata"
LAYOUT_FLAT_RUN = "run_*/metadata"
LAYOUT_ROOT_FLAT = "metadata"


def classify_metadata_relpath(rel_posix: str) -> Optional[str]:
    """Name the corpus layout *rel_posix* belongs to, or ``None`` when unrecognised (#2162).

    Exists so "I could not classify this path" is a distinguishable state rather than being folded
    into the legitimate flat-corpus case. The root-flat shape (``metadata/<file>``) genuinely has no
    cross-run identity to reconcile and is common in fixtures, so it must NOT warn; a path that is
    none of the three known shapes is a surprise and must.
    """
    parts = [p for p in rel_posix.replace("\\", "/").split("/") if p]
    feed_dir, run_seg = feed_dir_and_run_segment_from_relpath(rel_posix)
    if feed_dir is not None and run_seg is not None:
        return LAYOUT_FEED_RUN
    if run_segment_from_flat_relpath(rel_posix) is not None:
        return LAYOUT_FLAT_RUN
    if len(parts) >= 2 and parts[0] == filesystem.METADATA_SUBDIR:
        return LAYOUT_ROOT_FLAT
    return None


def latest_run_segment_by_feed_dir(rel_posixes: Iterable[str]) -> dict[str, str]:
    """Map ``feeds/<feedDir>/`` → greatest ``run_*`` directory name (lexicographic order)."""
    latest: dict[str, str] = {}
    for rel in rel_posixes:
        feed_dir, run_seg = feed_dir_and_run_segment_from_relpath(rel.replace("\\", "/"))
        if feed_dir is None or run_seg is None:
            continue
        cur = latest.get(feed_dir)
        if cur is None or run_seg > cur:
            latest[feed_dir] = run_seg
    return latest


def latest_feed_run_allowed_relpaths(rel_posixes: Iterable[str]) -> frozenset[str]:
    """Relative paths to keep when a feed directory has multiple ``run_*`` children.

    Under ``feeds/<feedDir>/run_*/metadata/``, only paths whose ``run_*`` equals the
    lexicographic maximum for that ``feedDir`` are retained. All other relative paths
    (flat ``metadata/``, ``search/``, etc.) are kept.
    """
    rels = [str(r).replace("\\", "/") for r in rel_posixes]
    latest = latest_run_segment_by_feed_dir(rels)
    kept: set[str] = set()
    for rel in rels:
        feed_dir, run_seg = feed_dir_and_run_segment_from_relpath(rel)
        if feed_dir is None:
            kept.add(rel)
            continue
        want = latest.get(feed_dir)
        if want is None:
            kept.add(rel)
            continue
        if run_seg == want:
            kept.add(rel)
    return frozenset(kept)


def filter_metadata_paths_to_latest_feed_run(corpus_root: Path, paths: List[Path]) -> List[Path]:
    """Drop metadata paths under older ``feeds/.../run_*`` siblings (multi-feed layout)."""
    root_res = corpus_root.resolve()
    rels: List[str] = []
    by_rel: dict[str, Path] = {}
    for p in paths:
        try:
            rel = p.resolve().relative_to(root_res).as_posix()
        except ValueError:
            continue
        rels.append(rel)
        by_rel[rel] = p
    allowed = latest_feed_run_allowed_relpaths(rels)
    out = [by_rel[r] for r in rels if r in allowed]
    return sorted(set(out))


# Run dirs come in two shapes and BOTH carry the timestamp:
#     run_<YYYYMMDD-HHMMSS>_<hash>
#     run_<run-uuid>_<YYYYMMDD-HHMMSS>_<hash>      <- every one of prod's 397 dirs, 2026-08-31
# Anchoring on ``^run_(\d{8}-\d{6})`` matched only the first, so the uuid-prefixed form fell
# through to the mtime fallback — meaning the timestamp path this function exists to prefer
# had never once fired on the production corpus, and supersession ordering there rested
# entirely on mtime, the thing the docstring calls out as unsafe for a real corpus
# (file-copy / backup-restore / rsync churn).
#
# GREEDY prefix, deliberately. ``filesystem.py`` PREPENDS run_id, so the layout is
# ``run_<run_id>_<ts>_<hash>``: a spurious timestamp can only appear BEFORE the real one
# (inside run_id), while what follows the real timestamp is just ``_<hash>`` and optionally a
# counter — neither of which can match ``\d{8}-\d{6}``. The real run timestamp is therefore
# always the LAST match, so greedy is the safe direction.
#
# A non-greedy ``.*?_`` looks equivalent on prod's UUID run_ids (no ``_``) and is wrong the
# moment a run_id contains a timestamp — ``sanitize_filename`` preserves ``_``, ``-`` and
# digits, so such a run_id survives into the dir name:
#     run_nightly_20260101-000000_20260830-144405_a1b2c3d4
#         non-greedy -> 20260101-000000   (the run_id's, WRONG)
#         greedy     -> 20260830-144405   (the run's)
#
# ``run_append_<hash>`` still legitimately falls through to mtime (no timestamp), and a
# shape-matching-but-invalid date falls through via the strptime guard below.
_RUN_TS_RE = re.compile(r"^run_(?:.*_)?(\d{8}-\d{6})")


def run_recency_epoch(meta_path: Path, run_seg: str) -> float:
    """Recency of a run for supersession ordering — newest wins.

    Prefer the run-folder timestamp (``run_YYYYMMDD-HHMMSS_*``): it is immune to file-copy /
    backup-restore / rsync mtime churn, which matters for a production corpus. Fall back to the
    metadata file's mtime ONLY for timestamp-less run dirs (``run_append_<hash>``, #444) — those
    sort lexicographically after every timestamped run, so a lexicographic rule would let a stale
    append copy win forever; real recency (mtime) breaks the tie correctly. Both axes are compared
    as local-epoch seconds (``time.mktime`` ∘ ``strptime`` vs ``st_mtime``).
    """
    m = _RUN_TS_RE.match(run_seg or "")
    if m:
        try:
            return time.mktime(time.strptime(m.group(1), filesystem.TIMESTAMP_FORMAT))
        except (ValueError, OverflowError):
            pass
    try:
        return meta_path.stat().st_mtime
    except OSError:
        return 0.0


def _read_feed_episode_ids(meta_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Read ``(normalized feed_id, episode_id)`` from a metadata file; ``(None, None)`` on failure.

    Mirrors ``server.corpus_catalog._feed_and_episode_ids`` without importing server code into the
    lower-level search layer.
    """
    try:
        text = meta_path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    try:
        name = meta_path.name.lower()
        if name.endswith((".yaml", ".yml")):
            import yaml

            doc = yaml.safe_load(text)
        else:
            doc = json.loads(text)
    except Exception:
        return None, None
    if not isinstance(doc, dict):
        return None, None
    feed = doc.get("feed")
    episode = doc.get("episode")
    fid = feed.get("feed_id") if isinstance(feed, dict) else None
    eid = episode.get("episode_id") if isinstance(episode, dict) else None
    if not (isinstance(eid, str) and eid.strip()):
        # Real metadata carries both ``episode_id`` and the RSS-native ``guid``; some fixtures /
        # older docs carry only ``guid``. Fall back to it so dedup identity stays aligned with the
        # enrichment bundle resolution (``discover_episode_bundles``) and never fails to collapse a
        # reprocessed episode just because ``episode_id`` is absent.
        eid = episode.get("guid") if isinstance(episode, dict) else None
    eid_s = eid.strip() if isinstance(eid, str) and eid.strip() else None
    return normalize_feed_id(fid), eid_s


def dedupe_metadata_paths_newest_run_per_episode(
    corpus_root: Path, paths: List[Path]
) -> List[Path]:
    """Central corpus-membership rule: one winner per ``(feed_id, episode_id)`` across ALL runs.

    - Non-feed-layout files (flat ``metadata/``, or ``feeds/...`` without a ``run_*`` segment) are
      always kept — no cross-run identity to reconcile.
    - A feed dir with a SINGLE ``run_*`` cannot have a cross-run collision, so all its files are
      kept WITHOUT reading them (keeps the common case as cheap as the old path-only filter).
    - A feed dir with MULTIPLE ``run_*`` dirs is deduped by ``(feed_id, episode_id)`` read from the
      metadata; the newest run wins (:func:`run_recency_epoch`). Disjoint episodes across runs (the
      incremental-add case) all survive; a reprocessed episode's older "trophy" copy is dropped.

    Files whose ``episode_id`` cannot be read are kept (never silently dropped).

    Paths under a retired / bookkeeping directory are dropped outright
    (:func:`corpus_relpath_is_excluded`), and a path matching NO known layout is logged rather
    than silently promoted to canonical — see the unrecognised handling below (#2161 / #2162).
    """
    root_res = corpus_root.resolve()
    by_feed_dir: dict[str, list[Tuple[str, Path]]] = {}
    keep: list[Path] = []
    # Kept paths that DID classify into a known layout. Used only to adjudicate unrecognised
    # paths (below), so the ids are read lazily — a corpus with no surprises pays nothing.
    classified_kept: list[Path] = []
    unrecognised: list[Tuple[str, Path]] = []
    for p in paths:
        try:
            rel = p.resolve().relative_to(root_res).as_posix()
        except ValueError:
            keep.append(p)
            continue
        if corpus_relpath_is_excluded(rel):
            logger.debug("corpus membership: excluding retired/non-content path %s (#2161)", rel)
            continue
        layout = classify_metadata_relpath(rel)
        if layout == LAYOUT_FEED_RUN:
            feed_dir, run_seg = feed_dir_and_run_segment_from_relpath(rel)
            by_feed_dir.setdefault(str(feed_dir), []).append((str(run_seg), p))
            continue
        if layout == LAYOUT_FLAT_RUN:
            # Flat ``run_<tag>/metadata/`` — a single-feed output dir, which is the layout
            # skip-existing reads during a run. Leaving it unreconciled left the caller's
            # first-wins loop to pick the OLDEST run for a reprocessed episode.
            by_feed_dir.setdefault(FLAT_CORPUS_FEED_KEY, []).append(
                (str(run_segment_from_flat_relpath(rel)), p)
            )
            continue
        if layout == LAYOUT_ROOT_FLAT:
            # Root ``metadata/`` — the hybrid/fixture shape, deliberately unioned with the feed
            # tree. No cross-run identity to reconcile, and NOT a surprise, so it does not warn.
            keep.append(p)
            classified_kept.append(p)
            continue
        unrecognised.append((rel, p))

    for entries in by_feed_dir.values():
        if len({rs for rs, _ in entries}) <= 1:
            keep.extend(p for _, p in entries)
            classified_kept.extend(p for _, p in entries)
            continue
        winners: dict[Tuple[Optional[str], str], Tuple[float, str, str, Path]] = {}
        for run_seg, p in entries:
            _fid, eid = _read_feed_episode_ids(p)
            if eid is None:
                keep.append(p)
                classified_kept.append(p)
                continue
            key = (_fid, eid)
            cand = (run_recency_epoch(p, run_seg), run_seg, p.as_posix(), p)
            cur = winners.get(key)
            if cur is None or cand[:3] > cur[:3]:
                winners[key] = cand
        keep.extend(w[3] for w in winners.values())
        classified_kept.extend(w[3] for w in winners.values())

    if unrecognised:
        # Defence in depth (#2162). The old code kept an unclassifiable path unconditionally and
        # never compared it with the copy that legitimately owns its ``(feed_id, episode_id)``, so
        # ``.trash/`` copies were walked as independent episodes sharing a live id: duplicate row
        # ids every build, and the prune deleting the live rows. Reading ids here is gated on a
        # surprise actually existing.
        owned: dict[Tuple[Optional[str], str], Path] = {}
        for p in classified_kept:
            _fid, eid = _read_feed_episode_ids(p)
            if eid is not None:
                owned.setdefault((_fid, eid), p)
        for rel, p in unrecognised:
            _fid, eid = _read_feed_episode_ids(p)
            owner = owned.get((_fid, eid)) if eid is not None else None
            if owner is not None:
                logger.warning(
                    "corpus membership: REFUSING to treat unrecognised path %s as canonical — it "
                    "carries (feed_id=%r, episode_id=%r), already owned by the recognised copy %s. "
                    "Indexing both would collide their row ids and let the prune delete the live "
                    "rows (#2162).",
                    rel,
                    _fid,
                    eid,
                    owner.as_posix(),
                )
                continue
            logger.warning(
                "corpus membership: path %s matches no known corpus layout (%s | %s | %s) and is "
                "being kept as canonical by default. If this is a retired or bookkeeping "
                "directory it must be EXCLUDED instead of indexed (#2162).",
                rel,
                LAYOUT_FEED_RUN,
                LAYOUT_FLAT_RUN,
                LAYOUT_ROOT_FLAT,
            )
            keep.append(p)

    return sorted(set(keep))


def _collect_root_run_dirs(root_normed: str, collect: "Callable[[str], None]") -> None:
    """Discover ``<root>/run_<id>/metadata/`` — the layout a SINGLE-FEED run produces.

    The corpus has three shapes in practice and this rule is the single source of truth for all of
    them (indexing, digest, topic-clusters, enrichment, catalog and staleness all read it, so a
    layout missing here is invisible to every one of them at once):

        <root>/feeds/<feed>/run_<id>/metadata/   multi-feed, the prod shape
        <root>/metadata/                          flat, the fixture shape
        <root>/run_<id>/metadata/                 single-feed, what `--rss --output-dir X` writes

    The third was undiscoverable. The walk that finds nested run dirs is gated on ``feeds/``
    existing, so a corpus with run dirs and no ``feeds/`` fell to the flat branch, which looks only
    at ``<root>/metadata``. Result: ingest one feed into a fresh directory and every downstream
    stage reports an EMPTY corpus — enrichment says ``no_bundles``, the index builds nothing — with
    no error anywhere, because "no episodes" is a legitimate state.

    A targeted glob rather than a full walk: the flat branch is the hot path for large corpora and
    ``os.walk`` over it would be a real cost for a layout that, by definition, has no nesting.
    """
    for run_dir in sorted(_glob.glob(os.path.join(root_normed, "run_*"))):
        if os.path.isdir(run_dir):
            collect(os.path.join(run_dir, filesystem.METADATA_SUBDIR))


def discover_metadata_files(output_root: Path) -> List[Path]:
    """List episode metadata files — the CENTRAL corpus-membership rule (single source of truth).

    Hybrid layout: if ``feeds/`` exists **and** top-level ``metadata/`` exists, both are
    included (GitHub #505 follow-up).

    Discovers every ``feeds/<feedDir>/run_*/metadata/`` tree (union across ALL runs) and returns one
    winner per ``(feed_id, episode_id)``: when the same episode is reprocessed into a newer run the
    newest run wins (:func:`dedupe_metadata_paths_newest_run_per_episode`). This is what makes an
    incremental add (a new run dir with only the new episode) weave into the corpus WITHOUT dropping
    the feed's prior episodes, while a full reindex still supersedes reprocessed "trophy" runs
    instead of double-indexing them. Indexing, digest, topic-clusters, enrichment, catalog, and
    staleness all share this one rule so they can never diverge (the 94-vs-106 split-brain).
    """
    corpus_root = safe_resolve_directory(output_root)
    if corpus_root is None:
        return []

    # CodeQL py/path-injection sanitiser: normpath then startswith on every
    # tainted value before it reaches a filesystem sink.  ``os.sep`` is used
    # as the non-tainted anchor (ensures the path is absolute).
    root_normed = os.path.normpath(str(corpus_root))
    if not root_normed.startswith(os.sep):
        return []

    safe_prefix = root_normed + os.sep
    patterns = ("*.metadata.json", "*.metadata.yaml", "*.metadata.yml")
    found: List[Path] = []

    def _collect(meta_dir_str: str) -> None:
        md = os.path.normpath(meta_dir_str)
        if not md.startswith(safe_prefix) and md != root_normed:
            return
        # Retired / bookkeeping trees are not corpus content (#2161). Belt-and-braces with the
        # walk pruning below: this also covers the glob-driven branches, which do not walk.
        if corpus_relpath_is_excluded(os.path.relpath(md, root_normed).replace(os.sep, "/")):
            return
        if not os.path.isdir(md):
            return
        for pat in patterns:
            for hit_str in _glob.glob(os.path.join(md, pat)):
                h = os.path.normpath(hit_str)
                if not h.startswith(safe_prefix) and h != root_normed:
                    continue
                if os.path.isfile(h):
                    found.append(Path(h))

    feeds_str = os.path.normpath(os.path.join(root_normed, "feeds"))
    if feeds_str.startswith(safe_prefix) and os.path.isdir(feeds_str):
        for dirpath, dirnames, _filenames in os.walk(root_normed):
            # Never descend into retired / bookkeeping trees (#2161). Pruning in place stops the
            # walk entering them at all: both the fix, and a real saving on a corpus whose
            # ``.trash/`` has accumulated every superseded run.
            dirnames[:] = [d for d in dirnames if not corpus_relpath_is_excluded(d)]
            dp = os.path.normpath(dirpath)
            if not dp.startswith(safe_prefix) and dp != root_normed:
                continue
            if os.path.basename(dp) == filesystem.METADATA_SUBDIR:
                _collect(dp)
        _collect(os.path.join(root_normed, filesystem.METADATA_SUBDIR))
    else:
        _collect(os.path.join(root_normed, filesystem.METADATA_SUBDIR))
        _collect_root_run_dirs(root_normed, _collect)

    corpus_path = Path(root_normed)
    deduped = dedupe_metadata_paths_newest_run_per_episode(corpus_path, list(set(found)))
    return sorted(deduped)


def discover_all_metadata_files(output_root: Path) -> List[Path]:
    """Like :func:`discover_metadata_files` but WITHOUT the last-run-only filter.

    Returns every ``*.metadata.json`` under ``feeds/<feedDir>/run_*/metadata/``
    (and the flat top-level ``metadata/``) for every run. Used by the corpus
    library + stats routes which need cumulative-unique counts across all
    runs (v2.6.1 hotfix: #818 / #820 / #821).

    Indexing, digest, and topic-clusters still call :func:`discover_metadata_files`
    so they continue seeing one-run-per-feed (load-bearing for index rebuild
    performance).
    """
    corpus_root = safe_resolve_directory(output_root)
    if corpus_root is None:
        return []

    root_normed = os.path.normpath(str(corpus_root))
    if not root_normed.startswith(os.sep):
        return []

    safe_prefix = root_normed + os.sep
    patterns = ("*.metadata.json", "*.metadata.yaml", "*.metadata.yml")
    found: List[Path] = []

    def _collect(meta_dir_str: str) -> None:
        md = os.path.normpath(meta_dir_str)
        if not md.startswith(safe_prefix) and md != root_normed:
            return
        # Retired / bookkeeping trees are not corpus content (#2161). Belt-and-braces with the
        # walk pruning below: this also covers the glob-driven branches, which do not walk.
        if corpus_relpath_is_excluded(os.path.relpath(md, root_normed).replace(os.sep, "/")):
            return
        if not os.path.isdir(md):
            return
        for pat in patterns:
            for hit_str in _glob.glob(os.path.join(md, pat)):
                h = os.path.normpath(hit_str)
                if not h.startswith(safe_prefix) and h != root_normed:
                    continue
                if os.path.isfile(h):
                    found.append(Path(h))

    feeds_str = os.path.normpath(os.path.join(root_normed, "feeds"))
    if feeds_str.startswith(safe_prefix) and os.path.isdir(feeds_str):
        for dirpath, dirnames, _filenames in os.walk(root_normed):
            # Never descend into retired / bookkeeping trees (#2161). Pruning in place stops the
            # walk entering them at all: both the fix, and a real saving on a corpus whose
            # ``.trash/`` has accumulated every superseded run.
            dirnames[:] = [d for d in dirnames if not corpus_relpath_is_excluded(d)]
            dp = os.path.normpath(dirpath)
            if not dp.startswith(safe_prefix) and dp != root_normed:
                continue
            if os.path.basename(dp) == filesystem.METADATA_SUBDIR:
                _collect(dp)
        _collect(os.path.join(root_normed, filesystem.METADATA_SUBDIR))
    else:
        _collect(os.path.join(root_normed, filesystem.METADATA_SUBDIR))
        # Same single-feed layout as discover_metadata_files — the two must agree, or the
        # cumulative-count routes and the indexer would disagree about corpus membership.
        _collect_root_run_dirs(root_normed, _collect)

    return sorted(set(found))
