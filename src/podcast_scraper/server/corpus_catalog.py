"""Filesystem-backed episode catalog for Corpus Library."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from podcast_scraper.builders.bridge_artifact_paths import bridge_json_path_adjacent_to_metadata
from podcast_scraper.search.corpus_scope import discover_metadata_files, normalize_feed_id
from podcast_scraper.utils.corpus_artwork import CORPUS_ART_REL_PREFIX
from podcast_scraper.utils.path_validation import (
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)


def _feed_and_episode_ids(doc: Optional[dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    if not doc:
        return None, None
    feed = doc.get("feed")
    episode = doc.get("episode")
    fid: Any = feed.get("feed_id") if isinstance(feed, dict) else None
    eid: Any = episode.get("episode_id") if isinstance(episode, dict) else None
    return normalize_feed_id(fid), eid.strip() if isinstance(eid, str) and eid.strip() else None


def _load_metadata_doc(meta_path: str | Path) -> Optional[dict[str, Any]]:
    ps = str(meta_path)
    try:
        # codeql[py/path-injection] -- callers pass normpath-sanitized strings.
        with open(ps, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    try:
        lower = os.path.basename(ps).lower()
        if lower.endswith((".yaml", ".yml")):
            import yaml

            raw = yaml.safe_load(text)
            return raw if isinstance(raw, dict) else None
        blob = json.loads(text)
        return blob if isinstance(blob, dict) else None
    except Exception:
        return None


def _gi_kg_relpaths_from_metadata(metadata_relpath: str) -> tuple[str, str]:
    mp = metadata_relpath
    if mp.endswith(".metadata.json"):
        base = mp[: -len(".metadata.json")]
        return f"{base}.gi.json", f"{base}.kg.json"
    if mp.endswith(".metadata.yaml"):
        base = mp[: -len(".metadata.yaml")]
        return f"{base}.gi.json", f"{base}.kg.json"
    if mp.endswith(".metadata.yml"):
        base = mp[: -len(".metadata.yml")]
        return f"{base}.gi.json", f"{base}.kg.json"
    stem = os.path.splitext(mp)[0]
    return f"{stem}.gi.json", f"{stem}.kg.json"


def _parse_publish_date_str(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    if isinstance(raw, str) and raw.strip():
        s = raw.strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except ValueError:
            return None
    return None


def _episode_title(doc: dict[str, Any]) -> str:
    ep = doc.get("episode")
    if isinstance(ep, dict):
        t = ep.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    t2 = doc.get("title")
    if isinstance(t2, str) and t2.strip():
        return t2.strip()
    return "(untitled)"


def _feed_display_title(doc: dict[str, Any]) -> Optional[str]:
    feed = doc.get("feed")
    if isinstance(feed, dict):
        t = feed.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return None


def _feed_rss_url(doc: dict[str, Any]) -> Optional[str]:
    feed = doc.get("feed")
    if isinstance(feed, dict):
        u = feed.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()
    return None


def _feed_description(doc: dict[str, Any]) -> Optional[str]:
    feed = doc.get("feed")
    if isinstance(feed, dict):
        d = feed.get("description")
        if isinstance(d, str) and d.strip():
            return d.strip()
    return None


def _optional_image_url(raw: Any) -> Optional[str]:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _optional_non_negative_int(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, float) and raw.is_integer():
        i = int(raw)
        return i if i >= 0 else None
    return None


def _optional_relpath_field(raw: Any) -> Optional[str]:
    if isinstance(raw, str) and raw.strip():
        s = raw.strip().replace("\\", "/")
        return s.lstrip("/")
    return None


def _verified_artwork_relpath(corpus_root: Path, rel: Optional[str]) -> Optional[str]:
    """Return ``rel`` only when it points at a file under ``CORPUS_ART_REL_PREFIX``."""
    if not rel:
        return None
    norm = rel.strip().replace("\\", "/").lstrip("/")
    prefix = f"{CORPUS_ART_REL_PREFIX}/"
    if norm == CORPUS_ART_REL_PREFIX or not norm.startswith(prefix):
        return None
    segments = [p for p in norm.split("/") if p]
    if ".." in segments:
        return None
    target_str = safe_relpath_under_corpus_root(corpus_root, norm)
    # codeql[py/path-injection] -- target_str from normpath+startswith in safe_relpath.
    if not target_str or not os.path.isfile(target_str):
        return None
    return norm


def _visual_fields_from_doc(
    corpus_root: Path,
    doc: dict[str, Any],
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[str],
]:
    """RSS URLs, scalars, and verified on-disk artwork paths from metadata."""
    feed = doc.get("feed")
    ep = doc.get("episode")
    feed_img = _optional_image_url(feed.get("image_url")) if isinstance(feed, dict) else None
    feed_loc_raw = (
        _optional_relpath_field(feed.get("image_local_relpath")) if isinstance(feed, dict) else None
    )
    ep_img = None
    ep_loc_raw = None
    duration: Optional[int] = None
    ep_num: Optional[int] = None
    if isinstance(ep, dict):
        ep_img = _optional_image_url(ep.get("image_url"))
        ep_loc_raw = _optional_relpath_field(ep.get("image_local_relpath"))
        duration = _optional_non_negative_int(ep.get("duration_seconds"))
        ep_num = _optional_non_negative_int(ep.get("episode_number"))
    feed_local = _verified_artwork_relpath(corpus_root, feed_loc_raw)
    ep_local = _verified_artwork_relpath(corpus_root, ep_loc_raw)
    return feed_img, ep_img, duration, ep_num, feed_local, ep_local


def _summary_fields(doc: dict[str, Any]) -> tuple[Optional[str], list[str]]:
    summary = doc.get("summary")
    if not isinstance(summary, dict):
        return None, []
    st = summary.get("title")
    title = st.strip() if isinstance(st, str) and st.strip() else None
    bullets_raw = summary.get("bullets")
    bullets: list[str] = []
    if isinstance(bullets_raw, list):
        for b in bullets_raw:
            if isinstance(b, str) and b.strip():
                bullets.append(b.strip())
    return title, bullets


def _summary_body_text(doc: dict[str, Any]) -> Optional[str]:
    """Long-form summary prose: ``raw_text`` or ``short_summary`` when present."""
    summary = doc.get("summary")
    if not isinstance(summary, dict):
        return None
    raw = summary.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    short = summary.get("short_summary")
    if isinstance(short, str) and short.strip():
        return short.strip()
    return None


@dataclass(frozen=True)
class CatalogEpisodeRow:
    """One episode row derived from a metadata file."""

    metadata_relative_path: str
    feed_id: str
    feed_title: Optional[str]
    episode_id: Optional[str]
    episode_title: str
    publish_date: Optional[str]
    summary_title: Optional[str]
    summary_bullets: tuple[str, ...]
    summary_text: Optional[str]
    gi_relative_path: str
    kg_relative_path: str
    bridge_relative_path: str
    has_gi: bool
    has_kg: bool
    has_bridge: bool
    feed_image_url: Optional[str] = None
    episode_image_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    episode_number: Optional[int] = None
    feed_image_local_relpath: Optional[str] = None
    episode_image_local_relpath: Optional[str] = None
    feed_rss_url: Optional[str] = None
    feed_description: Optional[str] = None

    def sort_key(self) -> tuple[int, int, str]:
        """Newest-first: dated episodes before undated; then by ordinal desc; then path."""
        if self.publish_date:
            try:
                d = date.fromisoformat(self.publish_date[:10])
                return (0, -d.toordinal(), self.metadata_relative_path)
            except ValueError:
                pass
        return (1, 0, self.metadata_relative_path)


def build_catalog_rows(corpus_root: Path) -> list[CatalogEpisodeRow]:
    """Scan corpus for ``*.metadata.*`` and build catalog rows."""
    root = safe_resolve_directory(corpus_root)
    if root is None:
        return []
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    rows: list[CatalogEpisodeRow] = []
    for meta_path in discover_metadata_files(root):
        try:
            rel = meta_path.relative_to(root).as_posix()
        except ValueError:
            continue
        doc = _load_metadata_doc(str(meta_path))
        if doc is None:
            continue
        fid_norm, eid = _feed_and_episode_ids(doc)
        feed_id = fid_norm or ""
        feed_title = _feed_display_title(doc)
        episode_title = _episode_title(doc)
        ep = doc.get("episode")
        pub_raw = None
        if isinstance(ep, dict):
            pub_raw = ep.get("published_date")
        publish_date = _parse_publish_date_str(pub_raw)
        stitle, sbullets = _summary_fields(doc)
        sbody = _summary_body_text(doc)
        f_img, e_img, dur_s, ep_n, f_loc, e_loc = _visual_fields_from_doc(root, doc)
        feed_url = _feed_rss_url(doc)
        feed_desc = _feed_description(doc)
        gi_rel, kg_rel = _gi_kg_relpaths_from_metadata(rel)
        bridge_rel = bridge_json_path_adjacent_to_metadata(rel)
        gi_safe = safe_relpath_under_corpus_root(root, gi_rel)
        kg_safe = safe_relpath_under_corpus_root(root, kg_rel)
        bridge_safe = safe_relpath_under_corpus_root(root, bridge_rel)
        if gi_safe:
            gi_safe = os.path.normpath(gi_safe)
        if kg_safe:
            kg_safe = os.path.normpath(kg_safe)
        if bridge_safe:
            bridge_safe = os.path.normpath(bridge_safe)
        has_gi = bool(gi_safe and gi_safe.startswith(safe_prefix) and os.path.isfile(gi_safe))
        has_kg = bool(kg_safe and kg_safe.startswith(safe_prefix) and os.path.isfile(kg_safe))
        has_bridge = bool(
            bridge_safe and bridge_safe.startswith(safe_prefix) and os.path.isfile(bridge_safe)
        )
        rows.append(
            CatalogEpisodeRow(
                metadata_relative_path=rel,
                feed_id=feed_id,
                feed_title=feed_title,
                episode_id=eid,
                episode_title=episode_title,
                publish_date=publish_date,
                summary_title=stitle,
                summary_bullets=tuple(sbullets),
                summary_text=sbody,
                gi_relative_path=gi_rel,
                kg_relative_path=kg_rel,
                bridge_relative_path=bridge_rel,
                has_gi=has_gi,
                has_kg=has_kg,
                has_bridge=has_bridge,
                feed_image_url=f_img,
                episode_image_url=e_img,
                duration_seconds=dur_s,
                episode_number=ep_n,
                feed_image_local_relpath=f_loc,
                episode_image_local_relpath=e_loc,
                feed_rss_url=feed_url,
                feed_description=feed_desc,
            )
        )
    rows.sort(key=lambda r: r.sort_key())
    return rows


def catalog_row_for_metadata_path(
    corpus_root: Path, metadata_relative_path: str
) -> Optional[CatalogEpisodeRow]:
    """Build a single catalog row from a metadata relative path (no full corpus scan)."""
    root = corpus_root.resolve()
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    safe_meta = safe_relpath_under_corpus_root(root, metadata_relative_path)
    if not safe_meta:
        return None
    safe_meta = os.path.normpath(safe_meta)
    if not safe_meta.startswith(safe_prefix) or not os.path.isfile(safe_meta):
        return None
    rel = os.path.relpath(safe_meta, root_s).replace("\\", "/")
    if rel.startswith(".."):
        return None
    doc = _load_metadata_doc(safe_meta)
    if doc is None:
        return None
    fid_norm, eid = _feed_and_episode_ids(doc)
    feed_id = fid_norm or ""
    feed_title = _feed_display_title(doc)
    episode_title = _episode_title(doc)
    ep = doc.get("episode")
    pub_raw = None
    if isinstance(ep, dict):
        pub_raw = ep.get("published_date")
    publish_date = _parse_publish_date_str(pub_raw)
    stitle, sbullets = _summary_fields(doc)
    sbody = _summary_body_text(doc)
    f_img, e_img, dur_s, ep_n, f_loc, e_loc = _visual_fields_from_doc(root, doc)
    feed_url = _feed_rss_url(doc)
    feed_desc = _feed_description(doc)
    gi_rel, kg_rel = _gi_kg_relpaths_from_metadata(rel)
    bridge_rel = bridge_json_path_adjacent_to_metadata(rel)
    gi_safe = safe_relpath_under_corpus_root(root, gi_rel)
    kg_safe = safe_relpath_under_corpus_root(root, kg_rel)
    bridge_safe = safe_relpath_under_corpus_root(root, bridge_rel)
    if gi_safe:
        gi_safe = os.path.normpath(gi_safe)
    if kg_safe:
        kg_safe = os.path.normpath(kg_safe)
    if bridge_safe:
        bridge_safe = os.path.normpath(bridge_safe)
    has_gi = bool(gi_safe and gi_safe.startswith(safe_prefix) and os.path.isfile(gi_safe))
    has_kg = bool(kg_safe and kg_safe.startswith(safe_prefix) and os.path.isfile(kg_safe))
    has_bridge = bool(
        bridge_safe and bridge_safe.startswith(safe_prefix) and os.path.isfile(bridge_safe)
    )
    return CatalogEpisodeRow(
        metadata_relative_path=rel,
        feed_id=feed_id,
        feed_title=feed_title,
        episode_id=eid,
        episode_title=episode_title,
        publish_date=publish_date,
        summary_title=stitle,
        summary_bullets=tuple(sbullets),
        summary_text=sbody,
        gi_relative_path=gi_rel,
        kg_relative_path=kg_rel,
        bridge_relative_path=bridge_rel,
        has_gi=has_gi,
        has_kg=has_kg,
        has_bridge=has_bridge,
        feed_image_url=f_img,
        episode_image_url=e_img,
        duration_seconds=dur_s,
        episode_number=ep_n,
        feed_image_local_relpath=f_loc,
        episode_image_local_relpath=e_loc,
        feed_rss_url=feed_url,
        feed_description=feed_desc,
    )


def index_rows_by_feed_episode(
    rows: Iterable[CatalogEpisodeRow],
) -> dict[tuple[str, str], CatalogEpisodeRow]:
    """Map ``(feed_id, episode_id)`` to catalog row (skips rows without ``episode_id``)."""
    out: dict[tuple[str, str], CatalogEpisodeRow] = {}
    for row in rows:
        if row.episode_id:
            out[(row.feed_id, row.episode_id)] = row
    return out


def index_rows_by_episode_id(
    rows: Iterable[CatalogEpisodeRow],
) -> dict[str, CatalogEpisodeRow]:
    """Map ``episode_id`` to a catalog row (first row wins if duplicates)."""
    out: dict[str, CatalogEpisodeRow] = {}
    for row in rows:
        eid = row.episode_id
        if eid and eid not in out:
            out[eid] = row
    return out


def aggregate_feeds(rows: Iterable[CatalogEpisodeRow]) -> list[dict[str, Any]]:
    """Distinct feeds with optional display title, image URL, and episode counts."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = by_id.setdefault(
            row.feed_id,
            {
                "feed_id": row.feed_id,
                "display_title": None,
                "episode_count": 0,
                "image_url": None,
                "image_local_relpath": None,
                "rss_url": None,
                "description": None,
            },
        )
        bucket["episode_count"] = int(bucket["episode_count"]) + 1
        if bucket["display_title"] is None and row.feed_title:
            bucket["display_title"] = row.feed_title
        if bucket["image_url"] is None and row.feed_image_url:
            bucket["image_url"] = row.feed_image_url
        if bucket["image_local_relpath"] is None and row.feed_image_local_relpath:
            bucket["image_local_relpath"] = row.feed_image_local_relpath
        if bucket["rss_url"] is None and row.feed_rss_url:
            bucket["rss_url"] = row.feed_rss_url.strip() or None
        if bucket["description"] is None and row.feed_description:
            bucket["description"] = row.feed_description.strip() or None
    out = list(by_id.values())
    out.sort(key=lambda x: (x["feed_id"] == "", x["feed_id"]))
    return out


def feed_display_title_by_feed_id(rows: Iterable[CatalogEpisodeRow]) -> dict[str, str]:
    """Map ``feed_id`` -> first non-empty ``feed_title`` (catalog iteration order).

    Matches :func:`aggregate_feeds` so digest/library rows can show a feed name when
    this episode's metadata omitted ``feed.title`` but another episode in the feed
    has it.
    """
    out: dict[str, str] = {}
    for row in rows:
        t = (row.feed_title or "").strip()
        if not t:
            continue
        fid = row.feed_id
        if fid not in out:
            out[fid] = t
    return out


def feed_rss_url_by_feed_id(rows: Iterable[CatalogEpisodeRow]) -> dict[str, str]:
    """Map ``feed_id`` -> first non-empty ``feed_rss_url`` (catalog iteration order)."""
    out: dict[str, str] = {}
    for row in rows:
        u = (row.feed_rss_url or "").strip()
        if not u:
            continue
        fid = row.feed_id
        if fid not in out:
            out[fid] = u
    return out


def feed_description_by_feed_id(rows: Iterable[CatalogEpisodeRow]) -> dict[str, str]:
    """Map ``feed_id`` -> first non-empty ``feed_description`` (catalog iteration order)."""
    out: dict[str, str] = {}
    for row in rows:
        d = (row.feed_description or "").strip()
        if not d:
            continue
        fid = row.feed_id
        if fid not in out:
            out[fid] = d
    return out


def resolve_feed_display_title(
    row: CatalogEpisodeRow,
    by_feed_id: Mapping[str, str],
) -> Optional[str]:
    """Prefer this row's ``feed_title``; else any title indexed for ``row.feed_id``."""
    t = (row.feed_title or "").strip()
    if t:
        return t
    alt = (by_feed_id.get(row.feed_id) or "").strip()
    return alt if alt else None


def resolve_feed_rss_url(
    row: CatalogEpisodeRow,
    by_feed_id: Mapping[str, str],
) -> Optional[str]:
    """Prefer this row's ``feed_rss_url``; else first URL indexed for ``row.feed_id``."""
    u = (row.feed_rss_url or "").strip()
    if u:
        return u
    alt = (by_feed_id.get(row.feed_id) or "").strip()
    return alt if alt else None


def resolve_feed_description(
    row: CatalogEpisodeRow,
    by_feed_id: Mapping[str, str],
) -> Optional[str]:
    """Prefer this row's ``feed_description``; else first description indexed for ``feed_id``."""
    d = (row.feed_description or "").strip()
    if d:
        return d
    alt = (by_feed_id.get(row.feed_id) or "").strip()
    return alt if alt else None


def encode_catalog_cursor(offset: int) -> str:
    """Opaque cursor for offset pagination (Phase 1)."""
    raw = json.dumps({"o": max(0, int(offset))}).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_catalog_cursor(cursor: Optional[str]) -> int:
    """Decode offset cursor from the episodes API; return ``0`` if missing or invalid."""
    if not cursor or not str(cursor).strip():
        return 0
    pad = "=" * ((4 - len(cursor) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(str(cursor).strip() + pad)
        data = json.loads(raw.decode("utf-8"))
        o = data.get("o", 0)
        return max(0, int(o))
    except (ValueError, json.JSONDecodeError, TypeError):
        return 0


def episode_list_topics(
    bullets: tuple[str, ...],
    *,
    max_items: int = 4,
    max_len: int = 160,
) -> list[str]:
    """Short labels for Library list pills (from summary bullets)."""
    out: list[str] = []
    for b in bullets[:max_items]:
        s = str(b).strip()
        if not s:
            continue
        if len(s) > max_len:
            s = s[: max_len - 1] + "…"
        out.append(s)
    return out


def episode_list_summary_preview(
    row: CatalogEpisodeRow,
    *,
    max_len: int = 240,
) -> Optional[str]:
    """One-line recap for episode list rows (title + bullets or prose), capped for UI density."""
    st = (row.summary_title or "").strip()
    bullets = [str(b).strip() for b in row.summary_bullets if str(b).strip()]
    body = (row.summary_text or "").strip()
    parts: list[str] = []
    if st and bullets:
        tail = bullets[0]
        if len(bullets) > 1:
            tail = f"{bullets[0]} · {bullets[1]}"
        parts.append(f"{st} — {tail}")
    elif st:
        parts.append(st)
    elif bullets:
        if len(bullets) == 1:
            parts.append(bullets[0])
        else:
            parts.append(f"{bullets[0]} · {bullets[1]}")
    elif body:
        parts.append(body[:200] + ("…" if len(body) > 200 else ""))
    if not parts:
        return None
    out = parts[0]
    if len(out) > max_len:
        return out[: max_len - 1] + "…"
    return out


def filter_rows(
    rows: list[CatalogEpisodeRow],
    *,
    feed_id: Optional[str] = None,
    title_q: Optional[str] = None,
    topic_q: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    has_gi: Optional[bool] = None,
) -> list[CatalogEpisodeRow]:
    """Filter catalog rows by feed id, title/topic substrings, publish date range, GI presence."""
    out = rows
    if feed_id is not None:
        want = str(feed_id).strip()
        out = [r for r in out if r.feed_id == want]
    if title_q and title_q.strip():
        q = title_q.strip().lower()
        out = [r for r in out if q in r.episode_title.lower()]
    if topic_q and topic_q.strip():
        tq = topic_q.strip().lower()

        def row_matches_topic(r: CatalogEpisodeRow) -> bool:
            st = r.summary_title
            if isinstance(st, str) and tq in st.lower():
                return True
            return any(tq in str(b).lower() for b in r.summary_bullets)

        out = [r for r in out if row_matches_topic(r)]
    if since and since.strip() and re.match(r"^\d{4}-\d{2}-\d{2}$", since.strip()):
        cutoff = date.fromisoformat(since.strip()[:10])
        filtered: list[CatalogEpisodeRow] = []
        for r in out:
            if not r.publish_date:
                continue
            try:
                d = date.fromisoformat(r.publish_date[:10])
            except ValueError:
                continue
            if d >= cutoff:
                filtered.append(r)
        out = filtered
    if until and until.strip() and re.match(r"^\d{4}-\d{2}-\d{2}$", until.strip()):
        upper = date.fromisoformat(until.strip()[:10])
        capped: list[CatalogEpisodeRow] = []
        for r in out:
            if not r.publish_date:
                continue
            try:
                d = date.fromisoformat(r.publish_date[:10])
            except ValueError:
                continue
            if d <= upper:
                capped.append(r)
        out = capped
    if has_gi is True:
        out = [r for r in out if r.has_gi]
    elif has_gi is False:
        out = [r for r in out if not r.has_gi]
    return out


def metadata_candidate_relpaths_from_artifact(relative_path_posix: str) -> list[str]:
    """Sibling metadata paths for corpus-relative ``*.gi`` / ``*.kg`` / ``*.bridge`` JSON paths."""
    r = relative_path_posix.replace("\\", "/").strip()
    if r.endswith(".gi.json"):
        stem = r[: -len(".gi.json")]
    elif r.endswith(".kg.json"):
        stem = r[: -len(".kg.json")]
    elif r.endswith(".bridge.json"):
        stem = r[: -len(".bridge.json")]
    else:
        return []
    return [
        f"{stem}.metadata.json",
        f"{stem}.metadata.yaml",
        f"{stem}.metadata.yml",
    ]


def publish_calendar_date_for_artifact_listing(
    corpus_root: Path,
    relative_path_posix: str,
    mtime_ts: float,
) -> str:
    """Calendar ``YYYY-MM-DD`` for artifact rows (metadata date, else UTC day from mtime)."""
    root = corpus_root.resolve()
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    for mp in metadata_candidate_relpaths_from_artifact(relative_path_posix):
        safe = safe_relpath_under_corpus_root(root, mp)
        if not safe:
            continue
        meta_abs = os.path.normpath(safe)
        if not meta_abs.startswith(safe_prefix) or not os.path.isfile(meta_abs):
            continue
        doc = _load_metadata_doc(meta_abs)
        if not doc:
            continue
        ep = doc.get("episode")
        if isinstance(ep, dict):
            parsed = _parse_publish_date_str(ep.get("published_date"))
            if parsed:
                return parsed
    dt = datetime.fromtimestamp(mtime_ts, tz=timezone.utc)
    return dt.date().isoformat()


def slice_page(
    rows: list[CatalogEpisodeRow], offset: int, limit: int
) -> tuple[list[CatalogEpisodeRow], Optional[str]]:
    """Return page slice and next_cursor when more rows exist."""
    off = max(0, offset)
    lim = max(1, min(200, limit))
    chunk = rows[off : off + lim]
    next_off = off + lim
    next_cur = encode_catalog_cursor(next_off) if next_off < len(rows) else None
    return chunk, next_cur
