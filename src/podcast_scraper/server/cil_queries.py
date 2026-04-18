"""RFC-072 cross-layer corpus queries (bridge + GI + KG on disk; no database).

Public query functions accept two path strings:

* ``root_path`` — user-influenced corpus root (may be tainted).
* ``anchor_path`` — the server's configured ``output_dir`` (not tainted).

Every function normalises ``root_path`` with ``os.path.normpath`` and then
guards it with ``root_s.startswith(anchor_s)`` before any filesystem access
(``os.walk``, ``os.path.isdir``, ``open``).  This is the exact sanitiser
shape that CodeQL's ``py/path-injection`` query recognises.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

from podcast_scraper.gi.edge_normalization import normalize_gil_edge_type
from podcast_scraper.server.corpus_catalog import (
    _optional_image_url,
    _optional_relpath_field,
    _verified_artwork_relpath,
)

logger = logging.getLogger(__name__)


def canonical_cil_entity_id(viewer_graph_id: str) -> str:
    """Map merged-graph node ids to bridge identity ids (``topic:…``, ``person:…``).

    The viewer may emit ``g:topic:…``, ``k:topic:…``, or stacked prefixes such as
    ``g:k:topic:…`` (``mergeGiKg.ts``). Bridge files list bare ``topic:…``. Use the
    same stripping loop as GI ABOUT endpoints (``_strip_layer_prefixes_for_cil``)
    so cluster timelines and topic queries match disk.
    """
    return _strip_layer_prefixes_for_cil(viewer_graph_id.strip())


def _strip_layer_prefixes_for_cil(raw: str) -> str:
    """Normalize ``g:`` / ``k:`` / ``kg:`` layer prefixes on edge endpoints.

    Matches ``web/gi-kg-viewer`` ``stripLayerPrefixesForCil`` so ABOUT ``to`` ids agree
    with bridge ``topic:…`` / ``person:…`` rows when GI files still use prefixed refs.
    """
    s = raw.strip()
    prev = None
    while s != prev:
        prev = s
        if s.startswith("g:"):
            s = s[2:]
        elif s.startswith("k:") and not s.startswith("kg:"):
            s = s[2:]
        elif s.startswith("kg:"):
            s = s[3:]
    return s


def _read_json(path_str: str) -> dict[str, Any] | None:
    try:
        with open(path_str, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        logger.debug("cil_queries: skip read %s: %s", path_str, exc)
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.debug("cil_queries: invalid JSON %s: %s", path_str, exc)
        return None
    return data if isinstance(data, dict) else None


def iter_cil_episode_bundles(
    root_path: str,
    anchor_path: str,
) -> Iterator[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Yield ``(safe_bridge_path, bridge, gi, kg)`` for each sibling triple.

    Only *anchor_path* (the **untainted** server ``output_dir``) is used for
    ``os.walk`` and ``os.path.isdir``.  The user-supplied *root_path* is
    normalised and used only as a string prefix filter on the results produced
    by the anchor walk — it never touches a filesystem call directly.
    """
    anchor_s = os.path.normpath(anchor_path)
    root_s = os.path.normpath(root_path)
    if root_s != anchor_s and not root_s.startswith(anchor_s + os.sep):
        return
    root_prefix = root_s + os.sep
    anchor_prefix = anchor_s + os.sep

    if not os.path.isdir(anchor_s):
        return

    bridge_paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(anchor_s):
        dnorm = os.path.normpath(dirpath)
        if dnorm != anchor_s and not dnorm.startswith(anchor_prefix):
            continue
        if dnorm != root_s and not dnorm.startswith(root_prefix):
            continue
        for fn in filenames:
            if not fn.endswith(".bridge.json"):
                continue
            joined = os.path.normpath(os.path.join(dnorm, fn))
            if not joined.startswith(root_prefix):
                continue
            if os.path.isfile(joined):
                bridge_paths.append(joined)
    bridge_paths.sort()

    for safe_bridge in bridge_paths:
        if not safe_bridge.startswith(root_prefix):
            continue
        parent = os.path.dirname(safe_bridge)
        name = os.path.basename(safe_bridge)
        if not name.endswith(".bridge.json"):
            continue
        stem = name[: -len(".bridge.json")]
        gi_j = os.path.normpath(os.path.join(parent, f"{stem}.gi.json"))
        kg_j = os.path.normpath(os.path.join(parent, f"{stem}.kg.json"))
        if not gi_j.startswith(root_prefix):
            continue
        if not kg_j.startswith(root_prefix):
            continue
        if not os.path.isfile(gi_j) or not os.path.isfile(kg_j):
            continue
        bridge = _read_json(safe_bridge)
        gi = _read_json(gi_j)
        kg = _read_json(kg_j)
        if bridge is None or gi is None or kg is None:
            continue
        yield safe_bridge, bridge, gi, kg


def iter_cil_bridge_bundles(
    root_path: str,
    anchor_path: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(safe_bridge_path, bridge)`` reading only each ``*.bridge.json``.

    Same traversal and prefix rules as ``iter_cil_episode_bundles`` but does not
    read GI/KG JSON (RFC-076 progressive graph expansion).
    """
    anchor_s = os.path.normpath(anchor_path)
    root_s = os.path.normpath(root_path)
    if root_s != anchor_s and not root_s.startswith(anchor_s + os.sep):
        return
    root_prefix = root_s + os.sep
    anchor_prefix = anchor_s + os.sep

    if not os.path.isdir(anchor_s):
        return

    bridge_paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(anchor_s):
        dnorm = os.path.normpath(dirpath)
        if dnorm != anchor_s and not dnorm.startswith(anchor_prefix):
            continue
        if dnorm != root_s and not dnorm.startswith(root_prefix):
            continue
        for fn in filenames:
            if not fn.endswith(".bridge.json"):
                continue
            joined = os.path.normpath(os.path.join(dnorm, fn))
            if not joined.startswith(root_prefix):
                continue
            if os.path.isfile(joined):
                bridge_paths.append(joined)
    bridge_paths.sort()

    for safe_bridge in bridge_paths:
        if not safe_bridge.startswith(root_prefix):
            continue
        bridge = _read_json(safe_bridge)
        if bridge is None:
            continue
        yield safe_bridge, bridge


def _posix_relpath_under_corpus(corpus_root: str, abs_path: str) -> str | None:
    """Return POSIX path relative to corpus root, or None if outside."""
    try:
        root_r = Path(os.path.normpath(corpus_root)).resolve()
        file_r = Path(os.path.normpath(abs_path)).resolve()
        rel = file_r.relative_to(root_r)
    except (OSError, ValueError):
        return None
    return rel.as_posix()


def episodes_for_bridge_node_id(
    root_path: str,
    anchor_path: str,
    raw_node_id: str,
    *,
    max_episodes: int | None = None,
) -> tuple[list[dict[str, Any]], bool, int | None]:
    """Episodes whose bridge lists the canonical CIL id (RFC-076).

    Returns:
        ``(rows, truncated, total_matched)`` where each row has corpus-relative
        ``gi_relative_path``, ``kg_relative_path``, ``bridge_relative_path``, and
        optional ``episode_id``. Sorted by ``gi_relative_path``.
        ``total_matched`` is the pre-cap count when ``truncated`` is True; else
        ``None``.
    """
    canon = canonical_cil_entity_id(raw_node_id)
    if not canon:
        return [], False, None

    root_prefix = os.path.normpath(root_path) + os.sep
    matches: list[dict[str, Any]] = []
    seen_gi: set[str] = set()

    for safe_bridge, bridge in iter_cil_bridge_bundles(root_path, anchor_path):
        if canon not in _bridge_all_ids(bridge):
            continue
        parent = os.path.dirname(safe_bridge)
        name = os.path.basename(safe_bridge)
        if not name.endswith(".bridge.json"):
            continue
        stem = name[: -len(".bridge.json")]
        gi_j = os.path.normpath(os.path.join(parent, f"{stem}.gi.json"))
        kg_j = os.path.normpath(os.path.join(parent, f"{stem}.kg.json"))
        if not gi_j.startswith(root_prefix) or not kg_j.startswith(root_prefix):
            continue
        if not os.path.isfile(gi_j) or not os.path.isfile(kg_j):
            continue
        gi_rel = _posix_relpath_under_corpus(root_path, gi_j)
        kg_rel = _posix_relpath_under_corpus(root_path, kg_j)
        br_rel = _posix_relpath_under_corpus(root_path, safe_bridge)
        if not gi_rel or not kg_rel or not br_rel:
            continue
        if gi_rel in seen_gi:
            continue
        seen_gi.add(gi_rel)
        eid = _episode_id_from_bridge(bridge)
        matches.append(
            {
                "gi_relative_path": gi_rel,
                "kg_relative_path": kg_rel,
                "bridge_relative_path": br_rel,
                "episode_id": eid or None,
            }
        )

    matches.sort(key=lambda r: str(r.get("gi_relative_path") or ""))
    total = len(matches)
    truncated = False
    total_out: int | None = None
    if max_episodes is not None and max_episodes > 0 and total > max_episodes:
        truncated = True
        total_out = total
        matches = matches[:max_episodes]
    return matches, truncated, total_out


def _bridge_gi_ids(bridge: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in bridge.get("identities") or []:
        if not isinstance(row, dict):
            continue
        src = row.get("sources")
        if not isinstance(src, dict) or not src.get("gi"):
            continue
        rid = row.get("id")
        if isinstance(rid, str) and rid.strip():
            out.add(rid.strip())
    return out


def _bridge_all_ids(bridge: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in bridge.get("identities") or []:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if isinstance(rid, str) and rid.strip():
            out.add(rid.strip())
    return out


def _bridge_gi_topic_ids(bridge: dict[str, Any]) -> set[str]:
    """Topic ids with ``sources.gi == true`` — these are the ids that appear as
    GI ABOUT edge targets.  Used to expand KG-only cluster topic ids to their
    GI equivalents so timeline lookups find insights."""
    out: set[str] = set()
    for row in bridge.get("identities") or []:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip():
            continue
        if not rid.startswith("topic:"):
            continue
        srcs = row.get("sources")
        if isinstance(srcs, dict) and srcs.get("gi"):
            out.add(rid.strip())
    return out


def _kg_publish_date(kg: dict[str, Any]) -> str | None:
    for n in kg.get("nodes") or []:
        if not isinstance(n, dict) or n.get("type") != "Episode":
            continue
        props = n.get("properties")
        if not isinstance(props, dict):
            continue
        pd = props.get("publish_date")
        if isinstance(pd, str) and pd.strip():
            return pd.strip()
    return None


def _insight_type(node: dict[str, Any]) -> str:
    props = node.get("properties")
    if isinstance(props, dict):
        t = props.get("insight_type")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return "unknown"


def _position_hint(node: dict[str, Any]) -> float:
    props = node.get("properties")
    if isinstance(props, dict):
        raw = props.get("position_hint")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip())
            except ValueError:
                pass
    return 0.0


def _quote_ids_spoken_by_person(gi: dict[str, Any], person: str) -> set[str]:
    out: set[str] = set()
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "SPOKEN_BY":
            continue
        if str(e.get("to")) != person:
            continue
        qid = e.get("from")
        if qid is not None:
            out.add(str(qid))
    return out


def _insight_ids_supported_by_quotes(gi: dict[str, Any], spoken_quotes: set[str]) -> set[str]:
    out: set[str] = set()
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "SUPPORTED_BY":
            continue
        if str(e.get("to")) not in spoken_quotes:
            continue
        iid = e.get("from")
        if iid is not None:
            out.add(str(iid))
    return out


def _node_by_id(gi: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for n in gi.get("nodes") or []:
        if isinstance(n, dict) and str(n.get("id")) == node_id:
            return n
    return None


def _optional_position_hint(props: dict[str, Any]) -> float | None:
    ph = props.get("position_hint")
    if isinstance(ph, (int, float)):
        return float(ph)
    if isinstance(ph, str):
        try:
            return float(ph.strip())
        except ValueError:
            return None
    return None


def _episode_id_from_bridge(bridge: dict[str, Any]) -> str:
    eid = bridge.get("episode_id")
    return str(eid).strip() if isinstance(eid, str) and eid.strip() else ""


def _cil_episode_metadata_fields(
    safe_bridge: str,
    root_prefix: str,
    corpus_root_norm: str,
) -> dict[str, Any]:
    """Read sibling ``*.metadata.json`` for CIL episode display (RFC-072 UI).

    Artwork local paths match **Corpus Library** (``corpus_catalog``): only emit
    ``*_image_local_relpath`` when the file exists under ``corpus_root_norm`` and
    lies under ``.podcast_scraper/corpus-art/``. Otherwise the client falls back
    to remote ``*_image_url`` (avoids ``PodcastCover`` 404 + empty tile).
    """
    out: dict[str, Any] = {
        "episode_title": None,
        "feed_title": None,
        "episode_number": None,
        "episode_image_url": None,
        "episode_image_local_relpath": None,
        "feed_image_url": None,
        "feed_image_local_relpath": None,
    }
    corpus_root = Path(os.path.normpath(corpus_root_norm))
    parent = os.path.dirname(safe_bridge)
    name = os.path.basename(safe_bridge)
    if not name.endswith(".bridge.json"):
        return out
    stem = name[: -len(".bridge.json")]
    meta_j = os.path.normpath(os.path.join(parent, f"{stem}.metadata.json"))
    if not meta_j.startswith(root_prefix):
        return out
    if not os.path.isfile(meta_j):
        return out
    data = _read_json(meta_j)
    if data is None:
        return out
    feed = data.get("feed")
    if isinstance(feed, dict):
        ft = feed.get("title")
        if isinstance(ft, str) and ft.strip():
            out["feed_title"] = ft.strip()
        out["feed_image_url"] = _optional_image_url(feed.get("image_url"))
        feed_loc_raw = _optional_relpath_field(feed.get("image_local_relpath"))
        out["feed_image_local_relpath"] = _verified_artwork_relpath(corpus_root, feed_loc_raw)
    ep = data.get("episode")
    if isinstance(ep, dict):
        title = ep.get("title")
        if isinstance(title, str) and title.strip():
            out["episode_title"] = title.strip()
        en = ep.get("episode_number")
        if isinstance(en, int):
            out["episode_number"] = en
        elif isinstance(en, str) and en.strip().isdigit():
            out["episode_number"] = int(en.strip())
        out["episode_image_url"] = _optional_image_url(ep.get("image_url"))
        ep_loc_raw = _optional_relpath_field(ep.get("image_local_relpath"))
        out["episode_image_local_relpath"] = _verified_artwork_relpath(corpus_root, ep_loc_raw)
    return out


def position_arc(
    root_path: str,
    anchor_path: str,
    target_person: str,
    target_topic: str,
    insight_types: tuple[str, ...] | None = ("claim",),
) -> list[dict[str, Any]]:
    """RFC-072 Pattern A — chronological insights for person + topic."""
    person = canonical_cil_entity_id(target_person)
    topic = canonical_cil_entity_id(target_topic)
    root_prefix = os.path.normpath(root_path) + os.sep
    results: list[dict[str, Any]] = []
    for safe_bridge, bridge, gi, kg in iter_cil_episode_bundles(root_path, anchor_path):
        gi_ids = _bridge_gi_ids(bridge)
        if person not in gi_ids or topic not in gi_ids:
            continue

        spoken_quotes = _quote_ids_spoken_by_person(gi, person)

        about_topic: set[str] = set()
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "ABOUT":
                continue
            if _strip_layer_prefixes_for_cil(str(e.get("to"))) != topic:
                continue
            iid = e.get("from")
            if iid is not None:
                about_topic.add(str(iid))

        supported_insight_ids = _insight_ids_supported_by_quotes(gi, spoken_quotes)

        relevant_insights = about_topic & supported_insight_ids
        insights: list[dict[str, Any]] = []
        for n in gi.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            if nid is None or str(nid) not in relevant_insights:
                continue
            insights.append(n)

        if insight_types is not None:
            allowed = {x.strip().lower() for x in insight_types if x.strip()}
            if allowed:
                insights = [n for n in insights if _insight_type(n).lower() in allowed]

        insights.sort(key=_position_hint)
        if not insights:
            continue
        episode_id = _episode_id_from_bridge(bridge)
        if not episode_id:
            continue
        meta = _cil_episode_metadata_fields(
            safe_bridge,
            root_prefix,
            os.path.normpath(root_path),
        )
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "publish_date": _kg_publish_date(kg),
            "insights": insights,
        }
        row.update(meta)
        results.append(row)

    return sorted(results, key=lambda r: (r.get("publish_date") or "", r.get("episode_id") or ""))


def _person_profile_append_for_episode(
    bridge: dict[str, Any],
    gi: dict[str, Any],
    person: str,
    by_topic: dict[str, list[dict[str, Any]]],
    all_quotes: list[dict[str, Any]],
) -> None:
    spoken_quotes = _quote_ids_spoken_by_person(gi, person)
    supported_insights = _insight_ids_supported_by_quotes(gi, spoken_quotes)
    episode_id = _episode_id_from_bridge(bridge)
    if not episode_id:
        return

    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "ABOUT":
            continue
        if str(e.get("from")) not in supported_insights:
            continue
        topic_id = _strip_layer_prefixes_for_cil(str(e.get("to")))
        ins_id = str(e.get("from"))
        insight_node = _node_by_id(gi, ins_id)
        if insight_node is None:
            continue
        props = insight_node.get("properties")
        props_d = props if isinstance(props, dict) else {}
        by_topic.setdefault(topic_id, []).append(
            {
                "episode_id": episode_id,
                "insight": insight_node,
                "insight_type": _insight_type(insight_node),
                "position_hint": _optional_position_hint(props_d),
            }
        )

    for qid in spoken_quotes:
        quote_node = _node_by_id(gi, qid)
        if quote_node is None:
            continue
        all_quotes.append({"episode_id": episode_id, "quote": quote_node})


def person_profile(root_path: str, anchor_path: str, target_person: str) -> dict[str, Any]:
    """RFC-072 Pattern B — insights by topic + quotes for a person (Person Profile)."""
    person = canonical_cil_entity_id(target_person)
    by_topic: dict[str, list[dict[str, Any]]] = {}
    all_quotes: list[dict[str, Any]] = []

    for _safe_bridge, bridge, gi, _kg in iter_cil_episode_bundles(root_path, anchor_path):
        gi_ids = _bridge_gi_ids(bridge)
        if person not in gi_ids:
            continue
        _person_profile_append_for_episode(bridge, gi, person, by_topic, all_quotes)

    return {
        "person_id": person,
        "topics": by_topic,
        "quotes": all_quotes,
    }


def topic_timeline(
    root_path: str,
    anchor_path: str,
    target_topic: str,
    insight_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """RFC-072 Pattern C — insights about a topic across episodes."""
    topic = canonical_cil_entity_id(target_topic)
    root_prefix = os.path.normpath(root_path) + os.sep
    results: list[dict[str, Any]] = []
    for safe_bridge, bridge, gi, kg in iter_cil_episode_bundles(root_path, anchor_path):
        if topic not in _bridge_all_ids(bridge):
            continue

        # Expand to include GI-sourced topic ids from this episode's bridge.
        # KG-only topics (e.g. ``topic:economic-struggles``) have no GI ABOUT
        # edges; the GI counterparts (sentence-style slugs) do.  The bridge
        # establishes they belong to the same episode scope.
        match_ids = {topic} | _bridge_gi_topic_ids(bridge)

        about_insights: set[str] = set()
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "ABOUT":
                continue
            if _strip_layer_prefixes_for_cil(str(e.get("to"))) not in match_ids:
                continue
            iid = e.get("from")
            if iid is not None:
                about_insights.add(str(iid))

        insights: list[dict[str, Any]] = []
        for n in gi.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            if nid is None or str(nid) not in about_insights:
                continue
            insights.append(n)

        if insight_types is not None:
            allowed = {x.strip().lower() for x in insight_types if x.strip()}
            if allowed:
                insights = [n for n in insights if _insight_type(n).lower() in allowed]

        insights.sort(key=_position_hint)
        if not insights:
            continue
        episode_id = _episode_id_from_bridge(bridge)
        if not episode_id:
            continue
        meta = _cil_episode_metadata_fields(
            safe_bridge,
            root_prefix,
            os.path.normpath(root_path),
        )
        row_tl: dict[str, Any] = {
            "episode_id": episode_id,
            "publish_date": _kg_publish_date(kg),
            "insights": insights,
        }
        row_tl.update(meta)
        results.append(row_tl)

    return sorted(results, key=lambda r: (r.get("publish_date") or "", r.get("episode_id") or ""))


def topic_timeline_merged(
    root_path: str,
    anchor_path: str,
    target_topics: Sequence[str],
    insight_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """RFC-072 Pattern C for multiple topics — one scan of episode bundles.

    For each episode, includes insights ABOUT any of the canonical topic ids in
    ``target_topics`` (bridge intersection + GI ABOUT edges). Insight nodes are
    deduped by id within an episode. Sorting matches ``topic_timeline`` per row
    and across rows.
    """
    topics_set: set[str] = set()
    for t in target_topics:
        if t is None or not str(t).strip():
            continue
        topics_set.add(canonical_cil_entity_id(str(t)))
    if not topics_set:
        return []

    root_prefix = os.path.normpath(root_path) + os.sep
    results: list[dict[str, Any]] = []
    for safe_bridge, bridge, gi, kg in iter_cil_episode_bundles(root_path, anchor_path):
        bridge_ids = _bridge_all_ids(bridge)
        active = topics_set & bridge_ids
        if not active:
            continue

        # Expand: when a requested topic is in the bridge (KG or GI), also
        # accept all GI-sourced topic ids from this bridge.  GI ABOUT edges
        # only reference GI topic ids (long sentence-style slugs); KG-only
        # topics (short slugs from topic_clusters.json) never appear as ABOUT
        # targets.  The bridge proves both belong to this episode.
        match_ids = active | _bridge_gi_topic_ids(bridge)

        about_insights: set[str] = set()
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "ABOUT":
                continue
            to_id = _strip_layer_prefixes_for_cil(str(e.get("to")))
            if to_id not in match_ids:
                continue
            iid = e.get("from")
            if iid is not None:
                about_insights.add(str(iid))

        insights: list[dict[str, Any]] = []
        seen_nid: set[str] = set()
        for n in gi.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            if nid is None or str(nid) not in about_insights:
                continue
            ns = str(nid)
            if ns in seen_nid:
                continue
            seen_nid.add(ns)
            insights.append(n)

        if insight_types is not None:
            allowed = {x.strip().lower() for x in insight_types if x.strip()}
            if allowed:
                insights = [it for it in insights if _insight_type(it).lower() in allowed]

        insights.sort(key=_position_hint)
        if not insights:
            continue
        episode_id = _episode_id_from_bridge(bridge)
        if not episode_id:
            continue
        meta = _cil_episode_metadata_fields(
            safe_bridge,
            root_prefix,
            os.path.normpath(root_path),
        )
        row_tl: dict[str, Any] = {
            "episode_id": episode_id,
            "publish_date": _kg_publish_date(kg),
            "insights": insights,
        }
        row_tl.update(meta)
        results.append(row_tl)

    return sorted(results, key=lambda r: (r.get("publish_date") or "", r.get("episode_id") or ""))


def person_topic_ids(root_path: str, anchor_path: str, target_person: str) -> list[str]:
    """Distinct topic ids linked to ``target_person`` via grounded GI edges."""
    profile = person_profile(root_path, anchor_path, target_person)
    topics = profile.get("topics")
    if not isinstance(topics, dict):
        return []
    return sorted({str(k) for k in topics.keys() if k})


def topic_person_ids(root_path: str, anchor_path: str, target_topic: str) -> list[str]:
    """Distinct person ids that speak (via quotes) to insights about ``target_topic``."""
    topic = canonical_cil_entity_id(target_topic)
    persons: set[str] = set()
    for _safe_bridge, bridge, gi, _kg in iter_cil_episode_bundles(root_path, anchor_path):
        if topic not in _bridge_all_ids(bridge):
            continue

        about_insights: set[str] = set()
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "ABOUT":
                continue
            if _strip_layer_prefixes_for_cil(str(e.get("to"))) != topic:
                continue
            iid = e.get("from")
            if iid is not None:
                about_insights.add(str(iid))

        quote_by_insight: dict[str, set[str]] = {}
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "SUPPORTED_BY":
                continue
            ins = str(e.get("from"))
            q = e.get("to")
            if ins in about_insights and q is not None:
                quote_by_insight.setdefault(ins, set()).add(str(q))

        for _ins, qids in quote_by_insight.items():
            for e in gi.get("edges") or []:
                if not isinstance(e, dict):
                    continue
                if normalize_gil_edge_type(e.get("type")) != "SPOKEN_BY":
                    continue
                if str(e.get("from")) not in qids:
                    continue
                pid = e.get("to")
                if pid is not None and str(pid).startswith("person:"):
                    persons.add(str(pid))

    return sorted(persons)
