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
from typing import Any, Iterator

from podcast_scraper.gi.edge_normalization import normalize_gil_edge_type

logger = logging.getLogger(__name__)


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
) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Yield ``(bridge, gi, kg)`` dicts for each sibling triple.

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
        yield bridge, gi, kg


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


def position_arc(
    root_path: str,
    anchor_path: str,
    target_person: str,
    target_topic: str,
    insight_types: tuple[str, ...] | None = ("claim",),
) -> list[dict[str, Any]]:
    """RFC-072 Pattern A — chronological insights for person + topic."""
    person = target_person.strip()
    topic = target_topic.strip()
    results: list[dict[str, Any]] = []
    for bridge, gi, kg in iter_cil_episode_bundles(root_path, anchor_path):
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
            if str(e.get("to")) != topic:
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
        results.append(
            {
                "episode_id": episode_id,
                "publish_date": _kg_publish_date(kg),
                "insights": insights,
            }
        )

    return sorted(results, key=lambda r: (r.get("publish_date") or "", r.get("episode_id") or ""))


def _guest_brief_append_for_episode(
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
        topic_id = str(e.get("to"))
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


def guest_brief(root_path: str, anchor_path: str, target_person: str) -> dict[str, Any]:
    """RFC-072 Pattern B — insights by topic + quotes for a person."""
    person = target_person.strip()
    by_topic: dict[str, list[dict[str, Any]]] = {}
    all_quotes: list[dict[str, Any]] = []

    for bridge, gi, _kg in iter_cil_episode_bundles(root_path, anchor_path):
        gi_ids = _bridge_gi_ids(bridge)
        if person not in gi_ids:
            continue
        _guest_brief_append_for_episode(bridge, gi, person, by_topic, all_quotes)

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
    topic = target_topic.strip()
    results: list[dict[str, Any]] = []
    for bridge, gi, kg in iter_cil_episode_bundles(root_path, anchor_path):
        if topic not in _bridge_all_ids(bridge):
            continue

        about_insights: set[str] = set()
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "ABOUT":
                continue
            if str(e.get("to")) != topic:
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
        results.append(
            {
                "episode_id": episode_id,
                "publish_date": _kg_publish_date(kg),
                "insights": insights,
            }
        )

    return sorted(results, key=lambda r: (r.get("publish_date") or "", r.get("episode_id") or ""))


def person_topic_ids(root_path: str, anchor_path: str, target_person: str) -> list[str]:
    """Distinct topic ids linked to ``target_person`` via grounded GI edges."""
    brief = guest_brief(root_path, anchor_path, target_person)
    topics = brief.get("topics")
    if not isinstance(topics, dict):
        return []
    return sorted({str(k) for k in topics.keys() if k})


def topic_person_ids(root_path: str, anchor_path: str, target_topic: str) -> list[str]:
    """Distinct person ids that speak (via quotes) to insights about ``target_topic``."""
    topic = target_topic.strip()
    persons: set[str] = set()
    for bridge, gi, _kg in iter_cil_episode_bundles(root_path, anchor_path):
        if topic not in _bridge_all_ids(bridge):
            continue

        about_insights: set[str] = set()
        for e in gi.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if normalize_gil_edge_type(e.get("type")) != "ABOUT":
                continue
            if str(e.get("to")) != topic:
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
