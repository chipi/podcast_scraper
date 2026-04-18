"""CIL topic pills for digest and library (RFC-072 bridge + RFC-075 clusters)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.schemas import CilDigestTopicPill
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)

CIL_TOPIC_PILL_CAP = 5

_TOPIC_CLUSTERS_REL = os.path.join("search", "topic_clusters.json")


@dataclass(frozen=True)
class TopicClusterIndex:
    """Membership rows: topic_id appears in a multi-member cluster with optional episode scope."""

    rows: tuple[tuple[str, str, frozenset[str] | None], ...]
    """(topic_id, graph_compound_parent_id, episode_ids or None when JSON omitted/empty)."""

    @classmethod
    def empty(cls) -> TopicClusterIndex:
        """Return an index with no cluster membership rows."""
        return cls(rows=())

    def cluster_for_topic(
        self,
        topic_id: str,
        episode_id: str | None,
    ) -> tuple[bool, str | None]:
        """Return (in_cluster, compound_id) for this topic in context of episode_id.

        Each pill exposes at most one ``tc:…`` parent: if the index lists the same
        ``topic_id`` more than once with different compound ids, the first matching
        row in ``rows`` wins (deterministic; bad data should be fixed upstream).
        """
        tid = topic_id.strip()
        eid = episode_id.strip() if isinstance(episode_id, str) and episode_id.strip() else None
        compound: str | None = None
        matched = False
        for t, comp, ep_set in self.rows:
            if t != tid:
                continue
            if ep_set is None:
                matched = True
                compound = compound or comp
            elif eid is not None and eid in ep_set:
                matched = True
                compound = compound or comp
        return matched, compound

    def episode_listed_on_cluster_member(self, topic_id: str, episode_id: str | None) -> bool:
        """True when *topic_id* appears on a cluster row that lists *episode_id* explicitly.

        Rows with empty ``episode_ids`` (legacy / hand-authored JSON) are ignored so the
        Library ``topic_cluster_only`` filter narrows to episodes that participate in a
        cluster with provenance, not every episode that merely shares a clustered topic id.
        """
        tid = topic_id.strip()
        eid = episode_id.strip() if isinstance(episode_id, str) and episode_id.strip() else None
        if not eid:
            return False
        for t, _comp, ep_set in self.rows:
            if t != tid:
                continue
            if ep_set and eid in ep_set:
                return True
        return False


def _read_json_object(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.loads(fh.read())
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("cil_digest_topics: skip read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _humanize_topic_id(topic_id: str) -> str:
    s = topic_id.strip()
    if s.startswith("topic:"):
        s = s[6:]
    s = s.replace("-", " ").replace("_", " ").strip()
    if not s:
        return topic_id.strip()
    return s[:1].upper() + s[1:] if len(s) > 1 else s.upper()


def load_topic_cluster_index(corpus_root: Path) -> TopicClusterIndex:
    """Load ``search/topic_clusters.json`` when present; else empty index."""
    root = corpus_root.resolve()
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    joined = os.path.normpath(os.path.join(root_s, _TOPIC_CLUSTERS_REL.replace("\\", "/")))
    if joined != root_s and not joined.startswith(safe_prefix):
        return TopicClusterIndex.empty()
    if not os.path.isfile(joined):
        return TopicClusterIndex.empty()
    doc = _read_json_object(joined)
    if doc is None:
        return TopicClusterIndex.empty()
    clusters = doc.get("clusters")
    if not isinstance(clusters, list):
        return TopicClusterIndex.empty()
    out_rows: list[tuple[str, str, frozenset[str] | None]] = []
    for cl in clusters:
        if not isinstance(cl, dict):
            continue
        members = cl.get("members")
        if not isinstance(members, list) or len(members) < 2:
            continue
        compound_raw = cl.get("graph_compound_parent_id") or cl.get("cluster_id")
        compound = str(compound_raw).strip() if isinstance(compound_raw, str) else ""
        if not compound:
            continue
        for m in members:
            if not isinstance(m, dict):
                continue
            tid_raw = m.get("topic_id")
            if not isinstance(tid_raw, str) or not tid_raw.strip():
                continue
            tid = tid_raw.strip()
            ep_ids: list[str] = []
            raw_eps = m.get("episode_ids")
            if isinstance(raw_eps, list):
                for x in raw_eps:
                    if isinstance(x, str) and x.strip():
                        ep_ids.append(x.strip())
            ep_frozen: frozenset[str] | None = frozenset(ep_ids) if ep_ids else None
            out_rows.append((tid, compound, ep_frozen))
    return TopicClusterIndex(rows=tuple(out_rows))


def _bridge_topic_entries(bridge: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Ordered (topic_id, label) from bridge identities (topics only)."""
    identities = bridge.get("identities")
    if not isinstance(identities, list):
        return []
    out: list[tuple[str, str]] = []
    for row in identities:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip():
            continue
        tid = rid.strip()
        if not tid.startswith("topic:"):
            continue
        dn = row.get("display_name")
        label = dn.strip() if isinstance(dn, str) and dn.strip() else _humanize_topic_id(tid)
        out.append((tid, label))
    return out


def build_cil_digest_topics_for_row(
    corpus_root: Path,
    row: CatalogEpisodeRow,
    index: TopicClusterIndex,
) -> list[CilDigestTopicPill]:
    """Build cluster-first capped CIL topic pills for one catalog row."""
    if not row.has_bridge:
        return []
    safe_bridge = safe_relpath_under_corpus_root(corpus_root, row.bridge_relative_path)
    if not safe_bridge or not os.path.isfile(safe_bridge):
        return []
    bridge = _read_json_object(safe_bridge)
    if bridge is None:
        return []
    entries = _bridge_topic_entries(bridge)
    if not entries:
        return []
    cluster_block: list[CilDigestTopicPill] = []
    rest_block: list[CilDigestTopicPill] = []
    for tid, label in entries:
        in_cl, compound = index.cluster_for_topic(tid, row.episode_id)
        pill = CilDigestTopicPill(
            topic_id=tid,
            label=label,
            in_topic_cluster=in_cl,
            topic_cluster_compound_id=compound if in_cl else None,
        )
        if in_cl:
            cluster_block.append(pill)
        else:
            rest_block.append(pill)
    merged = cluster_block + rest_block
    return merged[:CIL_TOPIC_PILL_CAP]


def row_has_cluster_topic(pills: Sequence[CilDigestTopicPill]) -> bool:
    """True when any pill marks RFC-075 cluster membership for this row."""
    return any(p.in_topic_cluster for p in pills)


def row_matches_library_topic_cluster_filter(
    corpus_root: Path,
    row: CatalogEpisodeRow,
    index: TopicClusterIndex,
) -> bool:
    """Library episode-list filter: bridge topic tied to this episode on a cluster member.

    Stricter than digest pills: requires ``episode_ids`` on the matching cluster member
    so the list narrows to episodes that actually appear in cluster membership data.
    """
    if not row.has_bridge:
        return False
    safe_bridge = safe_relpath_under_corpus_root(corpus_root, row.bridge_relative_path)
    if not safe_bridge or not os.path.isfile(safe_bridge):
        return False
    bridge = _read_json_object(safe_bridge)
    if bridge is None:
        return False
    entries = _bridge_topic_entries(bridge)
    if not entries:
        return False
    eid = row.episode_id
    for tid, _label in entries:
        if index.episode_listed_on_cluster_member(tid, eid):
            return True
    return False
