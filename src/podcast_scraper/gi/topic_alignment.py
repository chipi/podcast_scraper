"""Align GI Topic nodes with the KG's canonical topic labels (#585 / #653 Part D).

GI derives its Topic nodes from summary bullets, which produces sentence-style slugs
("stop-being-skeptical-about-ai-for-coding"). KG derives noun-phrase labels ("AI coding tools").
The CIL bridge merges the two graphs by exact node ID, so unless GI's topics are replaced with
KG's, the merge produces two disconnected topic vocabularies for one episode.

EXTRACTED so there is exactly one implementation. This ran inline in
``metadata_generation.generate_episode_metadata`` and rewrote ``gi.json`` after the fact. Any
tool that rebuilds a GI artifact outside that function — notably ``gi.repair`` — must apply the
same alignment or it writes an artifact that is subtly WORSE than a normally-produced one:
correct insights, wrong topic vocabulary, broken bridge. A second copy of these 20 lines would
drift the first time either changed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def kg_topic_labels(kg_payload: Dict[str, Any]) -> List[str]:
    """The KG's canonical topic labels, in document order."""
    return [
        (n.get("properties") or {}).get("label", "")
        for n in (kg_payload.get("nodes") or [])
        if isinstance(n, dict)
        and n.get("type") == "Topic"
        and (n.get("properties") or {}).get("label")
    ]


def align_gi_topics_with_kg(
    gi_payload: Dict[str, Any],
    kg_payload: Dict[str, Any],
) -> int:
    """Replace GI's bullet-derived Topic nodes with KG's, in place. Returns topics applied.

    No-op (returns 0) when the KG declares no topics — better to keep GI's own topics than to
    strip an episode's topic vocabulary down to nothing.

    Mutates ``gi_payload``; the caller persists it.
    """
    labels = kg_topic_labels(kg_payload)
    if not labels:
        return 0

    from .pipeline import _dedupe_topic_node_specs

    nodes = [n for n in (gi_payload.get("nodes") or []) if n.get("type") != "Topic"]
    edges = [e for e in (gi_payload.get("edges") or []) if e.get("type") != "ABOUT"]

    topic_specs = _dedupe_topic_node_specs(labels)
    for tid, label in topic_specs:
        nodes.append({"id": tid, "type": "Topic", "properties": {"label": label}})

    insight_ids = [n["id"] for n in nodes if n.get("type") == "Insight"]
    for iid in insight_ids:
        for tid, _label in topic_specs:
            edges.append({"type": "ABOUT", "from": iid, "to": tid})

    gi_payload["nodes"] = nodes
    gi_payload["edges"] = edges
    return len(topic_specs)
