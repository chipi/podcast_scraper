"""KG 3.0 as stored (R3b, the KG half; ``docs/wip/changes/0051_kg_3_0.md``).

:func:`to_v3_file` writes a KG 2.x artifact as a KG 3.0 file: the :func:`kg_as_v3` view without what
only a view carries (``adapted_from``) and without the root ``episode_id`` KG 3.0 does not allow.
When the publisher registry (``config/publishers.yaml``) lists the item's ``Source`` id, the file
gains ``PUBLISHED_BY`` from that ``Source`` to the publisher's ``org:`` id, with the publisher's
``Organization`` node (role ``publisher``) when the file has none (ADR-180 rule 6 as amended: an
identical id is one organisation). A KG 3.x artifact comes back as it is. R3b's migration writes it;
the KG writer writes it from R3b on.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

from podcast_scraper.kg.as_v3 import is_v3, kg_as_v3
from podcast_scraper.publishers import Registry


def _published_by(out: Dict[str, Any], registry: Registry) -> None:
    nodes = [n for n in out.get("nodes") or [] if isinstance(n, dict)]
    edges = out.setdefault("edges", [])
    for source in (n for n in nodes if n.get("type") == "Source"):
        publisher = registry.for_source(str(source.get("id")))
        if publisher is None:
            continue
        if not any(n.get("id") == publisher.id for n in nodes):
            org = {
                "id": publisher.id,
                "type": "Organization",
                "properties": {
                    "name": publisher.name,
                    "label": publisher.name,
                    "role": "publisher",
                },
            }
            out["nodes"].append(org)
            nodes.append(org)
        edge = {"type": "PUBLISHED_BY", "from": source["id"], "to": publisher.id}
        if edge not in edges:
            edges.append(edge)


def to_v3_file(
    kg: Mapping[str, Any],
    *,
    language: Optional[str] = None,
    registry: Optional[Registry] = None,
) -> Dict[str, Any]:
    """``kg`` (KG 2.x) as a KG 3.0 file (this module's docstring)."""
    if is_v3(kg) and "adapted_from" not in kg:
        out = copy.deepcopy(dict(kg))
    else:
        out = kg_as_v3(kg, language=language)
        out.pop("adapted_from", None)
        out.pop("episode_id", None)
    if registry is not None:
        _published_by(out, registry)
    return out
