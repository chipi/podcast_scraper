"""Read a KG 2.x artifact as ontology v3: the one adapter (docs/architecture/corpus/ontology-v3.md,
"Reading v2 artifacts as v3"; E04 step 10).

Podcasts keep writing KG 2.x until R3b rewrites the corpus (ADR-178). A reader that takes items of
several types reads every KG through :func:`kg_as_v3`, so it has one code path: a KG 3.0
artifact comes back as it is, and a KG 2.x one in v3's structure:

- the root ``episode_id`` is also the ``item_id``, with ``content_type: podcast_episode``;
- ``Podcast`` reads as ``Source`` (``source_kind: podcast``) and ``Episode`` as ``ContentItem``,
  ``podcast_id`` as ``feed_id`` and ``publish_date`` as ``published_at``; a KG without a
  ``Podcast`` node gets its source id from the episode's ``podcast_id`` (``podcast:{id}``);
- ``HAS_EPISODE`` reads as ``HAS_ITEM``, ``HOSTS`` and ``GUESTS_ON`` as ``CONTRIBUTES_TO`` with
  that role; every edge's ends are ``from`` and ``to`` (older KGs wrote ``source`` and
  ``target``);
- ``extraction.transcript_ref`` reads as ``document_ref``, with its ``variant`` from the file
  name: ``.cleaned.txt`` is ``cleaned``, ``.adfree.txt`` ``adfree``, ``<base>.<lang>.txt`` the
  source-language original (``raw``), and ``<base>.txt`` ``raw`` unless the episode's
  ``language`` is not English, when it is the English render (``en``; translation D-44).

It is a view, and lossless: node ids never change (T-8), and every other node, edge and property
is kept as written, including ones v3 does not define (``topic_source``, ``respelled_from``, a
legacy ``Entity``). So a view passes the v3 shape checks, not strict validation: making the files
strictly v3 is R3b's migration, which decides what each extra property becomes. A view carries
``adapted_from`` (the version it was read from) and is never written back. The input is never
modified.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Mapping, Optional

from podcast_scraper.content_types import PODCAST_EPISODE

_ROLE_BY_V2_EDGE = {"HOSTS": "host", "GUESTS_ON": "guest"}
_LANG_SUFFIX = re.compile(r"\.([a-z]{2,3})\.txt$")


def is_v3(kg: Mapping[str, Any]) -> bool:
    """Whether ``kg`` is already a KG 3.x artifact."""
    return str(kg.get("schema_version") or "").startswith("3")


def document_variant(document_ref: str, language: Optional[str] = None) -> str:
    """The ``variant`` of a podcast transcript, from its file name (and the episode's language)."""
    name = document_ref.rsplit("/", 1)[-1]
    if name.endswith(".cleaned.txt"):
        return "cleaned"
    if name.endswith(".adfree.txt"):
        return "adfree"
    if _LANG_SUFFIX.search(name):
        return "raw"
    primary = (language or "").split("-", 1)[0].strip().lower()
    return "en" if primary and primary != "en" else "raw"


def kg_as_v3(kg: Mapping[str, Any], *, language: Optional[str] = None) -> Dict[str, Any]:
    """``kg`` read as ontology v3 (this module's docstring). ``language`` is the episode's, from its
    metadata, for a transcript ``variant`` the file name alone cannot tell."""
    if is_v3(kg):
        return dict(kg)
    out: Dict[str, Any] = copy.deepcopy(dict(kg))
    nodes = out.get("nodes") or []
    podcast = next((n for n in nodes if isinstance(n, dict) and n.get("type") == "Podcast"), None)
    episode = next((n for n in nodes if isinstance(n, dict) and n.get("type") == "Episode"), None)
    # The root episode_id, else the Episode node's; a KG naming no episode gets no item_id (never
    # an empty one).
    item_id = str(kg.get("episode_id") or "")
    if not item_id and episode:
        props_id = (episode.get("properties") or {}).get("episode_id")
        item_id = str(props_id or str(episode.get("id") or "").removeprefix("episode:"))
    if item_id:
        out["item_id"] = item_id
    out["content_type"] = PODCAST_EPISODE
    feed_id = str(((episode or {}).get("properties") or {}).get("podcast_id") or "")
    source_id = str(podcast["id"]) if podcast else (f"podcast:{feed_id}" if feed_id else "")

    for node in nodes:
        if not isinstance(node, dict):
            continue
        props = node.setdefault("properties", {})
        if node.get("type") == "Podcast":
            node["type"] = "Source"
            props.setdefault("source_kind", "podcast")
        elif node.get("type") == "Episode":
            node["type"] = "ContentItem"
            props.setdefault("content_type", PODCAST_EPISODE)
            if source_id:
                props.setdefault("source_id", source_id)
            if "podcast_id" in props:
                props.setdefault("feed_id", props.pop("podcast_id"))
            if "publish_date" in props:
                props.setdefault("published_at", props.pop("publish_date"))

    for edge in out.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if "from" not in edge and "source" in edge:
            edge["from"] = edge.pop("source")
        if "to" not in edge and "target" in edge:
            edge["to"] = edge.pop("target")
        kind = edge.get("type")
        if kind == "HAS_EPISODE":
            edge["type"] = "HAS_ITEM"
        elif kind in _ROLE_BY_V2_EDGE:
            edge["type"] = "CONTRIBUTES_TO"
            edge.setdefault("properties", {})["role"] = _ROLE_BY_V2_EDGE[kind]

    extraction = out.get("extraction")
    if isinstance(extraction, dict) and extraction.get("transcript_ref"):
        ref = str(extraction.pop("transcript_ref"))
        extraction.setdefault("document_ref", ref)
        extraction.setdefault("variant", document_variant(ref, language))
    # A view, never an artifact: ``adapted_from`` says so, and :func:`is_view` lets a writer refuse
    # to save it as if it were a KG 3.0 file.
    out["adapted_from"] = str(kg.get("schema_version") or "")
    out["schema_version"] = "3.0"
    return out


def is_view(kg: Mapping[str, Any]) -> bool:
    """Whether ``kg`` is a :func:`kg_as_v3` view of a KG 2.x artifact, not a KG 3.0 artifact."""
    return "adapted_from" in kg
