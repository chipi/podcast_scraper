"""Derive meaning-bearing relational edges for the GIL semantic model (#874 family).

Two derivable, no-ML, no-diarization edges that the search-powered surfaces (PRD-033)
need from the foundation:

- ``Insight ─MENTIONS_PERSON→ Person`` / ``Insight ─MENTIONS_ORG→ Organization``
  — an insight concerns a person/org. Cross-layer: the target is a KG
  ``person:``/``org:`` id (unified in the cross-layer graph). Matched by the
  entity's surface name appearing verbatim in the insight text. Grounds FR4
  entity views / Topic Entity View.
- ``Podcast ─HAS_EPISODE→ Episode`` — ties an episode to its canonical show node so
  show-name navigation (FR3.4) and show-scoped views have a real edge, not hashed ids.

RFC-097 v3.0 split the generic ``MENTIONS`` into the typed ``MENTIONS_PERSON`` /
``MENTIONS_ORG`` so the viewer + relational query layer (RFC-094) can style/filter
descriptive edges without re-reading node types. Schema-permissive: artifacts with
the legacy generic ``MENTIONS`` still read; the post-pass here emits typed edges
and bumps ``schema_version`` to ``3.0``.

Both are persisted into the gi.json artifact. Conservative + idempotent.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Set, Tuple

from ..identity.slugify import slugify


def add_insight_entity_edges(
    artifact: Dict,
    entity_index: Mapping[str, Tuple[str, str]],
    *,
    nlp: Any = None,
) -> int:
    """Add typed ``Insight ─MENTIONS_PERSON/ORG→`` edges to *artifact* in place.

    *entity_index* maps ``entity_id`` (``person:``/``org:`` slug) → ``(name, kind)``
    where ``kind`` is ``"person"`` or ``"organization"``. An edge is added when
    the name appears as a whole-word phrase in an insight's text.

    Edge types: ``MENTIONS_PERSON`` (when kind=person) or ``MENTIONS_ORG`` (when
    kind=organization). Permissive over the legacy generic ``MENTIONS`` edges —
    deduplication considers all three types so repeated calls don't double up.

    RFC-097 chunk-4 retroactive sweep: when an edge points to a Person /
    Organization id that does NOT already exist as a node in the artifact,
    the function adds a minimal ``{name, label}`` Person / Organization
    node so the edge resolves locally and the viewer doesn't have to
    join across kg.json to render the relationship.

    #1076 chunk 4-A — optional spaCy NER pass (operator-gated via
    ``cfg.gi_typed_mentions_use_ner``). When *nlp* is provided, the
    function additionally extracts PERSON spans from each Insight's
    text and emits edges for spans that resolve against an
    entity_index entry by case-insensitive token overlap (spans like
    "Maya" match the index entry "Maya Hutchinson"; the substring
    regex above can't). Catches BART-paraphrased name fragments under
    airgapped_thin. False-positive bound: span must be ≥3 chars + at
    least one token must match the index entry, so single-token
    matches against indexed multi-word names work but spurious
    spaCy detections like "AI" (≥3-letter words tagged as PERSON
    by mistake) won't resolve to anyone.

    When at least one typed edge is added, the artifact's ``schema_version``
    is bumped to ``3.0`` (RFC-097); earlier versions otherwise pass through.

    Idempotent. Returns the number of edges added.
    """
    nodes = artifact.setdefault("nodes", [])
    edges = artifact.setdefault("edges", [])
    insights = [
        (n["id"], (n.get("properties") or {}).get("text") or "")
        for n in nodes
        if n.get("type") == "Insight" and isinstance(n.get("id"), str)
    ]
    patterns = [
        (eid, kind, name, re.compile(r"\b" + re.escape(name) + r"\b"))
        for eid, (name, kind) in entity_index.items()
        if name and len(name) >= 2
    ]
    existing = {
        (e.get("from"), e.get("to"), e.get("type"))
        for e in edges
        if e.get("type") in ("MENTIONS", "MENTIONS_PERSON", "MENTIONS_ORG")
    }
    existing_node_ids = {n.get("id") for n in nodes if isinstance(n.get("id"), str)}
    added = 0
    for insight_id, text in insights:
        for entity_id, kind, surface_name, pattern in patterns:
            edge_type = "MENTIONS_ORG" if kind == "organization" else "MENTIONS_PERSON"
            # Skip if EITHER the legacy generic MENTIONS or the typed edge is already present.
            keys = {
                (insight_id, entity_id, "MENTIONS"),
                (insight_id, entity_id, "MENTIONS_PERSON"),
                (insight_id, entity_id, "MENTIONS_ORG"),
            }
            if keys & existing:
                continue
            if pattern.search(text):
                # Add the missing Person / Organization node so the edge target
                # resolves within gi.json (no cross-layer join required).
                if entity_id not in existing_node_ids:
                    node_type = "Organization" if kind == "organization" else "Person"
                    nodes.append(
                        {
                            "id": entity_id,
                            "type": node_type,
                            "properties": {"name": surface_name},
                        }
                    )
                    existing_node_ids.add(entity_id)
                edges.append({"type": edge_type, "from": insight_id, "to": entity_id})
                existing.add((insight_id, entity_id, edge_type))
                added += 1
    # #1076 chunk 4-A — optional spaCy NER pass. Runs after the regex
    # pass so anything the substring match caught stays caught (the
    # dedup key check prevents double edges); the NER pass adds the
    # paraphrase-fragment misses that the regex couldn't reach.
    if nlp is not None:
        added += _apply_ner_mentions_pass(
            insights=insights,
            entity_index=entity_index,
            nlp=nlp,
            nodes=nodes,
            edges=edges,
            existing=existing,
            existing_node_ids=existing_node_ids,
        )

    if added and isinstance(artifact.get("schema_version"), str):
        # Bump to v3.0 (RFC-097) on first typed-edge addition; preserves existing higher versions.
        if artifact["schema_version"] in ("1.0", "2.0"):
            artifact["schema_version"] = "3.0"
    return added


def add_episode_show_edges(artifact: Dict, show_title: str) -> int:
    """Add a ``Podcast`` node + ``Podcast ─HAS_EPISODE→ Episode`` edges in place.

    The canonical show node is ``podcast:{slug(show_title)}`` so it unifies across
    episodes of the same show (cross-show navigation). Idempotent. Returns edges added.

    RFC-097 chunk-4 retroactive sweep: Podcast node carries ``title`` (not ``name``)
    to match ``gi.schema.json`` ``podcast_node.properties`` which requires ``title``.
    ``rss_url`` is unknowable from ``show_title`` alone and stays absent — see the
    schema's optional ``rss_url`` annotation.
    """
    if not show_title:
        return 0
    nodes = artifact.setdefault("nodes", [])
    edges = artifact.setdefault("edges", [])
    podcast_id = "podcast:" + slugify(show_title)
    episode_ids = [
        n["id"] for n in nodes if n.get("type") == "Episode" and isinstance(n.get("id"), str)
    ]
    node_ids = {n.get("id") for n in nodes}
    existing = {(e.get("from"), e.get("to")) for e in edges if e.get("type") == "HAS_EPISODE"}
    if podcast_id not in node_ids:
        nodes.append({"id": podcast_id, "type": "Podcast", "properties": {"title": show_title}})
    added = 0
    for episode_id in episode_ids:
        if (podcast_id, episode_id) not in existing:
            edges.append({"type": "HAS_EPISODE", "from": podcast_id, "to": episode_id})
            existing.add((podcast_id, episode_id))
            added += 1
    return added


def kg_entity_names(kg_artifact: Dict) -> Dict[str, str]:
    """Extract ``{entity_id: surface_name}`` from a KG artifact.

    Permissive over v1.x ``Entity`` and v2.0 ``Person`` / ``Organization``
    node types (RFC-097). Kept for backward compatibility with consumers
    that only need names; for typed-MENTIONS edges use :func:`kg_entity_index`.
    """
    from ..graph_id_utils import is_person_or_org_node

    out: Dict[str, str] = {}
    for node in kg_artifact.get("nodes") or []:
        if not is_person_or_org_node(node.get("type")):
            continue
        eid = node.get("id")
        name = (node.get("properties") or {}).get("name")
        if isinstance(eid, str) and isinstance(name, str) and name.strip():
            out[eid] = name.strip()
    return out


def kg_entity_index(kg_artifact: Dict) -> Dict[str, Tuple[str, str]]:
    """Extract ``{entity_id: (surface_name, kind)}`` from a KG artifact.

    ``kind`` is ``"person"`` or ``"organization"``, derived from the node type
    (v2.0 ``Person`` / ``Organization``) or properties (v1.x ``Entity`` with
    ``kind`` / ``entity_kind``). Required input for typed-MENTIONS emission
    via :func:`add_insight_entity_edges` (RFC-097 v3.0).
    """
    from ..graph_id_utils import is_person_or_org_node, normalized_entity_kind_from_node

    out: Dict[str, Tuple[str, str]] = {}
    for node in kg_artifact.get("nodes") or []:
        if not is_person_or_org_node(node.get("type")):
            continue
        eid = node.get("id")
        name = (node.get("properties") or {}).get("name")
        if isinstance(eid, str) and isinstance(name, str) and name.strip():
            kind = normalized_entity_kind_from_node(node)
            out[eid] = (name.strip(), kind)
    return out


def apply_typed_mentions_to_gi_artifact(
    gi_payload: Dict,
    kg_payload: Dict,
    *,
    nlp: Any = None,
) -> int:
    """Apply the RFC-097 v3.0 typed-MENTIONS post-pass to a GI artifact.

    The live GI emit path (``gi/pipeline.py::build_artifact``) does not
    materialize typed ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` cross-layer
    edges — they need a KG entity index that isn't available inside the
    GI build alone. This helper plugs the gap: caller passes in the
    finished KG payload, helper extracts the entity index and calls the
    in-place mutator.

    Mutates *gi_payload* in place. Caller owns any I/O (e.g. re-writing
    the updated artifact back to disk). Returns the number of typed edges
    added. Idempotent — calling repeatedly with the same payloads is a
    no-op after the first call.

    Args:
        gi_payload: GI artifact dict (mutated in place when edges are added).
        kg_payload: KG artifact dict (read-only).

    Returns:
        Count of typed edges added (0 if KG had no Person/Org entities, or
        if every match was already typed).
    """
    entity_index = kg_entity_index(kg_payload)
    if not entity_index:
        return 0
    return add_insight_entity_edges(gi_payload, entity_index, nlp=nlp)


def apply_typed_mentions_and_rewrite_gi(
    gi_payload: Dict,
    kg_payload: Dict,
    gi_path: str,
    *,
    nlp: Any = None,
) -> int:
    """Apply the typed-MENTIONS post-pass and re-write the GI artifact on disk.

    Composes :func:`apply_typed_mentions_to_gi_artifact` with a guarded
    disk re-write — re-writes only when edges are actually added so the
    on-disk artifact's mtime stays stable for the zero-op case (no KG
    entities matched, or every match was already typed).

    Used by the production orchestrator
    (:mod:`podcast_scraper.workflow.metadata_generation`). Caller must
    have validated that both payloads exist and the path is non-empty.

    Args:
        gi_payload: GI artifact dict (mutated in place when edges added).
        kg_payload: KG artifact dict (read-only).
        gi_path: Filesystem path to re-write when edges are added.

    Returns:
        Count of typed edges added. ``0`` means the on-disk artifact was
        left untouched.
    """
    from pathlib import Path

    from .io import write_artifact

    added = apply_typed_mentions_to_gi_artifact(gi_payload, kg_payload, nlp=nlp)
    if added > 0:
        write_artifact(Path(gi_path), gi_payload, validate=True)
    return added


def _apply_ner_mentions_pass(
    *,
    insights: List[Tuple[str, str]],
    entity_index: Mapping[str, Tuple[str, str]],
    nlp: Any,
    nodes: List[Any],
    edges: List[Any],
    existing: Set[Tuple[Any, Any, Any]],
    existing_node_ids: Set[Any],
) -> int:
    """#1076 chunk 4-A NER pass — extracted from ``add_insight_entity_edges``
    to keep the leaf function under the C901 complexity gate.

    Mutates *nodes*, *edges*, *existing*, *existing_node_ids* in place. See
    the leaf function's docstring for the matching contract: spans
    must be ≥3 chars and a token-subset of the indexed entity name.
    """
    # Pre-build per-entity name-token lookups so the NER pass resolves
    # spaCy spans in O(spans × entities), not O(spans × entities × tokens).
    index_entries: List[Tuple[str, str, str, Set[str]]] = []
    for eid, (name, kind) in entity_index.items():
        if not name or len(name) < 2:
            continue
        name_tokens = {t for t in re.split(r"\W+", name.lower()) if t}
        if name_tokens:
            index_entries.append((eid, kind, name, name_tokens))

    added = 0
    for insight_id, text in insights:
        if not text or not text.strip():
            continue
        try:
            doc = nlp(text)
        except Exception:  # nosec B112
            # Defensive: a malformed spaCy doc on one Insight shouldn't kill the pass.
            continue
        seen_span_norms: Set[str] = set()
        for ent in getattr(doc, "ents", []) or []:
            if getattr(ent, "label_", None) != "PERSON":
                continue
            span_text = (ent.text or "").strip()
            if len(span_text) < 3:
                continue
            norm = span_text.lower()
            if norm in seen_span_norms:
                continue
            seen_span_norms.add(norm)
            span_tokens = {t for t in re.split(r"\W+", norm) if t}
            if not span_tokens:
                continue
            added += _resolve_span_to_index(
                insight_id=insight_id,
                span_tokens=span_tokens,
                index_entries=index_entries,
                nodes=nodes,
                edges=edges,
                existing=existing,
                existing_node_ids=existing_node_ids,
            )
    return added


def _resolve_span_to_index(
    *,
    insight_id: str,
    span_tokens: Set[str],
    index_entries: List[Tuple[str, str, str, Set[str]]],
    nodes: List[Any],
    edges: List[Any],
    existing: Set[Tuple[Any, Any, Any]],
    existing_node_ids: Set[Any],
) -> int:
    """Resolve one spaCy span to zero-or-more entity_index entries via
    the token-subset rule, emitting the typed edge for each match.

    Mutates the shared edge/node/existing collections in place.
    Returns the count of edges added by this span.

    #1076 chunk 4-A — shared-surname disambiguation. When the span is
    a single token (e.g. "Trump") and multiple KG entries share that
    surname (e.g. "Donald Trump", "Eric Trump"), reject the match — we
    can't tell which one the text refers to and emitting an edge to
    every candidate scatters wrong attributions across the corpus.
    Multi-token spans like "Donald Trump" are still resolved
    deterministically because the disambiguation is per-distinct-
    candidate-set, not per-span.
    """
    # Pre-pass: gather all index entries whose token set is a superset of
    # the span. When the span is a single token shared by ≥2 candidates,
    # bail with zero edges — emitting to all of them would scatter wrong
    # attributions; emitting to one would be arbitrary.
    candidates: List[Tuple[str, str, str, Set[str]]] = []
    for entry in index_entries:
        _, _, _, name_tokens = entry
        if span_tokens & name_tokens and span_tokens.issubset(name_tokens):
            candidates.append(entry)
    if len(candidates) > 1 and len(span_tokens) == 1:
        # Ambiguous single-token (typically surname-only) match against
        # multiple KG entries that share that token. Conservative reject.
        return 0
    added = 0
    for entity_id, kind, surface_name, name_tokens in candidates:
        edge_type = "MENTIONS_ORG" if kind == "organization" else "MENTIONS_PERSON"
        keys = {
            (insight_id, entity_id, "MENTIONS"),
            (insight_id, entity_id, "MENTIONS_PERSON"),
            (insight_id, entity_id, "MENTIONS_ORG"),
        }
        if keys & existing:
            continue
        if entity_id not in existing_node_ids:
            node_type = "Organization" if kind == "organization" else "Person"
            nodes.append(
                {
                    "id": entity_id,
                    "type": node_type,
                    "properties": {"name": surface_name},
                }
            )
            existing_node_ids.add(entity_id)
        edges.append({"type": edge_type, "from": insight_id, "to": entity_id})
        existing.add((insight_id, entity_id, edge_type))
        added += 1
    return added
