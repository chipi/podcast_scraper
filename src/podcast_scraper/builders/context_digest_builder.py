"""Per-episode ``context.json`` — a flat, denormalized, reprocess-free content digest.

ADR-134. Answers "what is this episode ABOUT?" for downstream consumers (search, future
translation, batch processes) without re-parsing the graph or reprocessing the episode. It is a
**cache/view**: the graph artifacts remain the source of truth; this is a convenience surface.

Phase 1 is a pure, deterministic denormalization of outputs we already produce (no new LLM):

- ``basic``     — feed/episode facts (title, show, date, duration, language, hosts, guests)
- ``summary``   — the episode summary text (already generated)
- ``people``    — ``Person`` node names, EXCLUDING bare-speaker ``Voice`` (#1220): resolved humans
- ``companies`` — ``Organization`` node names
- ``topics``    — ``Topic`` node labels
- ``voices``    — unresolved diarization speakers: ``{total, labels[]}`` (the bare ``SPEAKER_NN``);
                  the ``unknown``/``unidentified`` split is added when a per-voice classification is
                  supplied (persisted nowhere today — a follow-up; backfill omits it)

Deferred to Phase 2+ (new LLM, ADR-134 non-goals): glossary/terminology, distinct concepts, jokes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..graph_id_utils import is_bare_speaker_label

CONTEXT_SCHEMA_VERSION = "1.0"


def _node_name(node: Mapping[str, Any]) -> Optional[str]:
    """Display name for an entity node — ``label`` for Topic, else ``name``."""
    props = node.get("properties")
    if not isinstance(props, dict):
        return None
    val = props.get("label") if node.get("type") == "Topic" else props.get("name")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _dedup_sorted(names: List[str]) -> List[str]:
    """Case-insensitive dedup (first-seen display kept), returned sorted."""
    seen: Dict[str, str] = {}
    for n in names:
        key = n.casefold()
        if key not in seen:
            seen[key] = n
    return sorted(seen.values(), key=str.casefold)


def _iter_nodes(*artifacts: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for art in artifacts:
        for node in (art or {}).get("nodes") or []:
            if isinstance(node, dict):
                out.append(node)
    return out


def _is_bare_voice(node: Mapping[str, Any]) -> bool:
    """True when a Person/Voice node is an unresolved diarization speaker (SPEAKER_NN)."""
    if node.get("type") not in ("Person", "Voice"):
        return False
    return bool(is_bare_speaker_label(_node_name(node)) or is_bare_speaker_label(node.get("id")))


def _basic_block(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    feed = metadata.get("feed") or {}
    episode = metadata.get("episode") or {}
    content = metadata.get("content") or {}
    return {
        "title": episode.get("title"),
        "show": feed.get("title"),
        "published_date": episode.get("published_date"),
        "duration_seconds": episode.get("duration_seconds"),
        "language": feed.get("language"),
        "hosts": list(content.get("detected_hosts") or []),
        "guests": list(content.get("detected_guests") or []),
    }


def _summary_text(metadata: Mapping[str, Any]) -> Optional[str]:
    summ = metadata.get("summary")
    if isinstance(summ, dict):
        for key in ("summary", "long_summary", "short_summary"):
            val = summ.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    elif isinstance(summ, str) and summ.strip():
        return summ.strip()
    return None


def _voices_block(
    bare_labels: List[str],
    voice_classification: Optional[Mapping[str, str]],
) -> Dict[str, Any]:
    """``{total, labels}``; adds the ``unknown``/``unidentified`` split when a per-voice
    classification (``{label: voice_type}``) is supplied — otherwise those stay ``None`` (the
    classification is persisted nowhere today, so the backfill cannot recover it)."""
    labels = sorted(set(bare_labels))
    block: Dict[str, Any] = {"total": len(labels), "labels": labels}
    if voice_classification:
        block["unknown"] = sorted(
            lbl for lbl in labels if voice_classification.get(lbl) == "unknown"
        )
        block["unidentified"] = sorted(
            lbl for lbl in labels if voice_classification.get(lbl) == "unidentified"
        )
    else:
        block["unknown"] = None
        block["unidentified"] = None
    return block


def build_context_digest(
    episode_id: str,
    *,
    gi_artifact: Optional[Mapping[str, Any]],
    kg_artifact: Optional[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    voice_classification: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build the ``context.json`` payload (ADR-134, Phase 1).

    Args:
        episode_id: Episode identifier.
        gi_artifact / kg_artifact: the GI / KG artifact dicts (or None).
        metadata: the ``metadata.json`` dict — supplies ``basic`` + ``summary``.
        voice_classification: optional ``{SPEAKER_NN: voice_type}`` map to fill the voices split;
            omit for the backfill (the per-voice type is not persisted in existing artifacts).

    Returns:
        The context digest dict — deterministic given the same inputs.
    """
    nodes = _iter_nodes(gi_artifact, kg_artifact)

    people: List[str] = []
    companies: List[str] = []
    topics: List[str] = []
    bare_labels: List[str] = []
    for node in nodes:
        name = _node_name(node)
        if name is None:
            continue
        ntype = node.get("type")
        if ntype in ("Person", "Voice"):
            if _is_bare_voice(node):
                bare_labels.append(name)
            elif ntype == "Person":
                people.append(name)
        elif ntype == "Organization":
            companies.append(name)
        elif ntype == "Topic":
            topics.append(name)

    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "episode_id": str(episode_id),
        "basic": _basic_block(metadata),
        "summary": _summary_text(metadata),
        "people": _dedup_sorted(people),
        "companies": _dedup_sorted(companies),
        "topics": _dedup_sorted(topics),
        "voices": _voices_block(bare_labels, voice_classification),
        "source": {
            "gi_schema_version": (gi_artifact or {}).get("schema_version"),
            "kg_schema_version": (kg_artifact or {}).get("schema_version"),
        },
    }
