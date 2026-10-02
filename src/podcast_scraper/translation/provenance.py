"""Translation provenance on every claim (RFC-124 / RFC-125 / S2.11).

WHAT THIS IS FOR, AND WHY IT IS NOT OPTIONAL. In v1 there is no user-visible "translated from
Spanish" marker (D-36), so a listener cannot tell a translated quote from a native-English one.
The compensation is that the PROVENANCE is written on every claim at the time it is made, which
is what lets v2 add the label, a quality score or a verification pass **without reprocessing**.
A claim that shipped without it can never acquire it, because the units that produced it are
not recoverable after the fact.

THE HASH IS OVER THE LABEL-FREE UNITS, NOT THE RENDERED ENGLISH (D-40). Speaker labels are
re-applied by the renderer and a rename changes every `Label:` prefix and therefore the whole
`<base>.txt`. Hashing the render would invalidate the provenance on every claim whenever naming is
re-run, for a reason that has nothing to do with the translation. The unit texts are already
label-free by construction (D-24), so the stable thing to hash already exists.

The block goes in node ``properties`` so it travels with the node into the KG and out to every
reader, rather than living in a side table nobody joins.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

from .artifacts import resolve_units_for_span, TranslationDocument

logger = logging.getLogger(__name__)

#: Node types whose spans index the transcript and therefore carry provenance.
SPAN_BEARING_TYPES = ("Quote", "Evidence")

PROVENANCE_KEY = "translation"


def target_units_sha256(doc: TranslationDocument) -> str:
    """Hash of the LABEL-FREE English unit texts, in unit order (D-40).

    Deliberately excludes the rendered `<base>.txt`, speaker labels and every char offset: all
    three move when a voice is renamed, and none of them is the translation. Two episodes whose
    translations are identical hash identically even if their speakers were named differently.
    """
    h = hashlib.sha256()
    h.update((doc.source_language or "").encode("utf-8"))
    for unit in doc.units:
        if not unit.ok:
            continue
        h.update(b"\x01")
        h.update(unit.unit_id.encode("utf-8"))
        # The unit's OWN model, not the document's: on a partial resume the document's model
        # describes only the units that were re-translated, so hashing it would compute the
        # whole episode's identity under a model most of it never saw.
        h.update(b"\x03")
        h.update((unit.model or doc.model or "").encode("utf-8"))
        for sentence in unit.sentences:
            h.update(b"\x02")
            h.update(str(sentence.get("en_text") or "").encode("utf-8"))
    return h.hexdigest()


def build_provenance_block(
    doc: TranslationDocument, unit_ids: Sequence[str], *, en_sha256: Optional[str] = None
) -> Dict[str, Any]:
    """The ``translation`` block for one claim.

    ``model`` and ``prompt_sha256`` are derived from the UNITS this claim resolves to, not from
    the document. A partial resume mixes models — run 1 under A, one turn edited, run 2 under
    B — and a document-level value would stamp B on every claim including those made of A's
    output. When the resolved units disagree, BOTH are listed: a claim spanning two models is a
    fact about that claim, and flattening it to one would be a guess.
    """
    by_id = {u.unit_id: u for u in doc.units}
    resolved = [by_id[uid] for uid in unit_ids if uid in by_id]

    def _spread(values: Sequence[Optional[str]], fallback: Optional[str]) -> Any:
        present = [v for v in values if v]
        distinct = sorted(set(present))
        if not distinct:
            return fallback
        return distinct[0] if len(distinct) == 1 else distinct

    return {
        "translated": True,
        "source_language": doc.source_language,
        "model": _spread([u.model for u in resolved], doc.model),
        "unit_ids": list(unit_ids),
        "en_sha256": en_sha256 or target_units_sha256(doc),
        # The prompt's own hash, so a claim can be traced to the exact instruction that produced
        # its text — a changed prompt changes the translation without changing the model id.
        "prompt_sha256": _spread(
            [u.prompt_sha256 for u in resolved], (doc.prompt or {}).get("sha256")
        ),
    }


def attach_translation_provenance(
    payload: Dict[str, Any],
    doc: TranslationDocument,
    en_segments: Sequence[Dict[str, Any]],
) -> int:
    """Decorate every span-bearing node with its ``translation`` block. Returns how many.

    IDEMPOTENT, because three different writers produce ``gi.json`` — the artifact builder,
    ``add_spoken_by_edges(replace=True)`` and ``gi/repair.py`` — and a node that passed through
    two of them must not end up with two blocks or a half-updated one.

    A node whose span resolves to NO units gets no block rather than an empty one. An empty
    ``unit_ids`` would read as "translated, from nothing", which is worse than absence: it
    asserts provenance while carrying none.
    """
    if not doc.complete:
        # Provenance describes a translation that was actually published. An incomplete one has
        # no the translation on disk (RFC-124 §5.3), so no claim should exist to decorate.
        logger.debug("translation provenance: skipped, the translation is incomplete")
        return 0

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return 0
    en_sha = target_units_sha256(doc)
    decorated = 0
    for node in nodes:
        if not isinstance(node, dict) or node.get("type") not in SPAN_BEARING_TYPES:
            continue
        props = node.get("properties")
        if not isinstance(props, dict):
            continue
        start, end = props.get("char_start"), props.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        unit_ids = resolve_units_for_span(en_segments, start, end)
        if not unit_ids:
            props.pop(PROVENANCE_KEY, None)
            continue
        props[PROVENANCE_KEY] = build_provenance_block(doc, unit_ids, en_sha256=en_sha)
        decorated += 1
    return decorated


def provenance_coverage(payload: Dict[str, Any]) -> Dict[str, int]:
    """``{span_nodes, with_provenance}`` — what an audit checks.

    The number that matters is the GAP: a span-bearing node without provenance on a translated
    episode is a claim v2 can never label, and it is invisible unless counted.
    """
    nodes = payload.get("nodes") or []
    span_nodes = [
        n
        for n in nodes
        if isinstance(n, dict)
        and n.get("type") in SPAN_BEARING_TYPES
        and isinstance((n.get("properties") or {}).get("char_start"), int)
    ]
    with_prov = [n for n in span_nodes if (n.get("properties") or {}).get(PROVENANCE_KEY)]
    return {"span_nodes": len(span_nodes), "with_provenance": len(with_prov)}


def load_for_provenance(rel_transcript_path: str, effective_output_dir: str) -> Optional[tuple]:
    """``(doc, en_adfree_segments)`` for an episode, or ``None`` when it was not translated.

    Reads the AD-FREE English segments, because that is the coordinate space GI's ``char_start``
    lives in (``TranscriptPurpose.ANALYSIS``). Using the non-ad-free ones would resolve every
    span against offsets shifted by however much ad text was removed — silently, and by a
    different amount per episode.
    """
    import json
    import os

    from .artifacts import load_translation_json

    doc = load_translation_json(rel_transcript_path, effective_output_dir)
    if doc is None or not doc.complete:
        return None
    base, _ = os.path.splitext(rel_transcript_path)
    # D-44: the translation sits at the CANONICAL sidecar names. Ad-free first, because that is
    # the ANALYSIS coordinate space a claim's offsets were computed in; the full-timeline
    # sidecar is the fallback for an episode with no ad-free base.
    for rel in (f"{base}.adfree.segments.json", f"{base}.segments.json"):
        path = os.path.join(effective_output_dir, rel)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                segments = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(segments, list) and segments:
            return doc, segments
    return None


#: The episode currently being processed, as ``(doc, en_adfree_segments)``.
#:
#: WHY A CONTEXTVAR AND NOT A PARAMETER. ``gi/io.write_artifact`` is the single choke point every
#: ``gi.json`` writer passes through — and there are more than ten call sites, including
#: ``add_spoken_by_edges(replace=True)``, ``gi/repair.py``, the bridge post-pass and topic
#: clustering. S2.11 requires provenance from EVERY one of them, and a parameter threaded to ten
#: sites is a parameter that will be missed at the eleventh. The alternative was deriving the
#: transcript path from the artifact's own path, which is corpus-layout archaeology in a
#: low-level writer.
_EPISODE_TRANSLATION: contextvars.ContextVar[
    Optional[Tuple[TranslationDocument, Sequence[Dict[str, Any]]]]
] = contextvars.ContextVar("podcast_scraper_episode_translation", default=None)


def publish_episode_translation(
    doc: Optional[TranslationDocument], en_segments: Optional[Sequence[Dict[str, Any]]]
) -> None:
    """Set the ambient translation for the episode now being processed.

    UNCONDITIONAL, AND THAT IS WHAT MAKES IT SAFE. Every episode passes the translation seam
    exactly once and calls this exactly once — with its own translation, or with ``None`` for an
    English one. So the value is always the current episode's, and a worker thread reused for a
    second episode cannot decorate it with the first episode's units. Nothing writes ``gi.json``
    before the seam, so there is no window where a stale value could be read.

    The alternative was a context manager wrapping the rest of ``generate_episode_metadata``,
    which would have meant splitting a very large function with many return paths — more risk
    than the leak it would prevent.
    """
    _EPISODE_TRANSLATION.set((doc, en_segments or []) if doc is not None else None)


@contextlib.contextmanager
def episode_translation(
    doc: Optional[TranslationDocument], en_segments: Optional[Sequence[Dict[str, Any]]]
) -> Iterator[None]:
    """Publish this episode's translation so every ``gi.json`` written inside is decorated.

    A ``None`` doc (an English episode, or an incomplete translation) publishes nothing, so the
    decoration is a no-op and English episodes pay one contextvar read.
    """
    token = _EPISODE_TRANSLATION.set((doc, en_segments or []) if doc is not None else None)
    try:
        yield
    finally:
        _EPISODE_TRANSLATION.reset(token)


def decorate_from_context(payload: Dict[str, Any]) -> int:
    """Attach provenance from the ambient episode, if any. Safe to call unconditionally."""
    current = _EPISODE_TRANSLATION.get()
    if current is None:
        return 0
    doc, segments = current
    try:
        return attach_translation_provenance(payload, doc, segments)
    except Exception:  # noqa: BLE001 — provenance must never fail an artifact write
        logger.warning("translation provenance: decoration failed", exc_info=True)
        return 0


def _iter_span_nodes(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for node in payload.get("nodes") or []:
        if isinstance(node, dict) and node.get("type") in SPAN_BEARING_TYPES:
            yield node


__all__ = [
    "PROVENANCE_KEY",
    "decorate_from_context",
    "episode_translation",
    "publish_episode_translation",
    "SPAN_BEARING_TYPES",
    "attach_translation_provenance",
    "build_provenance_block",
    "target_units_sha256",
    "load_for_provenance",
    "provenance_coverage",
]
