"""Stable graph node ids for GI and KG artifacts.

Episode nodes share ``episode:{episode_id}`` across layers. Topic, Entity, and Person
use global slug-style ids. Insight and Quote
use opaque hashes keyed by episode + content so they stay unique without embedding
episode id in the string (``properties.episode_id`` remains the anchor).

Canonical label→slug rules live in ``podcast_scraper.identity.slugify``;
this module applies graph-specific fallbacks (``topic``, ``unknown``) and length caps.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional

from podcast_scraper.identity.slugify import slugify as canonical_slugify

#: v2.0 (RFC-097) plus legacy: node types treated as Person/Org "entity-like".
PERSON_ORG_NODE_TYPES = frozenset({"Entity", "Person", "Organization"})

#: A bare diarization label the roster never resolved to a real person: ``SPEAKER_00``,
#: ``Speaker 3``, ``speaker-12``. Diarization numbers are assigned per-episode and are NOT
#: stable across episodes, so a bare label must be episode-scoped, never a global id (#1b).
_BARE_SPEAKER_LABEL_RE = re.compile(r"^\s*speaker[\s_\-]*\d+\s*$", re.IGNORECASE)


def is_bare_speaker_label(name: Optional[str]) -> bool:
    """True when *name* is an unresolved diarization label (``SPEAKER_03``), not a real name."""
    return bool(name and _BARE_SPEAKER_LABEL_RE.match(str(name)))


#: Words dropped when canonicalising a TOPIC label (#1933). Prepositions and articles are what
#: split one concept across variants — ``ai in education`` / ``ai education`` / ``ai and
#: education`` are one topic written three ways, and as separate ids they can never co-occur.
_TOPIC_STOPWORDS = frozenset({"the", "a", "an", "of", "and", "in", "for", "to", "on", "with"})

#: ``-``, ``_`` and ``/`` are WORD SEPARATORS in a topic label, not punctuation to strip. Treating
#: them as punctuation is what made an earlier measurement of this conclude only 0.8% of topics
#: collapse: ``open-source ai models`` became ``opensourceaimodels`` and never matched
#: ``open source ai models`` — two 16-episode topics, one hyphen apart (#1933).
_TOPIC_WORD_SEPARATORS = re.compile(r"[-_/]+")
_TOPIC_NON_WORD = re.compile(r"[^a-z0-9\s]+")


def _crude_singular(word: str) -> str:
    """Enough singularisation for topic labels; deliberately not a stemmer.

    ``creator economies``/``creator economy`` and ``dark factories``/``dark factory`` are the same
    topic. A real stemmer would over-merge (``policy``/``police``), which is worse here than
    missing a plural.
    """
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def canonical_topic_slug(label: str, max_len: int = 80) -> str:
    """Slug for a TOPIC label, canonicalised so spelling variants share one id (#1933).

    Lowercase; ``-_/`` become word separators; articles/prepositions dropped; crude
    singularisation; remaining tokens sorted. Two labels that mean the same thing collapse to the
    same slug, so they co-occur instead of splitting the concept's frequency across variants.

    Measured on the 2,618-label prod topic set: 50 families / 109 labels / 229 episodes of
    frequency currently split. Merging them raised pairs seen in >=3 episodes from 1 to 8.

    Sorting tokens is the aggressive part, and it is deliberate: it is what merges
    ``us-china ai competition`` with ``china-us ai competition`` and ``strait of hormuz
    disruption`` with ``hormuz strait disruption``.

    KNOWN false positives, reviewed across all 50 families on 2026-09-06 — both caused by
    dropping the preposition, which is the SAME mechanism that correctly merges ``ai in
    education`` with ``ai education``:

    * ``ai investment`` vs ``ai in investment`` — investing in AI vs using AI for investing;
    * ``ai scientist agents`` vs ``ai agents for scientists`` — agents that ARE scientists vs
      agents serving scientists.

    Both are 1+1 episodes. 229 episodes correctly merged against ~4 incorrectly is the trade, and
    it is accepted knowingly rather than unnoticed.

    Only the ID is canonicalised. The DISPLAY label is untouched — a reader still sees
    ``ai in education``, not ``ai education``.
    """
    base = (label or "").strip()
    if not base:
        return "topic"
    text = _TOPIC_WORD_SEPARATORS.sub(" ", base.lower())
    text = _TOPIC_NON_WORD.sub(" ", text)
    tokens = [_crude_singular(w) for w in text.split() if w and w not in _TOPIC_STOPWORDS]
    if not tokens:
        # A label that is ENTIRELY stopwords ("the a of") has no concept in it. Fall back to the
        # plain slug rather than returning the shared "topic" bucket, which would merge every
        # such label into one id.
        return slugify_label(base, max_len=max_len)
    return slugify_label(" ".join(sorted(tokens)), max_len=max_len)


def _scoped_speaker_person_id(label: str, episode_id: str) -> str:
    """Episode-scoped person id for an unnamed diarization voice: ``person:speaker-{ep}-{n}``.

    Keyed on (episode, label number) so ``SPEAKER_00`` in two different episodes — which may be
    two different people — never collapses into one phantom cross-episode person (#1b). Stays
    recognisable as a placeholder (``speaker-…-\\d+``) for the corpus-scope drop filters.
    """
    num = re.sub(r"\D", "", label) or "0"
    ep_slug = slugify_label(str(episode_id))
    return f"person:speaker-{ep_slug}-{num}"


def slugify_label(label: str, max_len: int = 80) -> str:
    """Lowercase filesystem-safe slug from a human label (CIL slugify + KG max length).

    Empty or unslugifiable labels return ``topic`` so topic dedupe in GI/KG stays safe.
    """
    base = (label or "").strip()
    if not base:
        return "topic"
    try:
        s = canonical_slugify(base)
    except ValueError:
        return "topic"
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s


def episode_node_id(episode_id: str) -> str:
    """Episode anchor node id (shared by GI and KG)."""
    return f"episode:{episode_id}"


def topic_node_id_from_slug(slug: str) -> str:
    """Topic node id from an already-normalized slug."""
    return f"topic:{slug}"


def entity_node_id(entity_kind: str, name: str, episode_id: Optional[str] = None) -> str:
    """KG entity node id: ``person:{slug}`` or ``org:{slug}``.

    When ``episode_id`` is given and ``name`` is a bare diarization label (``SPEAKER_03``), the
    person id is episode-scoped so the same anonymous label in different episodes never merges
    into one phantom person (#1b). Real names and orgs keep their global slug id.
    """
    ek = entity_kind if entity_kind in ("person", "organization") else "person"
    base = (name or "").strip()
    if episode_id and ek == "person" and is_bare_speaker_label(base):
        return _scoped_speaker_person_id(base, episode_id)
    slug = slugify_label(base) if base else "unknown"
    if ek == "organization":
        return f"org:{slug}"
    return f"person:{slug}"


def person_node_id(display_name: str, episode_id: Optional[str] = None) -> str:
    """GIL person node id from a diarization / display name.

    Real names → global ``person:{slug}`` (a recurring named person is one node across episodes).
    A bare diarization label (``SPEAKER_00``) the roster never resolved is episode-scoped when
    ``episode_id`` is supplied (``person:speaker-{ep}-{n}``) so anonymous voices don't merge
    across episodes (#1b).
    """
    base = (display_name or "").strip()
    if not base:
        return "person:unknown"
    if episode_id and is_bare_speaker_label(base):
        return _scoped_speaker_person_id(base, episode_id)
    try:
        slug = canonical_slugify(base)
    except ValueError:
        return "person:unknown"
    return f"person:{slug}"


def gil_insight_node_id(episode_id: str, index: int, insight_text: str) -> str:
    """Opaque insight id (stable for fixed episode, index, and text prefix)."""
    text_key = (insight_text or "").strip()[:2000]
    payload = f"{episode_id}\0{index}\0{text_key}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()[:16]
    return f"insight:{h}"


def gil_quote_node_id(
    episode_id: str,
    quote_index: int,
    quote_text: str,
    char_start: Any,
    char_end: Any,
) -> str:
    """Opaque quote id (stable for fixed episode, index, span, and text prefix)."""
    text_key = (quote_text or "").strip()[:2000]
    payload = f"{episode_id}\0{quote_index}\0{text_key}\0{char_start}\0{char_end}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()[:16]
    return f"quote:{h}"


def is_person_or_org_node(node_type: Any) -> bool:
    """True for both legacy ``Entity`` and v2.0 ``Person`` / ``Organization`` (RFC-097)."""
    return isinstance(node_type, str) and node_type in PERSON_ORG_NODE_TYPES


def normalized_entity_kind_from_node(node: Dict[str, Any]) -> str:
    """Return ``"person"`` | ``"organization"`` regardless of v1.x or v2.0 shape.

    v1.x: ``Entity`` node with ``properties.kind`` = ``"person"`` / ``"org"`` (v1.2)
    or ``properties.entity_kind`` = ``"person"`` / ``"organization"`` (legacy).
    v2.0 (RFC-097): ``Person`` or ``Organization`` node (kind encoded in node type).
    """
    nt = node.get("type") if isinstance(node, dict) else None
    if nt == "Person":
        return "person"
    if nt == "Organization":
        return "organization"
    props = node.get("properties") if isinstance(node, dict) else None
    if not isinstance(props, dict):
        return "person"
    raw = props.get("kind")
    if raw == "org":
        return "organization"
    if raw == "person":
        return "person"
    raw_ek = props.get("entity_kind")
    if raw_ek == "organization":
        return "organization"
    if raw_ek == "person":
        return "person"
    return "person"
