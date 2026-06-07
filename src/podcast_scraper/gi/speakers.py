"""Speaker attribution for GIL quotes — diarized transcript → Person / SPOKEN_BY (#874/#875).

Two transcript shapes are handled, in priority order:

1. **Named markers (#875)** — the new diarization writes a *named* screenplay
   (``Maya: …`` / ``Liam: …`` / ``Priya: …``); each line-start marker that matches a
   detected person is attributed **directly** to ``person:{slug}``. This is robust
   regardless of speaker count, so panels / multi-guest episodes work — no role
   heuristic needed.
2. **Generic ``Speaker N:`` markers (#874)** — anonymous publisher diarization (e.g.
   pre-diarized ``direct_download`` transcripts). Mapped to people via the host/guest
   **role heuristic** (opening cluster → host, dominant other → guest). This degrades
   on panels and is used only when no named markers are present.

**Honesty:** publisher labels (e.g. "Bloomberg") are never attributed as a person; a
quote with no confident speaker stays ``None`` (under-attributed beats wrong).
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from ..identity.slugify import person_id

_SPEAKER_RE = re.compile(r"Speaker\s*(\d+)\s*:")
# Line-start ``<Label>: `` turn markers (named screenplay). Constrained to line start
# + whitespace after the colon so it captures diarized turns, not mid-prose "Word:".
_NAMED_TURN_RE = re.compile(r"(?m)^[ \t]*([^\n:]{1,60}?)[ \t]*:[ \t]")

# Tokens that mark a "host" string as a publisher/network, not a person name.
_NON_PERSON_TOKENS = frozenset(
    {
        "bloomberg",
        "industries",
        "media",
        "podcast",
        "podcasts",
        "network",
        "news",
        "studios",
        "inc",
        "llc",
    }
)


def build_speaker_turns(transcript: str) -> List[Tuple[int, str]]:
    """Sorted ``[(char_offset, "Speaker N")]`` turn starts parsed from *transcript*."""
    return [(m.start(), f"Speaker {m.group(1)}") for m in _SPEAKER_RE.finditer(transcript)]


def speaker_for_char(char_start: int, turns: Sequence[Tuple[int, str]]) -> Optional[str]:
    """The speaker cluster whose turn contains *char_start* (last marker at/before it)."""
    spk: Optional[str] = None
    for off, label in turns:
        if off <= char_start:
            spk = label
        else:
            break
    return spk


def _is_publisher_label(name: Optional[str]) -> bool:
    """True when *name* contains a publisher/network token (not a person)."""
    return any(t.lower().strip(".,") in _NON_PERSON_TOKENS for t in (name or "").split())


def _looks_like_person(name: Optional[str]) -> bool:
    """Heuristic: a real person name has ≥2 tokens and no publisher/network token."""
    toks = (name or "").split()
    return len(toks) >= 2 and not _is_publisher_label(name)


def _detected_person_lookup(hosts: Sequence[str], guests: Sequence[str]) -> Dict[str, str]:
    """``{lowercased name: canonical name}`` for detected people (publishers excluded).

    Used to validate named screenplay markers: a marker is only attributed when its
    label matches a detected host/guest and is not a publisher label. Single-token
    first names (e.g. "Maya") are valid here — unlike the role-heuristic host check,
    which requires ≥2 tokens.
    """
    out: Dict[str, str] = {}
    for name in list(hosts) + list(guests):
        s = (name or "").strip()
        if s and not _is_publisher_label(s):
            out.setdefault(s.lower(), s)
    return out


def build_named_turns(transcript: str, known_names: Dict[str, str]) -> List[Tuple[int, str]]:
    """``[(char_offset, canonical_name)]`` for line-start ``<Name>:`` markers (#875).

    Only markers whose label matches a *detected* person (``known_names``) are kept, so
    prose like ``Note:`` / ``Q:`` is ignored and only real diarized speakers attribute.
    """
    turns: List[Tuple[int, str]] = []
    for m in _NAMED_TURN_RE.finditer(transcript):
        canonical = known_names.get(m.group(1).strip().lower())
        if canonical:
            turns.append((m.start(1), canonical))
    return turns


def map_clusters_to_people(
    turns: Sequence[Tuple[int, str]],
    *,
    hosts: Sequence[str],
    guests: Sequence[str],
) -> Dict[str, Optional[str]]:
    """Map each speaker cluster → person name (or ``None``) via the role heuristic.

    - **Guest** = the dominant cluster that is not the opening speaker → first detected
      guest. This is the reliable mapping (the opening speaker is the host doing the
      intro; the most-speaking non-host is the interviewed guest).
    - **Host** = opening cluster → first detected host, *only* if that host string looks
      like a person (not a publisher label).
    - Everything else → ``None`` (under-attributed rather than wrongly attributed).
    """
    if not turns:
        return {}
    counts = Counter(label for _, label in turns)
    order = list(OrderedDict.fromkeys(label for _, label in turns))
    opening = order[0]
    others = [(label, c) for label, c in counts.items() if label != opening]
    guest_cluster = max(others, key=lambda lc: lc[1])[0] if others else None

    out: Dict[str, Optional[str]] = {label: None for label in counts}
    if guest_cluster and guests:
        out[guest_cluster] = guests[0]
    if hosts and _looks_like_person(hosts[0]):
        out[opening] = hosts[0]
    return out


def attribute_quote_speakers(
    transcript: str,
    quote_char_starts: Dict[str, Optional[int]],
    *,
    hosts: Sequence[str],
    guests: Sequence[str],
) -> Dict[str, str]:
    """Attribute quotes to canonical ``person:{slug}`` ids.

    *quote_char_starts* maps ``quote_id -> char_start``. Returns ``{quote_id:
    person_id}`` for confidently-attributed quotes only (unattributable quotes are
    omitted, leaving ``speaker_id`` ``None`` as today).

    #875: when the transcript carries *named* diarized markers (``Maya:`` …) matching
    detected people, attribute directly to each named speaker — N-speaker capable, so
    panels work. Otherwise fall back to the generic ``Speaker N`` role heuristic.
    """
    out: Dict[str, str] = {}

    named_turns = build_named_turns(transcript, _detected_person_lookup(hosts, guests))
    if named_turns:
        for quote_id, char_start in quote_char_starts.items():
            if char_start is None:
                continue
            name = speaker_for_char(int(char_start), named_turns)
            if name:
                out[quote_id] = person_id(name)
        return out

    turns = build_speaker_turns(transcript)
    if not turns:
        return {}
    cluster_to_name = map_clusters_to_people(turns, hosts=hosts, guests=guests)
    for quote_id, char_start in quote_char_starts.items():
        if char_start is None:
            continue
        cluster = speaker_for_char(int(char_start), turns)
        name = cluster_to_name.get(cluster) if cluster is not None else None
        if name:
            out[quote_id] = person_id(name)
    return out


def add_spoken_by_edges(
    artifact: Dict,
    transcript: str,
    *,
    hosts: Sequence[str],
    guests: Sequence[str],
) -> int:
    """Mutate a gi.json *artifact* in place: add ``Person`` nodes + ``SPOKEN_BY`` edges
    (Quote → Person) for confidently-attributed quotes. Idempotent. Returns the number
    of ``SPOKEN_BY`` edges added.

    This is the derivable enrichment that unblocks the keystone ``Person→Insight`` link
    (#874): once a Quote is ``SPOKEN_BY`` a Person and the Insight is ``SUPPORTED_BY``
    that Quote, ``CorpusGraph._derive_speaker_links`` connects Person→Insight directly.
    """
    nodes = artifact.setdefault("nodes", [])
    edges = artifact.setdefault("edges", [])
    quote_char_starts = {
        n["id"]: (n.get("properties") or {}).get("char_start")
        for n in nodes
        if n.get("type") == "Quote" and isinstance(n.get("id"), str)
    }
    attribution = attribute_quote_speakers(
        transcript, quote_char_starts, hosts=hosts, guests=guests
    )
    existing_persons = {n["id"] for n in nodes if n.get("type") == "Person"}
    existing_spoken = {(e.get("from"), e.get("to")) for e in edges if e.get("type") == "SPOKEN_BY"}
    added = 0
    for quote_id, person in attribution.items():
        if person not in existing_persons:
            nodes.append(
                {
                    "id": person,
                    "type": "Person",
                    "properties": {"name": person.split(":", 1)[-1].replace("-", " ").title()},
                }
            )
            existing_persons.add(person)
        if (quote_id, person) not in existing_spoken:
            edges.append({"type": "SPOKEN_BY", "from": quote_id, "to": person})
            existing_spoken.add((quote_id, person))
            added += 1
    return added
