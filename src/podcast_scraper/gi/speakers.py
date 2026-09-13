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

import logging
import re
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from ..graph_id_utils import entity_node_id

logger = logging.getLogger(__name__)

# Alignment guard (#876/#925): a Quote's char_start must index into the transcript
# being attributed. Diarization rewrites the transcript (inline speaker markers
# shift offsets), so a char_start computed against a different transcript would
# silently attribute to the wrong speaker. When the quote carries its text we probe
# the transcript at char_start; a gross mismatch (beyond this slack) is skipped.
_OFFSET_PROBE_LEN = 24
_OFFSET_PROBE_SLACK = 64

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


def speaker_for_char(char_start: int, turns: Sequence[Tuple[int, Optional[str]]]) -> Optional[str]:
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


def build_named_turns(
    transcript: str, known_names: Dict[str, str]
) -> List[Tuple[int, Optional[str]]]:
    """``[(char_offset, canonical_name_or_None)]`` for line-start ``<Name>:`` markers (#875).

    A marker whose label matches a *detected* person (``known_names``) attributes to that person.
    Every OTHER line-start marker yields ``None`` — it is still a turn boundary (#2062).

    That ``None`` is the whole point. Dropping an unrecognised marker does not make it neutral: it
    makes the PREVIOUS speaker's span swallow the turn, because :func:`speaker_for_char` returns the
    last marker at or before the quote. Since guests are detected far less often than hosts, the
    swallowed turn is almost always the guest's and the name stamped on it is almost always the
    host's. A ``None`` boundary closes the span instead, so the quote is attributed to nobody —
    which is what this module promises in its own docstring.
    """
    turns: List[Tuple[int, Optional[str]]] = []
    for m in _NAMED_TURN_RE.finditer(transcript):
        canonical = known_names.get(m.group(1).strip().lower())
        turns.append((m.start(1), canonical or None))
    return turns


def build_unverified_named_turns(transcript: str) -> List[Tuple[int, Optional[str]]]:
    """``[(char_offset, name)]`` for line-start ``<Name>:`` markers, with NO whitelist.

    :func:`build_named_turns` only attributes a marker that matches an already-detected person, and
    the GI pipeline never had a detected-person list to give it — so that path never ran and every
    quote shipped with ``speaker_id: None``. Meanwhile the diarized transcript already carries the
    names ("Kevin Roose: ..."), so the speaker is sitting in the text, unread.

    This reads them directly. Since there is no whitelist to reject prose, the person heuristic does
    that job: a label must look like a person (>= 2 tokens, no publisher/network token), so
    ``Note:`` and ``Bloomberg:`` are ignored. Under-attributing beats attributing wrongly.
    """
    turns: List[Tuple[int, Optional[str]]] = []
    for m in _NAMED_TURN_RE.finditer(transcript):
        label = m.group(1).strip()
        # #2062: a label that is not a person (``SPEAKER_01``, ``Bloomberg``, ``Note``) still ENDS
        # the previous speaker's turn. Skipping it entirely is what let the host's name run over
        # the guest's words on the 63.1% of production episodes whose transcript mixes resolved
        # names with raw ``SPEAKER_NN`` markers.
        turns.append((m.start(1), label if _looks_like_person(label) else None))
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


def _person_node_id(name: str, episode_id: Optional[str]) -> str:
    """Person id for a GI speaker attribution, episode-scoping placeholders (#2059 / advisor H1).

    This path used ``identity.slugify.person_id`` directly, which has no placeholder check — so a
    detected host literally named "Host" minted the GLOBAL ``person:host`` here even after the KG
    layer started scoping it. The two layers would then disagree about who a person is within the
    same episode, which m0007's own docstring calls worse than not migrating at all.

    ``entity_node_id`` produces byte-identical ids to ``person_id`` for real names (verified across
    unicode, initials, apostrophes and hyphens), so this is a no-op except for placeholders.

    It deliberately does NOT apply the bare SINGLE-TOKEN name rule from
    ``identity.bare_name_scope``. That rule runs as ONE PASS over the finished payloads
    (``workflow/metadata_generation`` ~:5030, and the m0007 migration), because it needs the
    episode's whole roster and because there are three mint families — ``entity_node_id``,
    ``person_node_id`` and ``identity.slugify.person_id``. Scoping inside one of them makes that
    family disagree with the other two and pre-empts the pass's healing, which can bind "Sam" to a
    real person's id instead of scoping it.

    I got this wrong once: scoping here fixed a duplicate symptom and broke the cross-layer
    identity invariant (``test_entity_identity_invariants`` caught it on ``O'Brien``,
    ``will.i.am``, ``3Blue1Brown``, ``Speakman``). The duplicate's real cause is that
    ``enrich-edges`` runs AFTER the scoping pass, so its unscoped ids sat beside already-scoped
    ones; the fix is for that CLI to run the same pass, not for this function to freelance.
    """
    return entity_node_id("person", name, episode_id=episode_id)


def attribute_quote_speakers(
    transcript: str,
    quote_char_starts: Dict[str, Optional[int]],
    *,
    hosts: Sequence[str],
    guests: Sequence[str],
    episode_id: Optional[str] = None,
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
    # #2062: `named_turns` now also carries None boundaries, so a transcript whose markers are ALL
    # unrecognised produces a non-empty list of pure boundaries. Testing truthiness of the list
    # would take the named path on the strength of markers that name nobody, and silently skip the
    # generic ``Speaker N`` role heuristic below. Require at least one marker that actually names a
    # person before declaring this a named screenplay.
    if any(name for _, name in named_turns):
        for quote_id, char_start in quote_char_starts.items():
            if char_start is None:
                continue
            name = speaker_for_char(int(char_start), named_turns)
            if name:
                out[quote_id] = _person_node_id(name, episode_id)
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
            out[quote_id] = _person_node_id(name, episode_id)
    return out


def _strip_spoken_by(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Dict]:
    """Remove every ``SPOKEN_BY`` edge and any Person left with nothing else to hold it (#2062).

    Mutates *nodes* / *edges* in place. A Person referenced by any surviving edge stays: this pass
    owns the SPOKEN_BY relation, not the person's existence in the graph.

    Returns ``{person_id: properties}`` for the nodes it removed, so the caller can put back what
    the artifact already knew. Without that, re-attributing the SAME person — the ordinary outcome
    on an episode that was already correct — deleted a node carrying
    ``{"name": "Twiggy", "role": "guest"}`` and recreated it from the slug as
    ``{"name": "Unresolved Twiggy Ep 77"}``. That is the display name the insights panel renders and
    the role #2062 exists to get right, both destroyed by the pass meant to repair them.
    """
    was_spoken = {e.get("to") for e in edges if e.get("type") == "SPOKEN_BY"}
    edges[:] = [e for e in edges if e.get("type") != "SPOKEN_BY"]
    if not was_spoken:
        return {}
    still_referenced = {e.get("from") for e in edges} | {e.get("to") for e in edges}
    orphaned = {pid for pid in was_spoken if pid and pid not in still_referenced}
    if not orphaned:
        return {}
    removed: Dict[str, Dict] = {
        str(n.get("id")): dict(n.get("properties") or {})
        for n in nodes
        if n.get("type") == "Person" and n.get("id") in orphaned
    }
    nodes[:] = [n for n in nodes if not (n.get("type") == "Person" and n.get("id") in orphaned)]
    return removed


def _person_display_name(nodes: List[Dict], person_id: Optional[str]) -> Optional[str]:
    """The display name of a Person node, or ``None``."""
    if not person_id:
        return None
    for n in nodes:
        if n.get("type") == "Person" and n.get("id") == person_id:
            name = (n.get("properties") or {}).get("name")
            return str(name) if isinstance(name, str) and name.strip() else None
    return None


def _rewrite_quote_attribution(nodes: List[Dict], attribution: Dict[str, str]) -> None:
    """Point each Quote's OWN ``speaker_id`` / ``speaker_name`` at the recomputed answer (#2062).

    Rewriting SPOKEN_BY alone does not change what a reader sees. ``server/app_gi_view`` resolves a
    quote's speaker as ``speaker_name`` or the quote's own ``speaker_id`` BEFORE consulting the
    edge, and the pipeline stamps both onto the Quote — 4,898 of 7,343 quotes in a production
    sample carry ``speaker_id`` and 982 carry ``speaker_name``. So a remediation that fixed only
    the edges left the insights panel showing exactly the name the operator complained about, while
    the person->insight surfaces (which DO read edges) moved: two surfaces disagreeing per quote,
    which is worse than one being wrong.

    A quote that can no longer be attributed has both fields cleared. Leaving a stale id there is
    the same lie the edge rewrite removes.
    """
    for node in nodes:
        if node.get("type") != "Quote" or not isinstance(node.get("id"), str):
            continue
        props = node.setdefault("properties", {})
        if not isinstance(props, dict):
            continue
        person = attribution.get(node["id"])
        props["speaker_id"] = person
        props["speaker_name"] = _person_display_name(nodes, person)


def add_spoken_by_edges(
    artifact: Dict,
    transcript: str,
    *,
    hosts: Sequence[str],
    guests: Sequence[str],
    replace: bool = False,
) -> int:
    """Mutate a gi.json *artifact* in place: add ``Person`` nodes + ``SPOKEN_BY`` edges
    (Quote → Person) for confidently-attributed quotes. Idempotent. Returns the number
    of ``SPOKEN_BY`` edges added.

    This is the derivable enrichment that unblocks the keystone ``Person→Insight`` link
    (#874): once a Quote is ``SPOKEN_BY`` a Person and the Insight is ``SUPPORTED_BY``
    that Quote, ``CorpusGraph._derive_speaker_links`` connects Person→Insight directly.

    *replace* (#2062) makes the pass AUTHORITATIVE instead of purely additive. The default is
    additive, which is idempotent in one direction only: an edge that already exists is skipped and
    an edge that should not exist is never removed. So every improvement to attribution is inert on
    an already-processed episode — re-running this over a corpus built by older, wronger code
    reports ``SPOKEN_BY=0`` and changes nothing, which is exactly what a fresh two-episode DGX
    ingest showed on 2026-09-13.

    With *replace* the pass first drops the ``SPOKEN_BY`` edges it owns and any Person node that
    existed ONLY to receive one, then recomputes from the transcript. A Person still referenced by
    another edge is kept — this function owns ``SPOKEN_BY``, not the person. It stays opt-in
    because rewriting historical artifacts without being asked is its own failure mode.
    """
    nodes = artifact.setdefault("nodes", [])
    edges = artifact.setdefault("edges", [])
    stripped_person_props: Dict[str, Dict] = {}
    if replace:
        stripped_person_props = _strip_spoken_by(nodes, edges)
    # The gi.json carries its own episode id; placeholder speaker ids must be scoped to it so
    # this layer and the KG layer agree about who a person is (#2059 / advisor H1).
    episode_id = artifact.get("episode_id")
    episode_id = episode_id if isinstance(episode_id, str) and episode_id else None
    quote_char_starts: Dict[str, Optional[int]] = {}
    misaligned = 0
    for n in nodes:
        if n.get("type") != "Quote" or not isinstance(n.get("id"), str):
            continue
        props = n.get("properties") or {}
        cs = props.get("char_start")
        text = props.get("text")
        # Skip attribution for quotes whose char_start clearly does not index into
        # THIS transcript (offset-space mismatch -> would attribute to the wrong
        # speaker). Only checkable when the quote carries its text; lenient slack so
        # aligned quotes (and small marker shifts) are never falsely dropped.
        if isinstance(cs, int) and isinstance(text, str) and text.strip():
            probe = text.strip()[:_OFFSET_PROBE_LEN]
            window = transcript[max(0, cs) : cs + len(probe) + _OFFSET_PROBE_SLACK]
            if probe and probe not in window:
                misaligned += 1
                continue
        quote_char_starts[n["id"]] = cs
    if misaligned:
        logger.warning(
            "add_spoken_by_edges: %d quote(s) have char_start not aligned with the "
            "transcript; skipping their speaker attribution. Likely a transcript/offset "
            "mismatch -- e.g. enrich-edges run on a re-diarized transcript without "
            "rebuilding GI (diarization shifts char offsets). Re-run GI so quote offsets "
            "match the diarized transcript.",
            misaligned,
        )
    attribution = attribute_quote_speakers(
        transcript, quote_char_starts, hosts=hosts, guests=guests, episode_id=episode_id
    )
    existing_persons = {n["id"] for n in nodes if n.get("type") == "Person"}
    existing_spoken = {(e.get("from"), e.get("to")) for e in edges if e.get("type") == "SPOKEN_BY"}
    added = 0
    for quote_id, person in attribution.items():
        if person not in existing_persons:
            # Prefer what this artifact already knew (#2062). The slug-derived name stays the last
            # resort it always was for a genuinely NEW node; it is not a downgrade applied to a
            # node that was correct a moment ago.
            props = stripped_person_props.get(person) or {
                "name": person.split(":", 1)[-1].replace("-", " ").title()
            }
            nodes.append({"id": person, "type": "Person", "properties": props})
            existing_persons.add(person)
        if (quote_id, person) not in existing_spoken:
            edges.append({"type": "SPOKEN_BY", "from": quote_id, "to": person})
            existing_spoken.add((quote_id, person))
            added += 1
    if replace:
        _rewrite_quote_attribution(nodes, attribution)
    return added
