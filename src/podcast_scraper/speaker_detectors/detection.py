"""Speaker name detection orchestration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config
from .constants import (
    DEFAULT_SPEAKER_NAMES,
    DESCRIPTION_SNIPPET_LENGTH,
    INTRO_SNIPPET_LENGTH,
    MIN_SPEAKERS_REQUIRED,
)
from .guests import _is_likely_actual_guest, is_introduced_guest
from .hosts import detect_hosts_from_transcript_intro

logger = logging.getLogger(__name__)


def _extract_person_entities(text: str, nlp: Any) -> list[tuple[str, float]]:
    """Resolve extract_person_entities via public wrapper when loaded (patchable in tests)."""
    try:
        from podcast_scraper.providers.ml import speaker_detection

        return speaker_detection.extract_person_entities(text, nlp)
    except ImportError:
        from .entities import extract_person_entities as direct

        return direct(text, nlp)


def detect_speaker_names(
    episode_title: str,
    episode_description: Optional[str],
    nlp: Any,
    cfg: Optional[config.Config] = None,
    known_hosts: Optional[Set[str]] = None,
    cached_hosts: Optional[Set[str]] = None,
    heuristics: Optional[Dict[str, Any]] = None,
    transcript_text: Optional[str] = None,
) -> Tuple[List[str], Set[str], bool, bool]:
    """Detect guest names from episode title and description using NER."""
    _ = heuristics

    if cfg and not cfg.auto_speakers:
        return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

    if not nlp:
        return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

    hosts: Set[str] = set()
    if known_hosts:
        hosts.update(known_hosts)
    elif cached_hosts:
        hosts.update(cached_hosts)

    if not hosts and transcript_text and nlp:
        transcript_hosts = detect_hosts_from_transcript_intro(transcript_text, nlp)
        if transcript_hosts:
            hosts.update(transcript_hosts)
            logger.info("  → Hosts from transcript intro: %s", sorted(transcript_hosts))

    title_persons = _extract_person_entities(episode_title, nlp)

    description_snippet = None
    if episode_description:
        description_snippet = episode_description[:DESCRIPTION_SNIPPET_LENGTH].strip() or None

    desc_persons = _extract_person_entities(description_snippet, nlp) if description_snippet else []

    guests = _guests_from_candidates(
        title_persons + desc_persons, hosts, episode_title, episode_description
    )

    # The transcript intro is another "description" — the opening minutes name the guests the feed
    # metadata omits ("joining me today is …"). Same NER, but an ASR-grade strict filter
    # (First-Last + an adjacent interview cue) so noisy mentions don't become phantom guests.
    intro_snippet = transcript_text[:INTRO_SNIPPET_LENGTH].strip() if transcript_text else None
    if intro_snippet:
        intro_persons = _extract_person_entities(intro_snippet, nlp)
        guests = _merge_intro_guests(guests, intro_persons, hosts, intro_snippet)

    if guests:
        logger.info("  → Guest: %s", ", ".join(guests))

    max_names = cfg.screenplay_num_speakers if cfg else 2
    speaker_names, detection_succeeded, used_defaults = _build_speaker_names_list(
        hosts, guests, max_names
    )
    return speaker_names, hosts, detection_succeeded, used_defaults


def _guests_from_candidates(
    candidates: List[Tuple[str, float]],
    hosts: Set[str],
    episode_title: str,
    guest_context: Optional[str],
) -> List[str]:
    """Filter NER person candidates (title + description + intro) to actual guests.

    Drops names that are hosts or already seen, and keeps only those with an interview intent
    (``_is_likely_actual_guest``) in ``guest_context`` (description + transcript intro). Order kept.
    """
    hosts_lower = {h.lower().strip() for h in hosts}
    seen: Set[str] = set()
    guests: List[str] = []
    for name, _score in candidates:
        key = name.lower().strip()
        if key in hosts_lower or key in seen:
            continue
        if not _is_likely_actual_guest(name, episode_title, guest_context):
            continue
        seen.add(key)
        guests.append(name)
    return guests


def _merge_intro_guests(
    existing: List[str],
    intro_persons: List[Tuple[str, float]],
    hosts: Set[str],
    intro_snippet: str,
) -> List[str]:
    """Add transcript-intro persons that the intro *introduces* as guests (strict ASR filter).

    Skips hosts + already-found guests; only keeps a name with a First-Last shape and an adjacent
    interview cue (``is_introduced_guest``). Order preserved (existing guests first).
    """
    blocked = {h.lower().strip() for h in hosts} | {g.lower().strip() for g in existing}
    out = list(existing)
    for name, _score in intro_persons:
        key = name.lower().strip()
        if key in blocked:
            continue
        if not is_introduced_guest(name, intro_snippet):
            continue
        blocked.add(key)
        out.append(name)
    return out


def _build_speaker_names_list(
    hosts: Set[str],
    guests: List[str],
    max_names: int,
) -> Tuple[List[str], bool, bool]:
    """Build the final speaker names list from hosts and guests.

    ``max_names`` IS NOT APPLIED HERE ANY MORE, and that is the fix for #2095/#2078.

    It is the screenplay's seat count (``cfg.screenplay_num_speakers``, default 2) and it was
    truncating the list of people the episode STATES, which is a different thing entirely. On a
    show with two feed-stated hosts the arithmetic was ``hosts[:2] + guests[:2 - 2]`` — an empty
    guest slice — so the guest reached neither ``detected_guests`` nor ``metadata_named``. Because
    the record builder reads only those, the person vanished from ``content.speakers`` altogether,
    not even carried as ``placed: false``.

    Verified before the change: ``Mackenzie Price`` appears in NO record file in the whole control
    corpus, on a Hard Fork episode whose own transcript says *"So let's bring in Alpha School
    cofounder, Mackenzie Price"* followed by *"Thanks for having me"*. Same shape for
    ``Alexander Stubb``, ``Robert Malley`` / ``Mark Williams`` and ``Glenn Fogel``.

    The seat cap still exists — it is applied where seats actually exist, at the screenplay
    formatter — and the roster's arithmetic ("one spare name, one spare voice") keeps its own
    cap. What must NOT be capped is the list that answers "who does this episode say is here?",
    because a stated name may enter the record unplaced and may be bound by introduction,
    self-intro or sign-off evidence. It is only the arithmetic that needs a bounded list (#876).
    """
    detection_succeeded = bool(hosts or guests)
    used_defaults = False

    if not guests and hosts:
        speaker_names = sorted(hosts)
        logger.debug("  → Using detected host names: %s (no guests detected)", speaker_names)
    elif not hosts and not guests:
        logger.info("  → No hosts or guests detected (using defaults)")
        speaker_names = DEFAULT_SPEAKER_NAMES.copy()
        detection_succeeded = False
        used_defaults = True
    else:
        speaker_names = list(hosts) + guests
        if len(speaker_names) < MIN_SPEAKERS_REQUIRED:
            if hosts or guests:
                used_defaults = True
                logger.debug(
                    (
                        "  → Only %d speaker(s) detected, extending with defaults "
                        "to ensure %d+ speakers"
                    ),
                    len(speaker_names),
                    MIN_SPEAKERS_REQUIRED,
                )
                speaker_names.extend(DEFAULT_SPEAKER_NAMES[len(speaker_names) :])
            else:
                speaker_names = DEFAULT_SPEAKER_NAMES.copy()
                used_defaults = True

    _ = max_names  # kept in the signature: the seat cap now belongs to the screenplay formatter
    return speaker_names, detection_succeeded, used_defaults
