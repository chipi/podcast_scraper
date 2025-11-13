"""Named Entity Recognition (NER) for automatic speaker name detection from episode metadata."""

from __future__ import annotations

import logging
import re
import subprocess  # nosec B404 - subprocess is needed for spaCy model download
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy

from . import config

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# Valid spaCy model names contain only alphanumeric, underscore, hyphen, and dot
_VALID_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _validate_model_name(model_name: str) -> bool:
    """Validate spaCy model name to prevent command injection.

    Args:
        model_name: Model name to validate

    Returns:
        True if valid, False otherwise
    """
    if not model_name or len(model_name) > 100:  # Reasonable length limit
        return False
    return bool(_VALID_MODEL_NAME_PATTERN.match(model_name))


def _load_spacy_model(model_name: str) -> Optional[Any]:
    """Load spaCy model, automatically downloading if missing.

    Similar to Whisper's automatic model download, this function will attempt
    to download the model if it's not found locally.

    Args:
        model_name: Name of the spaCy model to load (e.g., 'en_core_web_sm')

    Returns:
        Loaded spaCy nlp object or None if download/load fails
    """
    # Validate model name to prevent command injection
    if not _validate_model_name(model_name):
        logger.error("Invalid spaCy model name: %s (contains invalid characters)", model_name)
        return None

    try:
        nlp = spacy.load(model_name)
        logger.debug("Loaded spaCy model: %s", model_name)
        return nlp
    except OSError:
        logger.info("spaCy model '%s' not found locally, attempting to download...", model_name)
        try:
            # Use subprocess to call 'python -m spacy download' (most reliable method)
            # This ensures we use the same Python interpreter and environment
            # Model name is validated above to prevent command injection
            subprocess.run(  # nosec B603 - model_name is validated above
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Successfully downloaded spaCy model: %s", model_name)
            # Now try loading again
            nlp = spacy.load(model_name)
            logger.debug("Loaded spaCy model after download: %s", model_name)
            return nlp
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to download spaCy model '%s': %s. Output: %s",
                model_name,
                exc,
                exc.stderr or exc.stdout or "",
            )
            logger.warning("You can manually install with: python -m spacy download %s", model_name)
            return None
        except OSError as exc:
            logger.error(
                "Failed to load spaCy model '%s' after download attempt: %s", model_name, exc
            )
            return None


def get_ner_model(cfg: config.Config) -> Optional[Any]:
    """Get the appropriate spaCy NER model based on configuration."""
    if not cfg.auto_speakers:
        return None

    model_name = cfg.ner_model
    if not model_name:
        # Derive default model from language
        if cfg.language == "en":
            model_name = config.DEFAULT_NER_MODEL
        else:
            # For other languages, try to construct model name (e.g., "fr" -> "fr_core_news_sm")
            # This is a simple heuristic; users can override with --ner-model
            logger.debug("No default NER model for language '%s', skipping detection", cfg.language)
            return None

    return _load_spacy_model(model_name)


def _sanitize_person_name(name: str) -> Optional[str]:
    """Sanitize a person name by removing non-letter characters and normalizing.

    Removes:
    - Parentheses and their contents: "John (Smith)" -> "John"
    - Trailing punctuation: "John," -> "John"
    - Leading/trailing whitespace

    Keeps:
    - Letters, spaces, hyphens (for names like "Mary-Jane")
    - Apostrophes (for names like "O'Brien")

    Args:
        name: Raw person name from NER

    Returns:
        Sanitized name, or None if name becomes invalid after sanitization
    """
    if not name:
        return None

    # Remove parentheses and their contents: "John (Smith)" -> "John"
    name = re.sub(r"\([^)]*\)", "", name)

    # Remove trailing punctuation (commas, periods, semicolons, etc.)
    name = re.sub(r"[,.;:!?]+$", "", name)

    # Remove leading punctuation
    name = re.sub(r"^[,.;:!?]+", "", name)

    # Strip whitespace
    name = name.strip()

    # Remove any remaining non-letter characters except spaces, hyphens, and apostrophes
    # This handles cases like "John, Smith" -> "John Smith"
    name = re.sub(r"[^\w\s\-\']+", "", name)

    # Normalize whitespace (multiple spaces -> single space)
    name = re.sub(r"\s+", " ", name).strip()

    # Validate: must have at least one letter and be at least 2 characters
    if not name or len(name) < 2:
        return None

    # Must contain at least one letter (not just numbers or punctuation)
    if not re.search(r"[a-zA-Z]", name):
        return None

    return name


def extract_person_entities(text: str, nlp: Any) -> List[Tuple[str, float]]:
    """Extract PERSON entities from text using spaCy NER with confidence scores.

    Sanitizes names to remove non-letter characters (parentheses, commas, etc.)
    and deduplicates to return unique person candidates.

    Uses a fallback strategy: if no entities found in full text, splits on
    common separators (|, —, –) and tries NER on each segment, prioritizing
    the last segment (often contains guest names).

    Args:
        text: Text to extract entities from (should already be cleaned of HTML)
        nlp: spaCy NLP model

    Returns:
        List of (sanitized_name, confidence_score) tuples, deduplicated.
        Confidence is 1.0 if not available.
    """
    if not text or not nlp:
        return []

    try:
        # First, try NER on the full text
        doc = nlp(text)
        persons = []
        seen_raw_names = set()  # Track raw names to avoid duplicates
        seen_sanitized_names = set()  # Track sanitized names for deduplication

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                raw_name = ent.text.strip()

                # Skip if already seen (raw name deduplication)
                if raw_name in seen_raw_names:
                    continue

                # Filter out obvious non-person entities (very short names, pure numbers, etc.)
                if not raw_name or len(raw_name) < 2:
                    continue

                # Filter out pure numbers and HTML-like patterns
                if re.match(r"^\d+$", raw_name) or re.search(r"[<>]", raw_name):
                    continue

                # Sanitize the name
                sanitized_name = _sanitize_person_name(raw_name)
                if not sanitized_name:
                    continue

                # Deduplicate based on sanitized name (case-insensitive)
                sanitized_lower = sanitized_name.lower()
                if sanitized_lower in seen_sanitized_names:
                    continue

                # Track both raw and sanitized names
                seen_raw_names.add(raw_name)
                seen_sanitized_names.add(sanitized_lower)

                # Try to get confidence score if available (spaCy transformer models)
                confidence = 1.0
                if hasattr(ent, "score"):
                    confidence = float(ent.score)
                elif hasattr(ent._, "score"):
                    confidence = float(ent._.score)

                persons.append((sanitized_name, confidence))

        # Fallback: if no entities found, try splitting on common separators
        # This handles cases like "Title | Guest Name" where NER fails on full text
        if not persons:
            # Common separators used in episode titles: pipe, em dash, en dash
            separators = ["|", "—", "–", " - "]
            segments = [text]
            last_segment = None

            # Split on first separator found
            for sep in separators:
                if sep in text:
                    segments = [s.strip() for s in text.split(sep)]
                    last_segment = segments[-1] if segments else None
                    break

            # Try NER on each segment, prioritizing the last one (often contains guest name)
            # Process segments in reverse order to prioritize last segment
            for segment in reversed(segments):
                if not segment or len(segment) < 2:
                    continue

                segment_doc = nlp(segment)
                for ent in segment_doc.ents:
                    if ent.label_ == "PERSON":
                        raw_name = ent.text.strip()

                        # Skip if already seen
                        if raw_name in seen_raw_names:
                            continue

                        # Filter out obvious non-person entities
                        if not raw_name or len(raw_name) < 2:
                            continue

                        if re.match(r"^\d+$", raw_name) or re.search(r"[<>]", raw_name):
                            continue

                        # Sanitize the name
                        sanitized_name = _sanitize_person_name(raw_name)
                        if not sanitized_name:
                            continue

                        # Deduplicate
                        sanitized_lower = sanitized_name.lower()
                        if sanitized_lower in seen_sanitized_names:
                            continue

                        seen_raw_names.add(raw_name)
                        seen_sanitized_names.add(sanitized_lower)

                        # Get confidence score
                        confidence = 1.0
                        if hasattr(ent, "score"):
                            confidence = float(ent.score)
                        elif hasattr(ent._, "score"):
                            confidence = float(ent._.score)

                        persons.append((sanitized_name, confidence))

                        # If we found entities in a segment, prioritize it
                        # (don't check earlier segments)
                        # This ensures we get the guest name from the last segment
                        if persons:
                            break

                # If we found entities, stop checking other segments
                if persons:
                    break

            # Pattern-based fallback: if still no entities and we have a last segment,
            # check if it looks like a person name (2-3 words, properly capitalized)
            if not persons and last_segment:
                # Pattern: 2-3 words, each starting with capital letter
                # Examples: "Dylan Field", "Mary Jane Watson", "John Smith"
                name_pattern = r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$"
                if re.match(name_pattern, last_segment):
                    # Check if it's not a common non-name phrase
                    common_phrases = {
                        "guest",
                        "host",
                        "episode",
                        "title",
                        "interview",
                        "conversation",
                    }
                    last_segment_lower = last_segment.lower()
                    if not any(phrase in last_segment_lower for phrase in common_phrases):
                        # Sanitize and add as candidate
                        sanitized_name = _sanitize_person_name(last_segment)
                        if sanitized_name:
                            sanitized_lower = sanitized_name.lower()
                            if sanitized_lower not in seen_sanitized_names:
                                seen_sanitized_names.add(sanitized_lower)
                                # Lower confidence since it's pattern-based, not NER
                                persons.append((sanitized_name, 0.7))
                                logger.debug(
                                    "Pattern-based fallback: extracted '%s' from last segment '%s'",
                                    sanitized_name,
                                    last_segment,
                                )

        return persons
    except Exception as exc:
        logger.debug("Error extracting PERSON entities: %s", exc)
        return []


def detect_hosts_from_feed(
    feed_title: Optional[str],
    feed_description: Optional[str],
    feed_authors: Optional[List[str]] = None,
    nlp: Optional[Any] = None,
) -> Set[str]:
    """Detect host names from feed-level metadata.

    Hosts are recurring speakers and should only be extracted from feed-level
    metadata, not from individual episodes.

    Priority:
    1. RSS author tags (<author>, <dc:creator>, <itunes:author>) - most reliable
    2. NER extraction from feed title/description - fallback if no authors

    Args:
        feed_title: Feed title
        feed_description: Feed description (optional)
        feed_authors: List of author names from RSS feed (optional, preferred source)
        nlp: spaCy NLP model (optional, only needed if no authors provided)

    Returns:
        Set of detected host names
    """
    hosts: Set[str] = set()

    # Priority 1: Use RSS author tags if available (most reliable)
    # RSS feeds typically have one channel-level author, plus optional iTunes author/owner
    # These should all refer to the same host(s), so we collect them all
    if feed_authors:
        for author in feed_authors:
            if author and author.strip():
                # Author tags may contain email format "Name <email>", extract just the name
                author_clean = author.strip()
                # Remove email part if present (format: "Name <email@example.com>")
                if "<" in author_clean and ">" in author_clean:
                    author_clean = author_clean.split("<")[0].strip()
                if author_clean:
                    hosts.add(author_clean)
        if hosts:
            logger.debug(
                "Detected hosts from RSS author tags (author/itunes:author/itunes:owner): %s",
                list(hosts),
            )
            return hosts

    # Priority 2: Fall back to NER extraction from feed title/description
    if nlp:
        if feed_title:
            title_persons = extract_person_entities(feed_title, nlp)
            hosts.update(name for name, _ in title_persons)
        if feed_description:
            desc_persons = extract_person_entities(feed_description, nlp)
            hosts.update(name for name, _ in desc_persons)
        if hosts:
            logger.debug("Detected hosts via NER from feed metadata: %s", list(hosts))

    return hosts


def analyze_episode_patterns(
    episodes: List[Any],
    nlp: Any,
    cached_hosts: Set[str],
    sample_size: int = 5,
) -> Dict[str, Any]:
    """Analyze patterns from sample episodes to extract heuristics for guest selection.

    Analyzes episode titles and first 20 characters of descriptions.

    Args:
        episodes: List of Episode objects to analyze
        nlp: spaCy NLP model
        cached_hosts: Set of detected host names to filter out
        sample_size: Number of episodes to sample (default 5)

    Returns:
        Dictionary with heuristics:
        - title_position_preference: "start", "end", or None
        - common_prefixes: List of common prefixes before guest names
        - common_suffixes: List of common suffixes after guest names
    """
    from .rss_parser import extract_episode_description

    if not episodes or not nlp:
        return {}

    sample_episodes = episodes[:sample_size]
    title_positions = []  # Track where guest names appear in titles
    prefixes = []
    suffixes = []

    for episode in sample_episodes:
        title = episode.title
        description = (
            extract_episode_description(episode.item) if hasattr(episode, "item") else None
        )

        # Limit description to first 20 characters
        description_snippet = description[:20].strip() if description else None

        # Extract persons from title and description snippet
        title_persons = extract_person_entities(title, nlp)
        description_persons = (
            extract_person_entities(description_snippet, nlp) if description_snippet else []
        )

        # Filter out hosts from title persons
        title_guests = [name for name, _ in title_persons if name not in cached_hosts]

        if not title_guests:
            continue

        # Find position of guest name in title
        for guest_name in title_guests:
            guest_lower = guest_name.lower()
            title_lower = title.lower()

            if guest_lower in title_lower:
                idx = title_lower.find(guest_lower)
                title_len = len(title)

                # Determine position: start (first 30%), end (last 30%), or middle
                if idx < title_len * 0.3:
                    title_positions.append("start")
                elif idx > title_len * 0.7:
                    title_positions.append("end")
                else:
                    title_positions.append("middle")

                # Extract context around guest name (prefixes/suffixes)
                if idx > 0:
                    prefix_start = max(0, idx - 20)
                    prefix_text = title[prefix_start:idx].strip().lower()
                    # Extract last few words as prefix
                    prefix_words = prefix_text.split()[-3:]
                    if prefix_words:
                        prefixes.append(" ".join(prefix_words))

                if idx + len(guest_name) < len(title):
                    suffix_end = min(len(title), idx + len(guest_name) + 20)
                    suffix_text = title[idx + len(guest_name) : suffix_end].strip().lower()
                    # Extract first few words as suffix
                    suffix_words = suffix_text.split()[:3]
                    if suffix_words:
                        suffixes.append(" ".join(suffix_words))

    # Determine most common title position
    title_position_preference = None
    if title_positions:
        from collections import Counter

        position_counts = Counter(title_positions)
        most_common = position_counts.most_common(1)[0]
        if most_common[1] >= len(title_positions) * 0.6:  # At least 60% consistency
            title_position_preference = most_common[0]

    # Find common prefixes/suffixes (appear in at least 2 episodes)
    common_prefixes = []
    common_suffixes = []
    if prefixes:
        from collections import Counter

        prefix_counts = Counter(prefixes)
        common_prefixes = [p for p, count in prefix_counts.items() if count >= 2]

    if suffixes:
        from collections import Counter

        suffix_counts = Counter(suffixes)
        common_suffixes = [s for s, count in suffix_counts.items() if count >= 2]

    heuristics = {
        "title_position_preference": title_position_preference,
        "common_prefixes": common_prefixes[:5],  # Top 5
        "common_suffixes": common_suffixes[:5],  # Top 5
    }

    logger.debug(
        "Extracted heuristics from %d sample episodes: %s", len(sample_episodes), heuristics
    )
    return heuristics


def detect_speaker_names(
    episode_title: str,
    episode_description: Optional[str],
    cfg: Optional[config.Config] = None,
    known_hosts: Optional[Set[str]] = None,
    cached_hosts: Optional[Set[str]] = None,
    heuristics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Set[str], bool]:
    """
    Detect speaker names from episode title and description using NER.

    IMPORTANT: Host names should be detected separately using detect_hosts_from_feed()
    and passed via cached_hosts or known_hosts. This function ONLY extracts guests
    from episode title and first 20 characters of description.

    Args:
        episode_title: Episode title (required for guest detection)
        episode_description: Episode description (only first 20 chars used for guest detection)
        cfg: Configuration object
        known_hosts: Manually specified host names (optional)
        cached_hosts: Previously detected hosts to reuse (optional)
        heuristics: Pattern-based heuristics from sample episodes (optional)

    Returns:
        Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        - detection_succeeded: True if real names were detected, False if defaults were used
        Note: detected_hosts_set will be empty as hosts are not detected here
    """
    if not cfg or not cfg.auto_speakers:
        logger.debug("Auto-speakers disabled, detection failed")
        return DEFAULT_SPEAKER_NAMES.copy(), set(), False

    nlp = get_ner_model(cfg)
    if not nlp:
        logger.warning("spaCy model not available, detection failed")
        return DEFAULT_SPEAKER_NAMES.copy(), set(), False

    # Use cached/known hosts, but do NOT detect hosts from episode metadata
    hosts: Set[str] = set()
    if known_hosts:
        hosts.update(known_hosts)
    elif cached_hosts:
        hosts.update(cached_hosts)
    # Note: We intentionally do NOT detect hosts from episode title/description
    # Hosts should only come from feed-level metadata

    # Extract PERSON entities from episode title and first 20 chars of description
    # These are guests, not hosts
    # Extract with confidence scores
    title_persons_with_scores = extract_person_entities(episode_title, nlp)

    # Limit description to first 20 characters for guest detection
    description_snippet = None
    if episode_description:
        description_snippet = episode_description[:20].strip()
        if not description_snippet:
            description_snippet = None

    description_persons_with_scores = (
        extract_person_entities(description_snippet, nlp) if description_snippet else []
    )

    # Filter out hosts from both sources (keep scores)
    title_guests_with_scores = [
        (name, score) for name, score in title_persons_with_scores if name not in hosts
    ]
    description_guests_with_scores = [
        (name, score) for name, score in description_persons_with_scores if name not in hosts
    ]

    # Build guest candidates with combined confidence (overlap bonus + heuristics)
    # Names appearing in both title and description get higher priority
    # Heuristics add position-based scoring
    guest_candidates = {}  # name -> (confidence, appears_in_both, heuristic_score)

    # Calculate heuristic scores based on position patterns
    def calculate_heuristic_score(
        name: str, title: str, heuristics: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate heuristic score based on position patterns."""
        if not heuristics:
            return 0.0

        score = 0.0
        name_lower = name.lower()
        title_lower = title.lower()

        if name_lower not in title_lower:
            return score

        idx = title_lower.find(name_lower)
        title_len = len(title)

        # Position-based scoring
        title_pos_pref = heuristics.get("title_position_preference")
        if title_pos_pref:
            if title_pos_pref == "start" and idx < title_len * 0.3:
                score += 0.3
            elif title_pos_pref == "end" and idx > title_len * 0.7:
                score += 0.3
            elif title_pos_pref == "middle" and 0.3 <= (idx / title_len) <= 0.7:
                score += 0.3

        # Prefix/suffix pattern matching
        common_prefixes = heuristics.get("common_prefixes", [])
        common_suffixes = heuristics.get("common_suffixes", [])

        if idx > 0:
            prefix_start = max(0, idx - 20)
            prefix_text = title[prefix_start:idx].strip().lower()
            for prefix in common_prefixes:
                if prefix in prefix_text:
                    score += 0.2
                    break

        if idx + len(name) < len(title):
            suffix_end = min(len(title), idx + len(name) + 20)
            suffix_text = title[idx + len(name) : suffix_end].strip().lower()
            for suffix in common_suffixes:
                if suffix in suffix_text:
                    score += 0.2
                    break

        return min(score, 1.0)  # Cap at 1.0

    # Process title guests
    for name, score in title_guests_with_scores:
        heuristic_score = calculate_heuristic_score(name, episode_title, heuristics)
        guest_candidates[name] = (score, False, heuristic_score)

    # Process description guests (check for overlap)
    for name, score in description_guests_with_scores:
        if name in guest_candidates:
            # Overlap found: boost confidence by averaging and mark as appearing in both
            title_score, _, heuristic_score = guest_candidates[name]
            combined_score = (title_score + score) / 2.0
            guest_candidates[name] = (combined_score, True, heuristic_score)
        else:
            guest_candidates[name] = (score, False, 0.0)

    # Select guest with highest combined score (overlap + confidence + heuristics)
    selected_guest = None
    selected_confidence = 0.0
    selected_has_overlap = False
    selected_heuristic_score = 0.0

    for name, (confidence, has_overlap, heuristic_score) in guest_candidates.items():
        # Combined score: overlap bonus + confidence + heuristics
        combined_score = confidence + (0.5 if has_overlap else 0.0) + heuristic_score

        if not selected_guest:
            selected_guest = name
            selected_confidence = confidence
            selected_has_overlap = has_overlap
            selected_heuristic_score = heuristic_score
        else:
            # Calculate current selected guest's combined score
            current_combined = (
                selected_confidence
                + (0.5 if selected_has_overlap else 0.0)
                + selected_heuristic_score
            )

            if combined_score > current_combined:
                selected_guest = name
                selected_confidence = confidence
                selected_has_overlap = has_overlap
                selected_heuristic_score = heuristic_score

    # Log selection details
    all_guest_names = list(guest_candidates.keys())
    if selected_guest:
        if len(all_guest_names) > 1:
            overlap_info = " (overlap)" if selected_has_overlap else ""
            heuristic_info = (
                f" (heuristic: +{selected_heuristic_score:.2f})"
                if selected_heuristic_score > 0
                else ""
            )
            logger.debug(
                "  → Selected '%s' (confidence: %.3f%s%s) from %d candidates: %s",
                selected_guest,
                selected_confidence,
                overlap_info,
                heuristic_info,
                len(all_guest_names),
                ", ".join(all_guest_names),
            )

    # Use single selected guest (or empty list if none)
    guests = [selected_guest] if selected_guest else []

    # Log guests clearly (always log, even if empty)
    if guests:
        # Always log the selected guest at INFO level
        logger.info("  → Guest: %s", selected_guest)
        # Selection details already logged above at DEBUG level
    elif title_persons_with_scores or description_persons_with_scores:
        # All persons were hosts (filtered out)
        logger.info("  → Guest: (none - all detected persons are hosts)")
    else:
        logger.info("  → Guest: (none detected in episode title/description)")

    # Cap total names at configured limit
    max_names = config.DEFAULT_MAX_DETECTED_NAMES
    all_names = list(hosts) + guests
    if len(all_names) > max_names:
        # Prioritize hosts, then guests
        capped_names = list(hosts)[:max_names] + guests[: max_names - len(hosts)]
        logger.debug(
            "Capped detected names from %d to %d (max=%d)",
            len(all_names),
            len(capped_names),
            max_names,
        )
        all_names = capped_names

    # Build speaker names list: hosts + guests
    # Detection succeeded if we have real names (hosts or guests), not defaults
    detection_succeeded = bool(hosts or guests)

    if not guests and hosts:
        # Hosts detected but no guests - use host-only labels
        speaker_names = [
            f"Host {i+1}" if i > 0 else "Host" for i in range(min(len(hosts), max_names))
        ]
        logger.debug("  → Using host-only labels: %s", speaker_names)
    elif not all_names:
        # No hosts AND no guests detected - detection failed
        logger.info("  → Detection failed: no hosts or guests found")
        speaker_names = DEFAULT_SPEAKER_NAMES.copy()
        detection_succeeded = False
    else:
        # Combine hosts and guests, prioritizing hosts
        speaker_names = list(hosts)[:max_names] + guests[: max_names - len(hosts)]
        if len(speaker_names) < 2:
            # Ensure at least 2 speakers for screenplay formatting
            logger.debug(
                "  → Only %d speaker(s) detected, extending with defaults to ensure 2+ speakers",
                len(speaker_names),
            )
            speaker_names.extend(DEFAULT_SPEAKER_NAMES[len(speaker_names) :])

    return speaker_names[:max_names], hosts, detection_succeeded


__all__ = [
    "detect_speaker_names",
    "detect_hosts_from_feed",
    "get_ner_model",
    "extract_person_entities",
    "DEFAULT_SPEAKER_NAMES",
]
