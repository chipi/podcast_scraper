"""Named Entity Recognition (NER) for automatic speaker name detection from episode metadata."""

from __future__ import annotations

import logging
import re

# Bandit: subprocess is needed for spaCy model download
import subprocess  # nosec B404
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from ... import config

# Note: spacy is imported lazily in _load_spacy_model() to avoid requiring ML dependencies
# at module import time. This allows unit tests to import this module without spacy installed.


logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# Valid spaCy model names contain only alphanumeric, underscore, hyphen, and dot
_VALID_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

# Model validation constants
MAX_MODEL_NAME_LENGTH = 100

# Name validation constants
MIN_NAME_LENGTH = 2
MIN_RAW_NAME_LENGTH = 2
MIN_SEGMENT_LENGTH = 2

# Confidence score constants
DEFAULT_CONFIDENCE_SCORE = 1.0
PATTERN_BASED_CONFIDENCE_SCORE = 0.7
MAX_HEURISTIC_SCORE = 1.0

# Pattern analysis constants
DEFAULT_SAMPLE_SIZE = 5
DESCRIPTION_SNIPPET_LENGTH = 20
CONTEXT_WINDOW_SIZE = 20
PREFIX_WORDS_COUNT = 3
SUFFIX_WORDS_COUNT = 3
MIN_PREFIX_SUFFIX_COUNT = 2
TOP_PREFIXES_SUFFIXES_COUNT = 5

# Position threshold constants
START_POSITION_THRESHOLD = 0.3  # 30% of title length
END_POSITION_THRESHOLD = 0.7  # 70% of title length
POSITION_CONSISTENCY_THRESHOLD = 0.6  # 60% consistency required

# Scoring constants
POSITION_SCORE_BONUS = 0.3
PREFIX_SUFFIX_SCORE_BONUS = 0.2
OVERLAP_SCORE_BONUS = 0.5
COMBINED_SCORE_DIVISOR = 2.0

# Minimum speakers constant
MIN_SPEAKERS_REQUIRED = 2

# Context-aware filtering patterns (Issue #325)
# These patterns help distinguish actual guests from merely mentioned people
INTERVIEW_INDICATOR_PATTERNS = [
    r"interview(?:ed|ing|s)?\s+(?:with\s+)?",  # "interview with X", "interviewing X"
    r"(?:we(?:'re|'ve)?|i(?:'m|'ve)?)\s+(?:been\s+)?joined\s+by\s+",  # "joined by X"
    r"speaking\s+(?:with|to)\s+",  # "speaking with X"
    r"talking\s+(?:with|to)\s+",  # "talking to X"
    r"conversation\s+with\s+",  # "conversation with X"
    r"guest(?:s)?(?:\s*:|\s+is|\s+are)?\s*",  # "guest: X", "guest is X"
    r"featuring\s+",  # "featuring X"
    r"(?:special\s+)?guest\s+",  # "special guest X"
    r"welcomes?\s+",  # "welcomes X"
    r"sits?\s+down\s+with\s+",  # "sits down with X"
    r"chats?\s+with\s+",  # "chats with X"
]

MENTIONED_ONLY_PATTERNS = [
    r"about\s+",  # "about John Smith"
    r"on\s+\w+(?:'s)?\s+",  # "on Smith's policies"
    r"discuss(?:es|ing|ed)?\s+",  # "discusses John Smith"
    r"analysis\s+of\s+",  # "analysis of Smith's..."
    r"according\s+to\s+",  # "according to Smith"
    r"(?:he|she|they)\s+says?\s+",  # "he says..."
    r"'s\s+(?:\w+\s+)*(?:policy|plan|speech|decision|statement)",  # "Smith's policy"
    r"(?:the\s+)?(?:president|ceo|senator|governor)\s+",  # "CEO Jane Doe" (title prefix)
    r"covers?\s+",  # "covers the story"
    r"examines?\s+",  # "examines the issue"
    r"looks?\s+at\s+",  # "looks at the data"
    r"(?:news|story|report)\s+(?:about|on)\s+",  # "news about the company"
]


def _has_interview_indicator(name: str, text: str) -> bool:
    """Check if name appears in an interview context.

    Args:
        name: Person name to check
        text: Text to search in (title or description)

    Returns:
        True if name appears after an interview indicator pattern
    """
    text_lower = text.lower()
    name_lower = name.lower()

    for pattern in INTERVIEW_INDICATOR_PATTERNS:
        # Check if pattern appears before the name
        full_pattern = pattern + r".*?" + re.escape(name_lower)
        if re.search(full_pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _has_mentioned_only_indicator(name: str, text: str) -> bool:
    """Check if name appears in a mentioned-only context (not an actual guest).

    Args:
        name: Person name to check
        text: Text to search in (title or description)

    Returns:
        True if name appears after a mentioned-only indicator pattern
    """
    text_lower = text.lower()
    name_lower = name.lower()

    for pattern in MENTIONED_ONLY_PATTERNS:
        # Check if pattern appears before the name
        full_pattern = pattern + r".*?" + re.escape(name_lower)
        if re.search(full_pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _is_likely_actual_guest(name: str, title: str, description: str | None) -> bool:
    """Determine if a detected person is likely an actual guest vs merely mentioned.

    Uses context-aware filtering to reduce false positives from spaCy NER.

    Args:
        name: Person name detected by NER
        title: Episode title
        description: Episode description (may be None)

    Returns:
        True if the person is likely an actual guest, False if likely just mentioned
    """
    combined_text = title
    if description:
        combined_text += " " + description

    # Check for interview indicators (strong signal for actual guest)
    has_interview = _has_interview_indicator(name, combined_text)

    # Check for mentioned-only indicators (strong signal for NOT a guest)
    has_mentioned_only = _has_mentioned_only_indicator(name, combined_text)

    # Decision logic (relaxed):
    # - If interview indicator found: likely actual guest
    # - If mentioned-only indicator found: still include
    #   (relaxed - may be guest mentioned in context)
    # - Default: include the name (conservative - don't filter without strong evidence)
    if has_interview:
        logger.debug("Name '%s' has interview indicator - likely actual guest", name)
        return True
    if has_mentioned_only:
        # Relaxed: Don't filter out based on mentioned-only indicator alone
        # The name might still be a guest mentioned in the description
        logger.debug(
            "Name '%s' has mentioned-only indicator - keeping anyway (relaxed filter)", name
        )
        return True

    # Default: include the name (conservative - don't filter without evidence)
    return True


def _validate_model_name(model_name: str) -> bool:
    """Validate spaCy model name to prevent command injection.

    Args:
        model_name: Model name to validate

    Returns:
        True if valid, False otherwise
    """
    if not model_name or len(model_name) > MAX_MODEL_NAME_LENGTH:
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
    # Lazy import: Only import spacy when this function is called
    # This allows the module to be imported without ML dependencies installed
    import spacy  # noqa: F401

    # Validate model name to prevent command injection
    if not _validate_model_name(model_name):
        logger.error("Invalid spaCy model name: %s (contains invalid characters)", model_name)
        return None

    try:
        # Load only NER component to reduce memory usage
        # Disable parser, tagger, and lemmatizer since we only need NER
        # This can reduce memory usage by 30-50% for most models
        try:
            nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
            logger.debug("Loaded spaCy model (NER only): %s", model_name)
        except (ValueError, KeyError):
            # Some models may not support disabling components, fall back to full load
            logger.debug(
                "Model %s doesn't support component disabling, loading full pipeline", model_name
            )
            nlp = spacy.load(model_name)
            logger.debug("Loaded spaCy model (full pipeline): %s", model_name)
        return nlp
    except OSError:
        logger.debug("spaCy model '%s' not found locally, attempting to download...", model_name)
        try:
            # Use subprocess to call 'python -m spacy download' (most reliable method)
            # This ensures we use the same Python interpreter and environment
            # Model name is validated above to prevent command injection
            subprocess.run(  # nosec B603
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug("Successfully downloaded spaCy model: %s", model_name)
            # Now try loading again with disabled components for memory efficiency
            try:
                nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
                logger.debug("Loaded spaCy model (NER only) after download: %s", model_name)
            except (ValueError, KeyError):
                # Fall back to full pipeline if component disabling not supported
                nlp = spacy.load(model_name)
                logger.debug("Loaded spaCy model (full pipeline) after download: %s", model_name)
            return nlp
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to download spaCy model '%s': %s. Output: %s",
                model_name,
                exc,
                exc.stderr or exc.stdout or "",
            )
            logger.info("You can manually install with: python -m spacy download %s", model_name)
            return None
        except OSError as exc:
            logger.error(
                "Failed to load spaCy model '%s' after download attempt: %s", model_name, exc
            )
            return None


def get_ner_model(cfg: config.Config) -> Optional[Any]:
    """Get the appropriate spaCy NER model based on configuration.

    This function loads the spaCy model directly without caching.
    Providers should load the model once during initialization and pass it
    to detect_speaker_names() to avoid redundant loads.

    Args:
        cfg: Configuration object

    Returns:
        Loaded spaCy nlp object or None if model unavailable
    """
    # Skip model loading in dry-run mode
    if cfg.dry_run:
        return None

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

    # Load model directly (no caching)
    nlp = _load_spacy_model(model_name)
    if nlp is not None:
        logger.debug("Loaded spaCy model: %s", model_name)

    return nlp


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

    # Validate: must have at least one letter and be at least MIN_NAME_LENGTH characters
    if not name or len(name) < MIN_NAME_LENGTH:
        return None

    # Must contain at least one letter (not just numbers or punctuation)
    if not re.search(r"[a-zA-Z]", name):
        return None

    return name


def _validate_person_entity(raw_name: str) -> bool:
    """Validate that a raw entity name is likely a person.

    Args:
        raw_name: Raw entity name from NER

    Returns:
        True if valid person entity, False otherwise
    """
    if not raw_name or len(raw_name) < MIN_RAW_NAME_LENGTH:
        return False
    # Filter out pure numbers and HTML-like patterns
    if re.match(r"^\d+$", raw_name) or re.search(r"[<>]", raw_name):
        return False
    return True


def _extract_confidence_score(ent: Any) -> float:
    """Extract confidence score from spaCy entity.

    Args:
        ent: spaCy entity object

    Returns:
        Confidence score (defaults to DEFAULT_CONFIDENCE_SCORE if not available)
    """
    if hasattr(ent, "score") and ent.score is not None:
        return float(ent.score)
    elif hasattr(ent, "_") and hasattr(ent._, "score") and ent._.score is not None:
        return float(ent._.score)
    return DEFAULT_CONFIDENCE_SCORE


def _extract_entities_from_doc(
    doc: Any, seen_raw_names: Set[str], seen_sanitized_names: Set[str]
) -> List[Tuple[str, float]]:
    """Extract PERSON entities from a spaCy document.

    Args:
        doc: spaCy document object
        seen_raw_names: Set of already seen raw names (for deduplication)
        seen_sanitized_names: Set of already seen sanitized names (for deduplication)

    Returns:
        List of (sanitized_name, confidence_score) tuples
    """
    persons = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue

        raw_name = ent.text.strip()

        # Skip if already seen
        if raw_name in seen_raw_names:
            continue

        # Validate entity
        if not _validate_person_entity(raw_name):
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

        # Get confidence score
        confidence = _extract_confidence_score(ent)
        persons.append((sanitized_name, confidence))

    return persons


def _split_text_on_separators(text: str) -> Tuple[List[str], Optional[str]]:
    """Split text on common separators used in episode titles.

    Args:
        text: Text to split

    Returns:
        Tuple of (segments_list, last_segment)
    """
    separators = ["|", "—", "–", " - "]
    segments = [text]
    last_segment = None

    # Split on first separator found
    for sep in separators:
        if sep in text:
            segments = [s.strip() for s in text.split(sep)]
            last_segment = segments[-1] if segments else None
            break

    return segments, last_segment


def _extract_entities_from_segments(
    segments: List[str],
    nlp: Any,
    seen_raw_names: Set[str],
    seen_sanitized_names: Set[str],
) -> List[Tuple[str, float]]:
    """Extract PERSON entities from text segments, prioritizing last segment.

    Args:
        segments: List of text segments
        nlp: spaCy NLP model
        seen_raw_names: Set of already seen raw names
        seen_sanitized_names: Set of already seen sanitized names

    Returns:
        List of (sanitized_name, confidence_score) tuples
    """
    persons = []
    # Process segments in reverse order to prioritize last segment
    for segment in reversed(segments):
        if not segment or len(segment) < MIN_SEGMENT_LENGTH:
            continue

        segment_doc = nlp(segment)
        segment_persons = _extract_entities_from_doc(
            segment_doc, seen_raw_names, seen_sanitized_names
        )
        persons.extend(segment_persons)

        # If we found entities in a segment, stop checking other segments
        # This ensures we get the guest name from the last segment
        if persons:
            break

    return persons


def _pattern_based_fallback(
    last_segment: str, seen_sanitized_names: Set[str]
) -> Optional[Tuple[str, float]]:
    """Pattern-based fallback for name extraction when NER fails.

    Args:
        last_segment: Last segment of text (often contains guest name)
        seen_sanitized_names: Set of already seen sanitized names

    Returns:
        Tuple of (sanitized_name, confidence_score) or None
    """
    if not last_segment:
        return None

    # Pattern: 2-3 words, each starting with capital letter
    # Examples: "Dylan Field", "Mary Jane Watson", "John Smith"
    name_pattern = r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$"
    if not re.match(name_pattern, last_segment):
        return None

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
    if any(phrase in last_segment_lower for phrase in common_phrases):
        return None

    # Sanitize and add as candidate
    sanitized_name = _sanitize_person_name(last_segment)
    if not sanitized_name:
        return None

    sanitized_lower = sanitized_name.lower()
    if sanitized_lower in seen_sanitized_names:
        return None

    # Lower confidence since it's pattern-based, not NER
    logger.debug(
        "Pattern-based fallback: extracted '%s' from last segment '%s'",
        sanitized_name,
        last_segment,
    )
    return (sanitized_name, PATTERN_BASED_CONFIDENCE_SCORE)


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
        seen_raw_names: Set[str] = set()  # Track raw names to avoid duplicates
        seen_sanitized_names: Set[str] = set()  # Track sanitized names for deduplication

        # First, try NER on the full text
        doc = nlp(text)
        persons = _extract_entities_from_doc(doc, seen_raw_names, seen_sanitized_names)

        # Fallback: if no entities found, try splitting on common separators
        if not persons:
            segments, last_segment = _split_text_on_separators(text)
            persons = _extract_entities_from_segments(
                segments, nlp, seen_raw_names, seen_sanitized_names
            )

            # Pattern-based fallback: if still no entities
            if not persons and last_segment:
                pattern_result = _pattern_based_fallback(last_segment, seen_sanitized_names)
                if pattern_result:
                    sanitized_name, confidence = pattern_result
                    seen_sanitized_names.add(sanitized_name.lower())
                    persons.append((sanitized_name, confidence))

        return persons
    except Exception as exc:
        logger.debug("Error extracting PERSON entities: %s", exc)
        return []


def detect_hosts_from_transcript_intro(
    transcript_text: str,
    nlp: Optional[Any] = None,
    intro_duration_seconds: int = 120,
    words_per_second: float = 2.5,
) -> Set[str]:
    """Detect host names from transcript intro patterns (first 60-120 seconds).

    Scans the first portion of transcript for common intro patterns like:
    - "I'm [Name]"
    - "This is [Show Name]... I'm [Name]"
    - "Welcome to [Show]... I'm [Name]"

    Args:
        transcript_text: Full transcript text
        nlp: spaCy NLP model (optional, only needed if NER is used)
        intro_duration_seconds: How many seconds of transcript to scan (default: 120)
        words_per_second: Average words per second for estimating intro length (default: 2.5)

    Returns:
        Set of detected host names from intro patterns
    """
    if not transcript_text or not nlp:
        return set()

    # Estimate intro length: ~120 seconds * 2.5 words/sec = ~300 words
    # Take first N words to scan for intro patterns
    intro_word_count = int(intro_duration_seconds * words_per_second)
    words = transcript_text.split()[:intro_word_count]
    intro_text = " ".join(words)

    # Common intro patterns
    intro_patterns = [
        r"I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "I'm John" or "I'm John Smith"
        # "This is The Indicator... I'm John"
        r"This is\s+[^.]+\s+I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        # "Welcome to Planet Money... I'm John"
        r"Welcome to\s+[^.]+\s+I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]

    detected_names = set()
    for pattern in intro_patterns:
        matches = re.finditer(pattern, intro_text, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            # Filter out common false positives
            if name and len(name) > 2 and name.lower() not in ["the", "this", "that"]:
                detected_names.add(name)

    # Also use NER on intro text to find person names
    if nlp:
        intro_persons = extract_person_entities(intro_text, nlp)
        for name, _ in intro_persons:
            detected_names.add(name)

    return detected_names


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
    # However, these may be organization names (e.g., "NPR") rather than person names
    # We'll collect them but treat them as "publisher" metadata, not necessarily hosts
    if feed_authors:
        for author in feed_authors:
            if author and author.strip():
                # Author tags may contain email format "Name <email>", extract just the name
                author_clean = author.strip()
                # Remove email part if present (format: "Name <email@example.com>")
                if "<" in author_clean and ">" in author_clean:
                    author_clean = author_clean.split("<")[0].strip()
                if author_clean:
                    # Check if this looks like an organization name (all caps, short, no spaces)
                    # Common patterns: "NPR", "BBC", "CNN", etc.
                    is_likely_org = (
                        len(author_clean) <= 10
                        and author_clean.isupper()
                        and " " not in author_clean
                    )
                    if is_likely_org:
                        logger.debug(
                            "RSS author '%s' appears to be an organization name, "
                            "treating as publisher metadata rather than host",
                            author_clean,
                        )
                        # Don't add to hosts - these are publishers, not actual hosts
                    else:
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


def _analyze_title_position(guest_name: str, title: str) -> Optional[str]:
    """Analyze where a guest name appears in the title.

    Args:
        guest_name: Guest name to find
        title: Episode title

    Returns:
        Position preference: "start", "end", "middle", or None
    """
    guest_lower = guest_name.lower()
    title_lower = title.lower()

    if guest_lower not in title_lower:
        return None

    idx = title_lower.find(guest_lower)
    title_len = len(title)

    # Determine position: start (first START_POSITION_THRESHOLD%),
    # end (last END_POSITION_THRESHOLD%), or middle
    if idx < title_len * START_POSITION_THRESHOLD:
        return "start"
    elif idx > title_len * END_POSITION_THRESHOLD:
        return "end"
    else:
        return "middle"


def _extract_prefix_suffix(guest_name: str, title: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract prefix and suffix context around guest name in title.

    Args:
        guest_name: Guest name to find context for
        title: Episode title

    Returns:
        Tuple of (prefix_text, suffix_text)
    """
    guest_lower = guest_name.lower()
    title_lower = title.lower()

    if guest_lower not in title_lower:
        return None, None

    idx = title_lower.find(guest_lower)
    prefix_text = None
    suffix_text = None

    # Extract prefix
    if idx > 0:
        prefix_start = max(0, idx - CONTEXT_WINDOW_SIZE)
        prefix_raw = title[prefix_start:idx].strip().lower()
        # Extract last few words as prefix
        prefix_words = prefix_raw.split()[-PREFIX_WORDS_COUNT:]
        if prefix_words:
            prefix_text = " ".join(prefix_words)

    # Extract suffix
    if idx + len(guest_name) < len(title):
        suffix_end = min(len(title), idx + len(guest_name) + CONTEXT_WINDOW_SIZE)
        suffix_raw = title[idx + len(guest_name) : suffix_end].strip().lower()
        # Extract first few words as suffix
        suffix_words = suffix_raw.split()[:SUFFIX_WORDS_COUNT]
        if suffix_words:
            suffix_text = " ".join(suffix_words)

    return prefix_text, suffix_text


def _find_common_patterns(
    patterns: List[str], min_count: int = MIN_PREFIX_SUFFIX_COUNT
) -> List[str]:
    """Find common patterns that appear at least min_count times.

    Args:
        patterns: List of pattern strings
        min_count: Minimum count threshold

    Returns:
        List of common patterns
    """
    if not patterns:
        return []

    from collections import Counter

    pattern_counts = Counter(patterns)
    return [p for p, count in pattern_counts.items() if count >= min_count]


def _determine_title_position_preference(title_positions: List[str]) -> Optional[str]:
    """Determine the most common title position preference.

    Args:
        title_positions: List of position strings ("start", "end", "middle")

    Returns:
        Most common position if consistent enough, None otherwise
    """
    if not title_positions:
        return None

    from collections import Counter

    position_counts = Counter(title_positions)
    most_common = position_counts.most_common(1)[0]
    if most_common[1] >= len(title_positions) * POSITION_CONSISTENCY_THRESHOLD:
        return most_common[0]
    return None


def analyze_episode_patterns(
    episodes: List[Any],
    nlp: Any,
    cached_hosts: Set[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Analyze patterns from sample episodes to extract heuristics for guest selection.

    Analyzes episode titles and first DESCRIPTION_SNIPPET_LENGTH characters of descriptions.

    Args:
        episodes: List of Episode objects to analyze
        nlp: spaCy NLP model
        cached_hosts: Set of detected host names to filter out
        sample_size: Number of episodes to sample (default DEFAULT_SAMPLE_SIZE)

    Returns:
        Dictionary with heuristics:
        - title_position_preference: "start", "end", or None
        - common_prefixes: List of common prefixes before guest names
        - common_suffixes: List of common suffixes after guest names
    """
    if not episodes or not nlp:
        return {}

    sample_episodes = episodes[:sample_size]
    title_positions = []  # Track where guest names appear in titles
    prefixes = []
    suffixes = []

    for episode in sample_episodes:
        title = episode.title
        # Extract persons from title
        title_persons = extract_person_entities(title, nlp)

        # Filter out hosts from title persons
        title_guests = [name for name, _ in title_persons if name not in cached_hosts]

        if not title_guests:
            continue

        # Analyze each guest name
        for guest_name in title_guests:
            # Analyze position
            position = _analyze_title_position(guest_name, title)
            if position:
                title_positions.append(position)

            # Extract prefix/suffix
            prefix, suffix = _extract_prefix_suffix(guest_name, title)
            if prefix:
                prefixes.append(prefix)
            if suffix:
                suffixes.append(suffix)

    # Determine most common title position
    title_position_preference = _determine_title_position_preference(title_positions)

    # Find common prefixes/suffixes
    common_prefixes = _find_common_patterns(prefixes)[:TOP_PREFIXES_SUFFIXES_COUNT]
    common_suffixes = _find_common_patterns(suffixes)[:TOP_PREFIXES_SUFFIXES_COUNT]

    heuristics = {
        "title_position_preference": title_position_preference,
        "common_prefixes": common_prefixes,
        "common_suffixes": common_suffixes,
    }

    logger.debug(
        "Extracted heuristics from %d sample episodes: %s", len(sample_episodes), heuristics
    )
    return heuristics


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
    """
    Detect speaker names from episode title and description using NER.

    IMPORTANT: Host names should be detected separately using detect_hosts_from_feed()
    and passed via cached_hosts or known_hosts. This function ONLY extracts guests
    from episode title and first DESCRIPTION_SNIPPET_LENGTH characters of description.

    Args:
        episode_title: Episode title (required for guest detection)
        episode_description: Episode description (only first
            DESCRIPTION_SNIPPET_LENGTH chars used for guest detection)
        nlp: Pre-loaded spaCy model (required). Providers should load the model
            once during initialization and pass it here to avoid redundant loads.
        cfg: Configuration object (optional, used for validation)
        known_hosts: Manually specified host names (optional)
        cached_hosts: Previously detected hosts to reuse (optional)
        heuristics: Pattern-based heuristics from sample episodes (optional)

    Returns:
        Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        - detection_succeeded: True if real names were detected, False if defaults were used
        Note: detected_hosts_set will be empty as hosts are not detected here
    """
    if cfg and not cfg.auto_speakers:
        logger.debug("Auto-speakers disabled, detection failed")
        return DEFAULT_SPEAKER_NAMES.copy(), set(), False  # type: ignore[return-value]

    if not nlp:
        logger.debug("spaCy model not available, detection failed")
        return DEFAULT_SPEAKER_NAMES.copy(), set(), False  # type: ignore[return-value]

    # Use cached/known hosts, but do NOT detect hosts from episode metadata
    # Priority: known_hosts > cached_hosts
    hosts: Set[str] = set()
    if known_hosts:
        hosts.update(known_hosts)
    elif cached_hosts:
        hosts.update(cached_hosts)

    # Fallback: If no hosts detected from RSS metadata, try transcript intro
    # This is a cheap fallback that scans first 60-90 seconds for intro patterns
    if not hosts and transcript_text and nlp:
        transcript_hosts = detect_hosts_from_transcript_intro(
            transcript_text, nlp, intro_duration_seconds=90, words_per_second=2.5
        )
        if transcript_hosts:
            hosts.update(transcript_hosts)
            logger.info(
                "  → Detected hosts from transcript intro (fallback): %s",
                ", ".join(sorted(transcript_hosts)),
            )

    # Note: We intentionally do NOT detect hosts from episode title/description
    # Hosts should only come from feed-level metadata, known_hosts config, or transcript fallback

    # Extract PERSON entities from episode title and first
    # DESCRIPTION_SNIPPET_LENGTH chars of description
    # These are guests, not hosts
    # Extract with confidence scores
    title_persons_with_scores = extract_person_entities(episode_title, nlp)

    # Limit description to first DESCRIPTION_SNIPPET_LENGTH characters for guest detection
    description_snippet = None
    if episode_description:
        description_snippet = episode_description[:DESCRIPTION_SNIPPET_LENGTH].strip()
        if not description_snippet:
            description_snippet = None

    description_persons_with_scores = (
        extract_person_entities(description_snippet, nlp) if description_snippet else []
    )

    # Filter out hosts from both sources (keep scores)
    # Use case-insensitive matching to catch variations like "NPR" vs "npr"
    # Also normalize whitespace for better matching
    hosts_normalized = {h.lower().strip() for h in hosts}
    title_guests_with_scores = [
        (name, score)
        for name, score in title_persons_with_scores
        if name.lower().strip() not in hosts_normalized
    ]
    description_guests_with_scores = [
        (name, score)
        for name, score in description_persons_with_scores
        if name.lower().strip() not in hosts_normalized
    ]

    # Apply context-aware filtering to reduce false positives (Issue #325)
    # Filter out people who are merely mentioned but not actual guests
    title_guests_with_scores = [
        (name, score)
        for name, score in title_guests_with_scores
        if _is_likely_actual_guest(name, episode_title, episode_description)
    ]
    description_guests_with_scores = [
        (name, score)
        for name, score in description_guests_with_scores
        if _is_likely_actual_guest(name, episode_title, episode_description)
    ]

    # Build guest candidates with confidence scores and heuristics
    guest_candidates = _build_guest_candidates(
        title_guests_with_scores, description_guests_with_scores, episode_title, heuristics
    )

    # Select best guest based on combined scores
    selected_guest, selected_confidence, selected_has_overlap, selected_heuristic_score = (
        _select_best_guest(guest_candidates)
    )

    # Collect all guest names for logging
    all_guest_names = list(guest_candidates.keys())

    # Log guest detection results
    _log_guest_detection(
        selected_guest,
        selected_confidence,
        selected_has_overlap,
        selected_heuristic_score,
        all_guest_names,
        title_persons_with_scores,
        description_persons_with_scores,
    )

    # Build final speaker names list
    guests = [selected_guest] if selected_guest else []
    screenplay_num_speakers = cfg.screenplay_num_speakers if cfg else 2
    speaker_names, detection_succeeded, _used_defaults = _build_speaker_names_list(
        hosts, guests, screenplay_num_speakers
    )

    # Return hosts set (hosts are passed in, not detected here)
    # Note: used_defaults is extracted but not returned (would require API change)
    # TODO: Add used_defaults to return tuple when updating callers
    return speaker_names, hosts, detection_succeeded  # type: ignore[return-value]


def _calculate_heuristic_score(
    name: str, title: str, heuristics: Optional[Dict[str, Any]]
) -> float:
    """Calculate heuristic score based on position patterns.

    Args:
        name: Guest name to score
        title: Episode title
        heuristics: Pattern-based heuristics dictionary

    Returns:
        Heuristic score (0.0 to MAX_HEURISTIC_SCORE)
    """
    if not heuristics:
        return 0.0  # type: ignore[return-value]

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
        if title_pos_pref == "start" and idx < title_len * START_POSITION_THRESHOLD:
            score += POSITION_SCORE_BONUS
        elif title_pos_pref == "end" and idx > title_len * END_POSITION_THRESHOLD:
            score += POSITION_SCORE_BONUS
        elif (
            title_pos_pref == "middle"
            and START_POSITION_THRESHOLD <= (idx / title_len) <= END_POSITION_THRESHOLD
        ):
            score += POSITION_SCORE_BONUS

    # Prefix/suffix pattern matching
    common_prefixes = heuristics.get("common_prefixes", [])
    common_suffixes = heuristics.get("common_suffixes", [])

    if idx > 0:
        prefix_start = max(0, idx - CONTEXT_WINDOW_SIZE)
        prefix_text = title[prefix_start:idx].strip().lower()
        for prefix in common_prefixes:
            if prefix in prefix_text:
                score += PREFIX_SUFFIX_SCORE_BONUS
                break

    if idx + len(name) < len(title):
        suffix_end = min(len(title), idx + len(name) + CONTEXT_WINDOW_SIZE)
        suffix_text = title[idx + len(name) : suffix_end].strip().lower()
        for suffix in common_suffixes:
            if suffix in suffix_text:
                score += PREFIX_SUFFIX_SCORE_BONUS
                break

    return min(score, MAX_HEURISTIC_SCORE)


def _build_guest_candidates(
    title_guests_with_scores: List[Tuple[str, float]],
    description_guests_with_scores: List[Tuple[str, float]],
    episode_title: str,
    heuristics: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[float, bool, float]]:
    """Build guest candidates dictionary with confidence scores and heuristics.

    Args:
        title_guests_with_scores: List of (name, score) from title
        description_guests_with_scores: List of (name, score) from description
        episode_title: Episode title for heuristic scoring
        heuristics: Pattern-based heuristics

    Returns:
        Dictionary mapping name -> (confidence, appears_in_both, heuristic_score)
    """
    guest_candidates: Dict[str, Tuple[float, bool, float]] = {}

    # Process title guests
    for name, score in title_guests_with_scores:
        heuristic_score = _calculate_heuristic_score(name, episode_title, heuristics)
        guest_candidates[name] = (score, False, heuristic_score)

    # Process description guests (check for overlap)
    for name, score in description_guests_with_scores:
        if name in guest_candidates:
            # Overlap found: boost confidence by averaging and mark as appearing in both
            title_score, _, heuristic_score = guest_candidates[name]
            combined_score = (title_score + score) / COMBINED_SCORE_DIVISOR
            guest_candidates[name] = (combined_score, True, heuristic_score)
        else:
            guest_candidates[name] = (score, False, 0.0)

    return guest_candidates


def _select_best_guest(
    guest_candidates: Dict[str, Tuple[float, bool, float]],
) -> Tuple[Optional[str], float, bool, float]:
    """Select the guest with the highest combined score.

    Args:
        guest_candidates: Dictionary mapping name -> (confidence, appears_in_both, heuristic_score)

    Returns:
        Tuple of (selected_guest, confidence, has_overlap, heuristic_score)
    """
    selected_guest = None
    selected_confidence = 0.0
    selected_has_overlap = False
    selected_heuristic_score = 0.0

    for name, (confidence, has_overlap, heuristic_score) in guest_candidates.items():
        # Combined score: overlap bonus + confidence + heuristics
        combined_score = (
            confidence + (OVERLAP_SCORE_BONUS if has_overlap else 0.0) + heuristic_score
        )

        if not selected_guest:
            selected_guest = name
            selected_confidence = confidence
            selected_has_overlap = has_overlap
            selected_heuristic_score = heuristic_score
        else:
            # Calculate current selected guest's combined score
            current_combined = (
                selected_confidence
                + (OVERLAP_SCORE_BONUS if selected_has_overlap else 0.0)
                + selected_heuristic_score
            )

            if combined_score > current_combined:
                selected_guest = name
                selected_confidence = confidence
                selected_has_overlap = has_overlap
                selected_heuristic_score = heuristic_score

    return selected_guest, selected_confidence, selected_has_overlap, selected_heuristic_score


def _log_guest_detection(
    selected_guest: Optional[str],
    selected_confidence: float,
    selected_has_overlap: bool,
    selected_heuristic_score: float,
    all_guest_names: List[str],
    title_persons_with_scores: List[Tuple[str, float]],
    description_persons_with_scores: List[Tuple[str, float]],
) -> None:
    """Log guest detection results.

    Args:
        selected_guest: Selected guest name (if any)
        selected_confidence: Confidence score of selected guest
        selected_has_overlap: Whether guest appears in both title and description
        selected_heuristic_score: Heuristic score of selected guest
        all_guest_names: All candidate guest names
        title_persons_with_scores: Persons found in title
        description_persons_with_scores: Persons found in description
    """
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
        # Always log the selected guest at INFO level
        logger.info("  → Guest: %s", selected_guest)
    elif title_persons_with_scores or description_persons_with_scores:
        # All persons were hosts (filtered out)
        logger.info("  → Guest: (none - all detected persons are hosts)")
    else:
        logger.info("  → Guest: (none detected in episode title/description)")


def _build_speaker_names_list(
    hosts: Set[str],
    guests: List[str],
    max_names: int,
) -> Tuple[List[str], bool, bool]:
    """Build final speaker names list from hosts and guests.

    Args:
        hosts: Set of host names
        guests: List of guest names
        max_names: Maximum number of names to return

    Returns:
        Tuple of (speaker_names_list, detection_succeeded, used_defaults)
        - detection_succeeded: True if real names were detected, False if defaults were used
        - used_defaults: True if defaults were added to reach min speakers (quality flag)
    """
    # Detection succeeded if we have real names (hosts or guests), not defaults
    # Note: Having hosts but no guests is still a success (host-only episodes are valid)
    detection_succeeded = bool(hosts or guests)
    used_defaults = False  # Quality flag: track if defaults were used

    if not guests and hosts:
        # Hosts detected but no guests - use actual host names
        # Convert set to sorted list for deterministic ordering
        host_list = sorted(list(hosts))[:max_names]
        speaker_names = host_list
        logger.debug("  → Using detected host names: %s (no guests detected)", speaker_names)
    elif not hosts and not guests:
        # No hosts AND no guests detected - this is "unknown host" not a failure
        # Allow "unknown host" without flagging as detection failure
        # This is common when RSS metadata has organization names (publishers) not person names
        logger.info("  → No hosts or guests detected (using defaults)")
        speaker_names = DEFAULT_SPEAKER_NAMES.copy()
        detection_succeeded = False  # Still mark as failed for backward compatibility
        used_defaults = True  # Quality flag: defaults were used
    else:
        # Combine hosts and guests, prioritizing hosts
        speaker_names = list(hosts)[:max_names] + guests[: max_names - len(hosts)]
        if len(speaker_names) < MIN_SPEAKERS_REQUIRED:
            # Ensure at least MIN_SPEAKERS_REQUIRED speakers for screenplay formatting
            # Only add defaults if we have at least one real speaker (host or guest)
            # This prevents adding defaults when we have hosts but no guests (host-only episodes)
            if hosts or guests:
                used_defaults = True  # Quality flag: defaults were used
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
                # No real speakers - use defaults (this case should be rare)
                speaker_names = DEFAULT_SPEAKER_NAMES.copy()
                used_defaults = True

    return speaker_names[:max_names], detection_succeeded, used_defaults


__all__ = [
    "detect_speaker_names",
    "detect_hosts_from_feed",
    "get_ner_model",
    "extract_person_entities",
    "DEFAULT_SPEAKER_NAMES",
]
