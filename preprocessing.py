"""Provider-agnostic preprocessing functions.

This module contains preprocessing functions that are shared across
all providers (local transformers, OpenAI, etc.). These functions
clean and normalize transcripts before summarization or other processing.

Functions moved from summarizer.py in Stage 1 of incremental modularization.
"""

import re

# Outro removal patterns
OUTRO_BLOCK_PATTERNS = [
    r"thank you so much for listening.*?(?=\n\n|\Z)",
    r"if you enjoyed this episode.*?(?=\n\n|\Z)",
    r"please (rate|review|subscribe).*?(?=\n\n|\Z)",
    r"you can find (more|the newsletter) at .*?(?=\n\n|\Z)",
    r"lennyspodcast\.com.*?(?=\n\n|\Z)",
]


def clean_transcript(
    text: str,
    remove_timestamps: bool = True,
    normalize_speakers: bool = True,
    collapse_blank_lines: bool = True,
    remove_fillers: bool = False,
) -> str:
    """Clean podcast transcript for better summarization quality.

    Preprocessing is essential for improving summarization quality:
    - Strips timestamps like [00:12:34] (language-agnostic, works for all languages)
    - Removes only generic speaker tags (Speaker 1:, SPEAKER 1:, Host:, etc.)
      Preserves actual speaker names (e.g., "John Doe:", "Jane Smith:")
    - Collapses excessive blank lines
    - Optionally removes filler tokens (English-only, disabled by default
      for multi-language support)

    Note: This function is conservative to preserve speaker names detected via NER
    and to work with transcripts in any language. Only generic English patterns are removed.

    Args:
        text: Raw transcript text
        remove_timestamps: Whether to remove timestamp patterns like [00:12:34]
        normalize_speakers: Whether to remove generic speaker tags (preserves actual names)
        collapse_blank_lines: Whether to collapse multiple blank lines into single line
        remove_fillers: Whether to remove common English filler words (disabled by default)

    Returns:
        Cleaned transcript text
    """
    cleaned = text

    # Remove timestamps like [00:12:34] or [1:23:45]
    # This is language-agnostic (numbers work for all languages)
    if remove_timestamps:
        # Match patterns like [00:12:34], [1:23:45], [12:34]
        timestamp_pattern = r"\[\d{1,2}:\d{2}(?::\d{2})?\]"
        cleaned = re.sub(timestamp_pattern, "", cleaned)

    # Normalize speaker tags - ONLY remove generic patterns, preserve actual names
    # This is conservative to avoid removing real speaker names detected via NER
    if normalize_speakers:
        # Only match generic English patterns that are clearly not real names:
        # - "SPEAKER 1:", "Speaker 1:", "Speaker 2:" (all caps or title case with number)
        # - "Host:", "Guest:", "Interviewer:", "Interviewee:" (generic role labels)
        # - "Person 1:", "Person 2:" (generic person labels)
        #
        # We do NOT remove patterns that look like real names:
        # - "John Doe:", "Jane Smith:" (capitalized words without numbers/roles)
        # - Multi-word capitalized names
        #
        # Pattern explanation:
        # - (^|\n)\s*SPEAKER\s+\d+\s*: matches "SPEAKER 1:" at start of line
        #   (with optional leading whitespace)
        # - (^|\n)\s*Speaker\s+\d+\s*: matches "Speaker 1:" at start of line
        #   (with optional leading whitespace)
        # - (^|\n)\s*(Host|Guest|Interviewer|Interviewee)\s*: matches generic roles at start of line
        # - (^|\n)\s*Person\s+\d+\s*: matches "Person 1:" at start of line
        #
        # Note: We use (^|\n) to match start of string OR start of line (after newline)
        # This handles cases where timestamps were removed, leaving leading spaces
        generic_speaker_patterns = [
            r"(^|\n)\s*SPEAKER\s+\d+\s*:",  # "SPEAKER 1:" (all caps)
            r"(^|\n)\s*Speaker\s+\d+\s*:",  # "Speaker 1:" (title case)
            r"(^|\n)\s*(Host|Guest|Interviewer|Interviewee)\s*:",  # Generic roles
            r"(^|\n)\s*Person\s+\d+\s*:",  # "Person 1:"
        ]
        for pattern in generic_speaker_patterns:
            cleaned = re.sub(pattern, r"\1", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # Collapse excessive blank lines (3+ consecutive newlines -> 2 newlines)
    # This is language-agnostic
    if collapse_blank_lines:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Optionally remove filler words/phrases (English-only, disabled by default)
    # Note: This is language-specific and won't work for non-English transcripts
    # Only enable if you're certain transcripts are English
    if remove_fillers:
        # Common English filler words/phrases (case-insensitive)
        # WARNING: These patterns are English-specific and may not work for other languages
        fillers = [
            r"\buh\b",
            r"\bum\b",
            r"\byou know\b",
            r"\bi mean\b",
            # Note: "like", "well", "so" are too common as legitimate words
            # Only remove them if they appear in specific filler contexts
        ]
        # Only remove fillers that appear at start of sentence or after punctuation
        for filler in fillers:
            # Match filler at start of line or after punctuation/space
            pattern = rf"(?:^|[\s\.\!\?])({filler})(?:\s|$)"
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # Clean up extra whitespace (language-agnostic)
    cleaned = re.sub(r" +", " ", cleaned)  # Multiple spaces -> single space
    cleaned = re.sub(r"\n +", "\n", cleaned)  # Spaces at start of line
    cleaned = cleaned.strip()

    return cleaned


def remove_sponsor_blocks(text: str) -> str:
    """Remove sponsor/advertisement blocks from transcript.

    Args:
        text: Transcript text potentially containing sponsor blocks

    Returns:
        Text with sponsor blocks removed
    """
    lower = text.lower()
    cleaned = text
    for phrase in [
        "this episode is brought to you by",
        "today's episode is sponsored by",
        "today's episode is sponsored by",
        "our sponsors today are",
    ]:
        idx = lower.find(phrase)
        if idx == -1:
            continue
        # Remove, say, up to the next blank line OR up to N chars
        end = cleaned.find("\n\n", idx)
        if end == -1 or end - idx > 2000:
            end = min(idx + 2000, len(cleaned))
        cleaned = cleaned[:idx] + cleaned[end:]
        lower = cleaned.lower()
    return cleaned


def remove_outro_blocks(text: str) -> str:
    """Remove outro/closing blocks from transcript.

    Args:
        text: Transcript text potentially containing outro blocks

    Returns:
        Text with outro blocks removed
    """
    cleaned = text
    for pattern in OUTRO_BLOCK_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def clean_for_summarization(text: str) -> str:
    """High-level cleaner for BOTH:
      - offline .cleaned.txt generation
      - runtime summarization (if you want consistency)

    This function applies a complete cleaning pipeline:
    1. Removes timestamps and normalizes speakers
    2. Removes sponsor blocks
    3. Removes outro blocks

    Args:
        text: Raw transcript text

    Returns:
        Fully cleaned transcript text ready for summarization
    """
    text = clean_transcript(
        text,
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,  # or True if you're sure it's all English
    )
    text = remove_sponsor_blocks(text)
    text = remove_outro_blocks(text)
    return text.strip()
