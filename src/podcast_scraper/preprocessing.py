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

# Artifact patterns that leak into BART/LED summaries (Issue #283)
# These are formatting markers, speaker labels, and non-content tokens
# that get copied verbatim by summarization models
SUMMARIZATION_ARTIFACT_PATTERNS = [
    r"TextColor-?\s*",  # Caption/subtitle formatting
    r"\bMUSIC\b",  # Production markers
    r"\bSPEAKER\s*\d+\s*:?",  # Generic speaker labels
    r"\[(?:MUSIC|LAUGHTER|APPLAUSE|CROSSTALK|INAUDIBLE)\]",  # Bracketed annotations
    r"<[^>]+>",  # HTML-like tags
    r"▬+",  # Box drawing characters
    r"escription-?\s*",  # Truncated "description" artifact
    r"N-P-R\.?\s*",  # Spelled-out NPR
    r"\bSOUNDBITE\b",  # Soundbite markers
    r"\bUNIDENTIFIED\s+(PERSON|SPEAKER|VOICE)\s*:?",  # Unidentified speaker
]

# Garbage line patterns - website boilerplate that leaks into transcripts
# BART/LED will surface these as "clean declarative sentences" if not removed
# This is the #1 easy win for transcript sanitation
#
# IMPORTANT: These patterns must be VERY SPECIFIC to avoid false positives.
# Use anchors (^) and non-greedy matches where possible.
# The transcript may be a single long line, so patterns like "back to .* page"
# can match across the entire content!
GARBAGE_LINE_PATTERNS = [
    r"^back to (?:the )?(?:home|main|top|previous) page",  # Specific navigation
    r"^back to top$",  # Standalone "back to top"
    r"^article continues",  # "Article continues below" etc.
    r"^home\s*page$",  # Standalone "home page"
    r"^click here\b",
    r"^sign up\b",
    r"^log in\b",
    r"^subscribe now\b",
    r"^share this\b",
    r"^read more\b",
    r"^learn more$",
    r"^view all$",
    r"^see all$",
    r"^load more$",
    r"^show more$",
    r"^privacy policy$",
    r"^terms of service$",
    r"^cookie policy$",
    r"^all rights reserved$",
    r"^copyright \d{4}",
    r"^©\s*\d{4}",
    r"^skip to content$",
    r"^skip to main$",
    r"^advertisement$",
    r"^sponsored content$",
    r"^related articles$",
    r"^you may also like$",
    r"^recommended for you$",
    r"^continue reading$",
    r"^mail online$",
]

# Inline garbage patterns - website chrome that can appear WITHIN text
# These are safe to remove from anywhere because they're very specific
# and unlikely to appear in legitimate podcast content
INLINE_GARBAGE_PATTERNS = [
    r"Back to Mail Online[^.]*\.",  # "Back to Mail Online home page."
    r"Back to the page you came from[^.]*\.",
    r"Click here to [^.]*\.",
    r"Visit our website at [^.]*\.",
    r"For more information,? visit [^.]*\.",
    r"Subscribe to our newsletter[^.]*\.",
    r"Follow us on [^.]*\.",
    r"Share this (?:article|story|episode)[^.]*\.",
    r"Read the full (?:article|story|transcript) at [^.]*\.",
]

# Credit block patterns - podcast credits that should be removed FIRST
# Credits are grammatically perfect, compact, and named-entity rich
# If not removed early, they become the highest-probability summary target
# when other content is filtered out
#
# IMPORTANT: These patterns must be VERY SPECIFIC to avoid removing content.
# Use ^ anchors and re.match() to ensure patterns only match at line start.
CREDIT_BLOCK_PATTERNS = [
    r"^this episode was produced by\b",
    r"^produced by\b",
    r"^edited by\b",
    r"^fact[- ]?check(?:ed|ing)? by\b",
    r"^music by\b",
    r"^sound design by\b",
    r"^engineering by\b",
    r"^mixed by\b",
    r"^mastered by\b",
    r"^executive producer\b",
    r"^senior producer\b",
    r"^associate producer\b",
    r"^(?:the )?(?:indicator|planet money|npr|wsj) is a production of\b",
    r"^special thanks to\b",
    r"^additional reporting by\b",
    r"^research by\b",
    r"^graphics by\b",
    r"^video by\b",
    r"^(?:our )?theme music (?:is |was )?(?:by|from)\b",
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


def remove_summarization_artifacts(text: str) -> str:
    """Remove artifacts that BART/LED models copy verbatim.

    These include formatting markers, speaker labels, and other
    non-content tokens that leak into summaries. This should be
    called before summarization to prevent garbage in output.

    Args:
        text: Text potentially containing artifacts

    Returns:
        Text with artifacts removed
    """
    cleaned = text
    for pattern in SUMMARIZATION_ARTIFACT_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Clean up any resulting double spaces or orphaned punctuation
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return cleaned.strip()


def strip_garbage_lines(text: str) -> str:
    """Remove website boilerplate and navigation garbage from transcripts.

    BART/LED will surface garbage like "Back to Mail Online home page"
    as clean declarative sentences if not removed. This is the #1 easy win
    for transcript sanitation.

    This function handles TWO types of garbage:
    1. Line-level: Entire lines that are garbage (anchored patterns)
    2. Inline: Garbage phrases embedded within legitimate text

    Args:
        text: Text potentially containing garbage lines

    Returns:
        Text with garbage removed
    """
    # Step 1: Remove inline garbage (embedded within text)
    # These are very specific patterns safe to remove from anywhere
    for pattern in INLINE_GARBAGE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Step 2: Remove garbage lines (anchored to line start)
    lines = text.splitlines()
    kept = []
    for ln in lines:
        stripped_ln = ln.strip()
        if not stripped_ln:
            kept.append(ln)  # Keep blank lines for structure
            continue
        # Check if line matches any garbage pattern (anchored to start)
        if any(re.match(p, stripped_ln, re.IGNORECASE) for p in GARBAGE_LINE_PATTERNS):
            continue
        kept.append(ln)
    return "\n".join(kept)


def strip_credits(text: str) -> str:
    """Remove podcast credits blocks from transcripts.

    Credits are grammatically perfect, compact, and named-entity rich.
    If not removed FIRST (before chunking), they become the highest-probability
    summary target when other content is filtered out.

    This MUST be applied before:
    - chunking
    - MAP
    - REDUCE
    - DISTILL

    Credits should never reach the model at all.

    Args:
        text: Text potentially containing credits

    Returns:
        Text with credit lines removed
    """
    lines = text.splitlines()
    kept = []
    for ln in lines:
        # Strip leading/trailing whitespace for accurate matching
        stripped_ln = ln.strip()
        if not stripped_ln:
            kept.append(ln)  # Keep blank lines for structure
            continue

        # Check if line matches any credit pattern, anchored to start of line
        if any(re.match(p, stripped_ln, re.IGNORECASE) for p in CREDIT_BLOCK_PATTERNS):
            continue
        kept.append(ln)
    return "\n".join(kept)


def clean_for_summarization(text: str) -> str:
    """High-level cleaner for BOTH:
      - offline .cleaned.txt generation
      - runtime summarization (if you want consistency)

    Pipeline:
    1. Strips credits FIRST (they're high-confidence targets if left in)
    2. Strips garbage lines (website boilerplate)
    3. Removes timestamps and normalizes speakers
    4. Removes sponsor blocks
    5. Removes outro blocks
    6. Removes summarization artifacts (Issue #283)

    NOTE: Show framing and anecdote removal are now DISABLED here.
    These were too aggressive and removed signal content.
    Light pruning should only happen AFTER DISTILL, not before MAP.

    Args:
        text: Raw transcript text

    Returns:
        Cleaned transcript text ready for summarization
    """
    # Strip credits FIRST - they're high-confidence summary targets if left in
    # Credits are grammatically perfect, compact, and named-entity rich
    text = strip_credits(text)

    # Strip garbage lines - website boilerplate
    text = strip_garbage_lines(text)

    text = clean_transcript(
        text,
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,  # or True if you're sure it's all English
    )
    text = remove_sponsor_blocks(text)
    text = remove_outro_blocks(text)
    text = remove_summarization_artifacts(text)

    # NOTE: remove_show_framing() and remove_anecdotes() are intentionally
    # NOT called here anymore. They were too aggressive and removed signal.
    # Light pruning should only happen AFTER DISTILL, not before MAP.

    return text.strip()
