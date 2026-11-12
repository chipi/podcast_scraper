"""Transcript cleaning modules for podcast_scraper.

Public keeps the cleaning protocol and the pattern cleaner (generic cleaning: credits, garbage
lines, timestamps, speaker labels, outros). The LLM and hybrid cleaners left with ad detection
(E04 IP extraction plan, item 5); an extension that owns them registers them on the cleaning
stage hooks (``stage_hooks.transcript_cleaner``).
"""

from .base import TranscriptCleaningProcessor
from .pattern_based import PatternBasedCleaner

__all__ = [
    "TranscriptCleaningProcessor",
    "PatternBasedCleaner",
]
