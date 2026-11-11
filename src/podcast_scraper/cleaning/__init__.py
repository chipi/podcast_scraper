"""Transcript cleaning modules for podcast_scraper.

The LLM and hybrid cleaners load on first use: they leave public with ad detection (E04 IP
extraction plan, item 5), and importing this package must not need them.
"""

from typing import Any, TYPE_CHECKING

from .base import TranscriptCleaningProcessor
from .pattern_based import PatternBasedCleaner

if TYPE_CHECKING:  # names for the type checker only; at run time they load on first use
    from .hybrid import HybridCleaner
    from .llm_based import LLMBasedCleaner

_LAZY = {
    "HybridCleaner": ".hybrid",
    "LLMBasedCleaner": ".llm_based",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


__all__ = [
    "TranscriptCleaningProcessor",
    "PatternBasedCleaner",
    "LLMBasedCleaner",
    "HybridCleaner",
]
