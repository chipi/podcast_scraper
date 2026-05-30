"""Backward-compatible re-export of speaker detection public API.

NER/heuristic logic lives in speaker_detectors submodules.
This module re-exports the public API for existing imports.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ... import config
from ...speaker_detectors.constants import (
    DEFAULT_CONFIDENCE_SCORE,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SPEAKER_NAMES,
    PATTERN_BASED_CONFIDENCE_SCORE,
)
from ...speaker_detectors.detection import _build_speaker_names_list, detect_speaker_names
from ...speaker_detectors.entities import (
    _extract_entities_from_doc,
    _extract_entities_from_segments,
    _pattern_based_fallback,
    _split_text_on_separators,
    extract_person_entities as _extract_person_entities_impl,
)
from ...speaker_detectors.guests import (
    _has_interview_indicator,
    _has_mentioned_only_indicator,
    _is_likely_actual_guest,
)
from ...speaker_detectors.hosts import detect_hosts_from_feed, detect_hosts_from_transcript_intro
from ...speaker_detectors.ner import (
    _ensure_spacy_sentence_boundaries,
    _load_spacy_model as _load_spacy_model_impl,
    _validate_model_name,
)
from ...speaker_detectors.normalization import (
    _extract_confidence_score,
    _sanitize_person_name,
    _validate_person_entity,
    filter_default_speaker_names,
    is_default_speaker_name,
)
from ...speaker_detectors.patterns import analyze_episode_patterns

logger = logging.getLogger(__name__)


def _load_spacy_model(model_name: str) -> Optional[Any]:
    """Load spaCy model (delegates to speaker_detectors.ner; patchable on this module)."""
    return _load_spacy_model_impl(model_name)


def get_ner_model(cfg: config.Config) -> Optional[Any]:
    """Get the appropriate spaCy NER model based on configuration."""
    if cfg.dry_run:
        return None

    if not cfg.auto_speakers:
        return None

    model_name = cfg.ner_model
    if not model_name:
        if cfg.language == "en":
            model_name = config.DEFAULT_NER_MODEL
        else:
            logger.debug("No default NER model for language '%s', skipping detection", cfg.language)
            return None

    nlp = _load_spacy_model(model_name)
    if nlp is not None:
        logger.debug("Loaded spaCy model: %s", model_name)

    return nlp


def extract_person_entities(text: str, nlp: Any) -> list[tuple[str, float]]:
    """Extract PERSON entities (delegates to speaker_detectors.entities; patchable)."""
    return _extract_person_entities_impl(text, nlp)


__all__ = [
    "analyze_episode_patterns",
    "DEFAULT_CONFIDENCE_SCORE",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SPEAKER_NAMES",
    "PATTERN_BASED_CONFIDENCE_SCORE",
    "detect_hosts_from_feed",
    "detect_hosts_from_transcript_intro",
    "detect_speaker_names",
    "extract_person_entities",
    "filter_default_speaker_names",
    "get_ner_model",
    "is_default_speaker_name",
    "logger",
    "_build_speaker_names_list",
    "_ensure_spacy_sentence_boundaries",
    "_extract_confidence_score",
    "_extract_entities_from_doc",
    "_extract_entities_from_segments",
    "_has_interview_indicator",
    "_has_mentioned_only_indicator",
    "_is_likely_actual_guest",
    "_load_spacy_model",
    "_pattern_based_fallback",
    "_sanitize_person_name",
    "_split_text_on_separators",
    "_validate_model_name",
    "_validate_person_entity",
]
