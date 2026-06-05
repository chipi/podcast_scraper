"""Speaker detection: provider protocol and NER implementation submodules."""

from __future__ import annotations

from .base import SpeakerDetector
from .constants import DEFAULT_SAMPLE_SIZE, DEFAULT_SPEAKER_NAMES
from .detection import detect_speaker_names
from .entities import extract_person_entities
from .factory import create_speaker_detector
from .hosts import detect_hosts_from_feed, detect_hosts_from_transcript_intro
from .ner import get_ner_model
from .normalization import filter_default_speaker_names, is_default_speaker_name
from .patterns import analyze_episode_patterns

__all__ = [
    "SpeakerDetector",
    "analyze_episode_patterns",
    "create_speaker_detector",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SPEAKER_NAMES",
    "detect_hosts_from_feed",
    "detect_hosts_from_transcript_intro",
    "detect_speaker_names",
    "extract_person_entities",
    "filter_default_speaker_names",
    "get_ner_model",
    "is_default_speaker_name",
]
