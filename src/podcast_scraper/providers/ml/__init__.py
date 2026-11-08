"""Unified ML provider for transcription, speaker detection, and summarization.

This module provides a single MLProvider class that implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using:
- Whisper for transcription
- spaCy for speaker detection
- Transformers for summarization

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared ML libraries.

Note: This module was moved from root-level ml/ to providers/ml/ for better
organization. All provider implementations are now under providers/.
"""

from importlib import import_module
from typing import Any

__all__ = ["MLProvider", "speaker_detection"]


def __getattr__(name: str) -> Any:
    # Loaded on first use, not with the package (E04 S2): `config_constants` imports
    # `providers.ml.model_registry` at load, and an eager re-export here dragged the speaker stage
    # into every `import podcast_scraper`.
    if name == "speaker_detection":
        return import_module(".speaker_detection", __name__)
    if name == "MLProvider":
        return import_module(".ml_provider", __name__).MLProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
