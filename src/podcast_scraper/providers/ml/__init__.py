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

# Re-export speaker_detection for convenience
from . import speaker_detection
from .ml_provider import MLProvider

__all__ = ["MLProvider", "speaker_detection"]
