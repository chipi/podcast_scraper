"""Response-shape guardrails for any inference service (ADR-099).

Catches the "200 OK with semantically corrupted content" failure mode that
the connection-level :mod:`..resilience` layer cannot detect. Each per-
service check raises :class:`GuardrailViolation` on failure; consumers
that already catch :data:`..resilience.TimeoutLike` add a sibling
``except GuardrailViolation`` block — the failure goes through the same
fallback path as any other inference-side error.

Used to be embedded in ``providers/tailnet_dgx/resilience.py`` — extracted
here in 2026-06-15 so the guardrails are deployment-agnostic from day one
and any provider (self-hosted or cloud) imports from a single place.

Sub-modules:

- :mod:`.exceptions` — :class:`GuardrailViolation`
- :mod:`.chat` — :func:`check_chat_response` for any chat-completion-shaped API
- :mod:`.transcription` — :func:`check_whisper_response` for any whisper-style API
- :mod:`.diarization` — :func:`check_pyannote_response` for any pyannote-style API
"""

from __future__ import annotations

from .chat import (
    check_chat_response,
    REASON_CHAT_BAD_JSON,
    REASON_CHAT_EMPTY,
    REASON_CHAT_FINISH_LENGTH,
    REASON_CHAT_THINKING_PROSE,
)
from .diarization import check_pyannote_response, REASON_DIARIZATION_EMPTY_SEGMENTS
from .exceptions import GuardrailViolation
from .transcription import (
    check_whisper_response,
    REASON_TRANSCRIPTION_EMPTY,
    REASON_TRANSCRIPTION_LENGTH_FLOOR,
)

__all__ = [
    "GuardrailViolation",
    # Helpers (one per service shape)
    "check_chat_response",
    "check_pyannote_response",
    "check_whisper_response",
    # Reason enum constants — stable strings used as Prometheus label values
    # and as ``GuardrailViolation.reason``. Pinned so tests can assert
    # exact-equality without typo risk.
    "REASON_CHAT_BAD_JSON",
    "REASON_CHAT_EMPTY",
    "REASON_CHAT_FINISH_LENGTH",
    "REASON_CHAT_THINKING_PROSE",
    "REASON_DIARIZATION_EMPTY_SEGMENTS",
    "REASON_TRANSCRIPTION_EMPTY",
    "REASON_TRANSCRIPTION_LENGTH_FLOOR",
]
