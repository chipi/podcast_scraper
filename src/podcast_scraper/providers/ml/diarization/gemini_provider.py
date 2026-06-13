"""Gemini audio-modality diarization provider (#962).

Calls Google's Gemini 2.5 audio API with the audio bytes and a structured-
output prompt that asks for speaker turns as JSON. Used by `cloud_*` profiles
that don't have access to DGX pyannote.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from ....exceptions import ProviderDependencyError
from .base import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)


_DIARIZATION_PROMPT = """You are a speaker-diarization system. Listen to the \
attached audio and identify each speaker turn.

Return ONLY a JSON object with this exact schema (no prose, no markdown):

{{
  "speakers": [
    {{"start": <seconds, float>, "end": <seconds, float>, "speaker": "SPEAKER_00"}},
    ...
  ]
}}

Rules:
- Use stable labels SPEAKER_00, SPEAKER_01, ... — same speaker → same label \
across the whole audio.
- Times are floating-point seconds from the start of the audio.
- One entry per contiguous speaker turn (no overlap with the previous entry).
- {speaker_hint}
"""


def _speaker_hint(num_speakers: Optional[int], min_speakers: int, max_speakers: int) -> str:
    if num_speakers is not None:
        return f"Exactly {num_speakers} distinct speaker(s) are present."
    return (
        f"Between {min_speakers} and {max_speakers} distinct speakers "
        "may be present; choose the smallest number that fits the audio."
    )


def _parse_diarization_json(raw: str) -> List[DiarizationSegment]:
    """Parse the Gemini JSON payload into DiarizationSegments.

    Tolerates leading/trailing whitespace and Markdown ``` fences. Raises
    ValueError on a structurally-bad response so the caller can log and
    fall back rather than silently returning an empty result.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip leading ```json (or just ```) and trailing ```
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini diarization response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or "speakers" not in payload:
        raise ValueError("Gemini diarization response missing the top-level 'speakers' key")
    speakers = payload["speakers"]
    if not isinstance(speakers, list):
        raise ValueError("Gemini diarization 'speakers' field must be a list")
    segments: List[DiarizationSegment] = []
    for entry in speakers:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry["start"])
            end = float(entry["end"])
            speaker = str(entry["speaker"]).strip() or "SPEAKER_UNK"
        except (KeyError, TypeError, ValueError):
            # Skip malformed entries rather than blowing up the whole result —
            # one missing time field shouldn't lose the whole transcript.
            continue
        if end <= start:
            continue
        segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))
    return segments


class GeminiDiarizationProvider:
    """Speaker diarization via Gemini 2.5 audio understanding."""

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
    ) -> None:
        try:
            import google.genai as genai
        except ImportError as exc:  # pragma: no cover - guard rail
            raise ProviderDependencyError(
                message="google-genai is required for Gemini diarization",
                provider="GeminiDiarizationProvider",
                dependency="google-genai",
                suggestion="Install with: pip install google-genai",
            ) from exc
        self._genai = genai
        self.model_name = model_name
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key)

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Send audio to Gemini and parse speaker turns from the response."""
        prompt = _DIARIZATION_PROMPT.format(
            speaker_hint=_speaker_hint(num_speakers, min_speakers, max_speakers)
        )
        # Upload audio via the Files API so we don't blow up the request body.
        file_ref = self.client.files.upload(file=audio_path)
        try:
            config = self._genai.types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
            )
            contents: List[Any] = [file_ref, prompt]
            response: Any = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        finally:
            # Clean up server-side state even on error — Gemini file uploads
            # have a 48h auto-expiry, but we shouldn't rely on it.
            try:
                file_name = getattr(file_ref, "name", None) or ""
                if file_name:
                    self.client.files.delete(name=file_name)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.debug("Gemini file cleanup failed (ignored): %s", exc)

        raw = str(response.text or "").strip()
        if not raw:
            raise ValueError("Gemini diarization returned empty response")
        segments = _parse_diarization_json(raw)
        unique_speakers = {seg.speaker for seg in segments}
        return DiarizationResult(
            segments=segments,
            num_speakers=len(unique_speakers),
            model_name=self.model_name,
        )
