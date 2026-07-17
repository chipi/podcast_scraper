"""Deepgram cloud diarization provider.

The cloud-only path for ``diarize: true`` when the pipeline host can't run
pyannote (no ``[ml]`` extras, no HF_TOKEN). POSTs the audio to Deepgram's
Listen API with ``diarize=true`` and parses speaker turns from the
response — no local install required.

When paired with ``transcription_provider: deepgram`` this means two
Deepgram API calls per episode (one for transcription, one for
diarization). That's a deliberate tradeoff for stage independence — same
shape as pairing OpenAI Whisper with local pyannote on an ML-capable
host. A "reuse transcription response" optimization would require the
orchestrator to thread the transcription response into the diarization
stage, which crosses the provider-interface boundary; not done here.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from .base import DiarizationProvider, DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)


class DeepgramDiarizationProvider(DiarizationProvider):
    """Diarization-only wrapper around Deepgram's Listen API."""

    def __init__(
        self,
        api_key: str,
        model: str = "nova-3-general",
        api_base: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Deepgram API key required for diarization_provider='deepgram'. "
                "Set DEEPGRAM_API_KEY or deepgram_api_key in config."
            )
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self._client: Any = None

    def initialize(self) -> None:
        """Construct the Deepgram SDK client.

        Mirrors the construction shape used by the transcription
        provider so the SDK-environment override (for testing against
        the e2e mock server) works the same way.
        """
        try:
            from deepgram import DeepgramClient
        except ImportError as exc:  # pragma: no cover - missing dep is user error
            raise RuntimeError(
                "deepgram-sdk not installed. Install with the deepgram extras."
            ) from exc

        if self.api_base:
            try:
                from deepgram.environment import (  # type: ignore[import-untyped]
                    DeepgramClientEnvironment,
                )

                # Mirror the transcription provider's env EXACTLY — the SDK routes
                # the /v1/listen call off ``base``/``production``; the old
                # ``nightly``/``legacy`` kwargs aren't valid fields, so this raised
                # and silently fell back to a no-base client hitting real Deepgram
                # (401 in CI, where DEEPGRAM_API_BASE points at the E2E stub).
                env = DeepgramClientEnvironment(  # type: ignore[call-arg]
                    base=self.api_base,
                    production=self.api_base,
                    agent=self.api_base,
                    agent_rest=self.api_base,
                )
                self._client = DeepgramClient(api_key=self.api_key, environment=env)
            except Exception as exc:  # noqa: BLE001 - SDK shape can vary
                # FAIL LOUD when a base was configured but can't be applied — never
                # silently fall back to real Deepgram. Silent fallback here is what
                # sent CI to the hosted endpoint (401); in prod it leaks audio + spend.
                raise RuntimeError(
                    f"deepgram_api_base={self.api_base!r} is set but could not be applied "
                    f"to the Deepgram SDK ({exc}); refusing to fall back to the hosted "
                    "endpoint."
                ) from exc
        else:
            self._client = DeepgramClient(api_key=self.api_key)

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """POST the audio to Deepgram and return speaker turns."""
        if self._client is None:
            self.initialize()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        started = time.time()
        with open(audio_path, "rb") as audio_file:
            kwargs: Dict[str, Any] = {
                "request": audio_file.read(),
                "model": self.model,
                "diarize": True,
                # We don't need utterances / smart_format for the diarize-only
                # path — speaker turns come from the words[].speaker field.
                "punctuate": False,
                "smart_format": False,
            }
            response = self._client.listen.v1.media.transcribe_file(**kwargs)

        elapsed = time.time() - started

        segments = self._extract_speaker_turns(response)
        num_speakers_detected = len({s.speaker for s in segments})
        logger.info(
            "Deepgram diarization completed in %.2fs (%d segments, %d speakers)",
            elapsed,
            len(segments),
            num_speakers_detected,
        )
        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers_detected,
            model_name=f"deepgram/{self.model}",
        )

    @staticmethod
    def _extract_speaker_turns(response: Any) -> List[DiarizationSegment]:
        """Parse Deepgram words[].speaker into contiguous DiarizationSegments."""
        try:
            if hasattr(response, "results"):
                results = response.results
            elif isinstance(response, dict):
                results = response.get("results", {})
            else:
                results = {}
            if hasattr(results, "channels"):
                channels = results.channels
            else:
                channels = results.get("channels") if isinstance(results, dict) else []
            if not channels:
                return []
            first_channel = channels[0]
            if hasattr(first_channel, "alternatives"):
                alternatives = first_channel.alternatives
            else:
                alternatives = (
                    first_channel.get("alternatives", []) if isinstance(first_channel, dict) else []
                )
            if not alternatives:
                return []
            first_alt = alternatives[0]
            if hasattr(first_alt, "words"):
                words = first_alt.words
            else:
                words = first_alt.get("words", []) if isinstance(first_alt, dict) else []
        except Exception as exc:  # noqa: BLE001 - defensive shape parsing
            logger.warning(
                "Deepgram diarization response could not be parsed (type=%s): %s",
                type(response).__name__,
                exc,
            )
            return []

        if not words:
            return []

        # Group consecutive words by speaker into DiarizationSegments.
        segments: List[DiarizationSegment] = []
        current_speaker: Optional[Any] = None
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        for word in words:
            w_speaker = (
                word.get("speaker") if isinstance(word, dict) else getattr(word, "speaker", None)
            )
            w_start = word.get("start") if isinstance(word, dict) else getattr(word, "start", None)
            w_end = word.get("end") if isinstance(word, dict) else getattr(word, "end", None)
            if w_start is None or w_end is None or w_speaker is None:
                continue
            if w_speaker != current_speaker:
                if current_speaker is not None:
                    segments.append(
                        DiarizationSegment(
                            start=float(current_start or 0.0),
                            end=float(current_end or 0.0),
                            speaker=f"SPEAKER_{int(current_speaker):02d}",
                        )
                    )
                current_speaker = w_speaker
                current_start = float(w_start)
            current_end = float(w_end)
        if current_speaker is not None:
            segments.append(
                DiarizationSegment(
                    start=float(current_start or 0.0),
                    end=float(current_end or 0.0),
                    speaker=f"SPEAKER_{int(current_speaker):02d}",
                )
            )
        return segments
