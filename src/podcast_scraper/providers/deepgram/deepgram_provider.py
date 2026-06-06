"""Deepgram cloud transcription provider."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, cast, Dict, List, Tuple

from ... import config
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metrics import _safe_deepgram_retryable, retry_with_metrics
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)


def _response_to_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return cast(Dict[str, Any], response.model_dump())
    if isinstance(response, dict):
        return response
    return {}


def _words_to_segments(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current_speaker = words[0].get("speaker")
    chunk_words: List[Dict[str, Any]] = [words[0]]

    def _flush(chunk: List[Dict[str, Any]]) -> None:
        if not chunk:
            return
        text = " ".join(str(w.get("punctuated_word") or w.get("word") or "") for w in chunk).strip()
        if not text:
            return
        segments.append(
            {
                "start": float(chunk[0].get("start", 0.0)),
                "end": float(chunk[-1].get("end", chunk[-1].get("start", 0.0))),
                "text": text,
                "speaker": chunk[0].get("speaker"),
            }
        )

    for word in words[1:]:
        speaker = word.get("speaker")
        if speaker != current_speaker:
            _flush(chunk_words)
            chunk_words = [word]
            current_speaker = speaker
        else:
            chunk_words.append(word)
    _flush(chunk_words)
    return segments


def parse_deepgram_transcript(response: Any) -> Dict[str, Any]:
    """Convert Deepgram Listen response to pipeline segment dict."""
    data = _response_to_dict(response)
    if not data and response is not None:
        # An unrecognized response shape yields an empty result silently otherwise —
        # surface it so "couldn't parse" isn't mistaken for "empty audio".
        logger.warning("Deepgram response could not be parsed (type=%s)", type(response).__name__)
    results = data.get("results") or {}
    channels = results.get("channels") or []
    transcript = ""
    if channels:
        alternatives = channels[0].get("alternatives") or []
        if alternatives:
            transcript = str(alternatives[0].get("transcript") or "").strip()

    segments: List[Dict[str, Any]] = []
    utterances = results.get("utterances") or []
    for utt in utterances:
        text = str(utt.get("transcript") or "").strip()
        if not text:
            continue
        seg: Dict[str, Any] = {
            "start": float(utt.get("start", 0.0)),
            "end": float(utt.get("end", utt.get("start", 0.0))),
            "text": text,
        }
        if utt.get("speaker") is not None:
            seg["speaker"] = utt.get("speaker")
        segments.append(seg)

    if not segments and channels:
        alternatives = channels[0].get("alternatives") or []
        if alternatives:
            words = alternatives[0].get("words") or []
            segments = _words_to_segments(words)

    if not transcript and segments:
        transcript = " ".join(str(s.get("text", "")) for s in segments).strip()

    return {"text": transcript, "segments": segments}


def _create_deepgram_client(api_key: str, base_url: str | None = None) -> Any:
    """Construct Deepgram SDK client (isolated for unit-test patching).

    When ``base_url`` is set, point the SDK at it (self-hosted / on-prem
    Deepgram, or a test mock server) by overriding the client environment. The
    SDK posts ``{production}/v1/listen``, so every environment URL is set to the
    given root. If the installed SDK lacks the environment override, fall back
    to the default hosted client and warn — the override is opt-in.
    """
    try:
        from deepgram import DeepgramClient
    except ImportError as exc:
        raise RuntimeError(
            "deepgram-sdk is required for transcription_provider='deepgram'. "
            "Install with: pip install -e '.[llm]'"
        ) from exc
    if base_url:
        try:
            from deepgram.environment import DeepgramClientEnvironment

            env = DeepgramClientEnvironment(
                base=base_url, production=base_url, agent=base_url, agent_rest=base_url
            )
            return DeepgramClient(api_key=api_key, environment=env)
        except Exception as exc:  # noqa: BLE001 - override is best-effort
            logger.warning(
                "Could not apply deepgram_api_base=%s (SDK lacks environment override: %s); "
                "using the hosted endpoint.",
                base_url,
                format_exception_for_log(exc),
            )
    return DeepgramClient(api_key=api_key)


class DeepgramTranscriptionProvider:
    """Transcription-only provider using Deepgram Nova models."""

    def __init__(self, cfg: config.Config) -> None:
        """Store config and Deepgram model name."""
        self.cfg = cfg
        self._client: Any = None
        self._initialized = False
        self.model = (cfg.deepgram_model or "nova-3").strip()

    def initialize(self) -> None:
        """Create Deepgram SDK client after validating API key."""
        if self._initialized:
            return
        if not self.cfg.deepgram_api_key:
            raise ValueError(
                "Deepgram API key required for transcription_provider='deepgram'. "
                "Set DEEPGRAM_API_KEY or deepgram_api_key in config."
            )
        self._client = _create_deepgram_client(
            self.cfg.deepgram_api_key,
            base_url=getattr(self.cfg, "deepgram_api_base", None),
        )
        self._initialized = True

    def cleanup(self) -> None:
        """Release SDK client state."""
        self._client = None
        self._initialized = False

    def get_capabilities(self) -> ProviderCapabilities:
        """Return transcription-only capabilities with GI segment timing support."""
        return ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=False,
            supports_summarization=False,
            supports_semantic_cleaning=False,
            supports_audio_input=True,
            supports_json_mode=False,
            max_context_tokens=0,
            provider_name="deepgram",
            supports_gi_segment_timing=True,
        )

    def format_screenplay_from_segments(
        self,
        segments: List[Dict[str, Any]],
        num_speakers: int | None = None,
        speaker_names: List[str] | None = None,
        gap_s: float | None = None,
    ) -> str | None:
        """Render Deepgram's native speaker-labelled segments as a screenplay.

        Deepgram diarizes server-side and tags each segment with an integer
        ``speaker``. Map those to detected names (first-appearance order; falls
        back to ``Speaker N``) and reuse the shared screenplay formatter. Returns
        ``None`` when there is nothing diarized to format so the caller falls back
        to the plain transcript.
        """
        from ..ml.diarization.formatting import format_diarized_screenplay_from_segments

        names = list(speaker_names or [])
        speaker_to_name: Dict[Any, str] = {}
        labelled: List[Dict[str, Any]] = []
        for seg in segments:
            if not isinstance(seg, dict) or not (seg.get("text") or "").strip():
                continue
            speaker = seg.get("speaker")
            if speaker is None:
                label = "Speaker"
            else:
                if speaker not in speaker_to_name:
                    idx = len(speaker_to_name)
                    speaker_to_name[speaker] = (
                        names[idx] if idx < len(names) else f"Speaker {idx + 1}"
                    )
                label = speaker_to_name[speaker]
            enriched = dict(seg)
            enriched["speaker_label"] = label
            labelled.append(enriched)

        if not labelled:
            return None
        return format_diarized_screenplay_from_segments(labelled) or None

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Return transcript text from ``transcribe_with_segments``."""
        result, _elapsed = self.transcribe_with_segments(audio_path, language=language)
        return str(result.get("text") or "")

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Transcribe audio with diarization and return segments plus elapsed seconds."""
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "DeepgramTranscriptionProvider not initialized. Call initialize() first."
            )
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        effective_language = language if language is not None else (self.cfg.language or None)
        started = time.time()
        try:
            with open(audio_path, "rb") as audio_file:
                kwargs: Dict[str, Any] = {
                    "request": audio_file.read(),
                    "model": self.model,
                    "diarize": True,
                    "smart_format": True,
                    "utterances": True,
                    "punctuate": True,
                }
                if effective_language:
                    kwargs["language"] = effective_language
                # Retry transient API/network failures with backoff, like every
                # sibling provider (Deepgram rate limits are real).
                response = retry_with_metrics(
                    lambda: self._client.listen.v1.media.transcribe_file(**kwargs),
                    retryable_exceptions=_safe_deepgram_retryable(),
                    metrics=call_metrics,
                    error_context="deepgram",
                )
        except Exception as exc:
            logger.error("Deepgram transcription failed: %s", format_exception_for_log(exc))
            raise

        elapsed = time.time() - started
        result = parse_deepgram_transcript(response)

        if call_metrics is not None:
            call_metrics.set_provider_name("deepgram")

        self._record_transcription_cost(
            audio_path, episode_duration_seconds, pipeline_metrics, call_metrics
        )

        logger.info(
            "Deepgram transcription completed in %.2fs (%d segments)",
            elapsed,
            len(result.get("segments") or []),
        )
        return result, elapsed

    def _record_transcription_cost(
        self,
        audio_path: str,
        episode_duration_seconds: int | None,
        pipeline_metrics: Any | None,
        call_metrics: Any | None,
    ) -> None:
        """Record per-minute transcription cost (D5).

        Deepgram bills per audio-minute. Prefer the precise feed duration
        (``episode_duration_seconds``); fall back to a bitrate-aware file-size
        estimate so cost isn't silently 0 when the caller doesn't pass it —
        mirrors the gemini/mistral providers. The orchestration layer's
        ``apply_estimated_cost_if_missing`` is a backstop; recording here keeps
        Deepgram symmetric and precise when the feed duration is known.
        """
        audio_minutes = 0.0
        if episode_duration_seconds is not None:
            audio_minutes = float(episode_duration_seconds) / 60.0
        else:
            try:
                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                bitrate_kbps_cfg = getattr(self.cfg, "preprocessing_mp3_bitrate_kbps", None)
                bitrate_kbps = float(bitrate_kbps_cfg) if bitrate_kbps_cfg else 128.0
                audio_minutes = file_size_mb * (128.0 / bitrate_kbps)
            except OSError:
                pass

        if audio_minutes <= 0:
            return

        from ...utils.provider_metrics import record_provider_call_cost
        from ...workflow.helpers import calculate_provider_cost

        cost = calculate_provider_cost(
            cfg=self.cfg,
            provider_type="deepgram",
            capability="transcription",
            model=self.model,
            audio_minutes=audio_minutes,
        )
        if call_metrics is not None:
            record_provider_call_cost(
                call_metrics,
                cost,
                cfg=self.cfg,
                provider_type="deepgram",
                capability="transcription",
                model=self.model,
                audio_minutes=audio_minutes,
            )
        if pipeline_metrics is not None:
            estimated = getattr(call_metrics, "estimated_cost", cost) if call_metrics else cost
            pipeline_metrics.record_llm_transcription_call(audio_minutes, cost_usd=estimated)
