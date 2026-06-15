"""Real-pyannote diarization integration test (RFC-058).

This is the missing end-to-end check the RFC called for: run the *actual* pyannote
pipeline on a known two-voice TTS fixture and confirm it separates the host and
guest. Every other diarization test mocks pyannote, so they prove the wiring but
not that diarization works on audio.

Skipped unless ``pyannote.audio`` is installed (the ``[ml]`` extra) AND a
HuggingFace token is available — ``pyannote/speaker-diarization-3.1`` is a gated
model. Run with::

    HF_TOKEN=hf_... .venv/bin/pytest -m diarization tests/integration/providers/ml/test_diarization.py

Ground truth (tests/fixtures/FIXTURES_SPEC.md): ``p01_multi_e01`` is podcast p01
"Singletrack Sessions" — host **Maya** (TTS voice Samantha) and guest **Liam**
(TTS voice Daniel) — i.e. two distinct speakers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.ml_models, pytest.mark.diarization]

# pyannote.audio ships in the [ml] extra, not [dev]. Skip if it can't be imported
# *for any reason* — a version-mismatched install (e.g. pyannote 3.4 against
# torchaudio>=2.9, which removed AudioMetaData) raises AttributeError, not
# ImportError, so plain importorskip would error collection instead of skipping.
try:
    import pyannote.audio  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"pyannote.audio unavailable: {exc}", allow_module_level=True)

# Audio fixtures are versioned (#902). Use v2 (the current default): its host (female,
# ~178Hz) and guest (male, ~117Hz) voices are acoustically distinct (cross-voice embedding
# cosine ~0.22) and pyannote separates them deterministically. An earlier note pinned this
# to v1 claiming v2 "doesn't separate" (#921); that did not reproduce — v1 and v2 both
# diarize into two clean voices. The old non-versioned path no longer exists.
_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "audio" / "v2" / "p01_multi_e01.mp3"


def _hf_token_available() -> bool:
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"):
        return True
    return (Path.home() / ".huggingface" / "token").is_file()


pytestmark.append(
    pytest.mark.skipif(
        not _hf_token_available(),
        reason="no HuggingFace token (pyannote/speaker-diarization-3.1 is gated)",
    )
)


@pytest.mark.serial
def test_pyannote_separates_two_voices() -> None:
    """pyannote must detect both speakers in the two-voice host+guest fixture."""
    assert _FIXTURE.is_file(), f"missing fixture: {_FIXTURE}"

    from podcast_scraper import config
    from podcast_scraper.providers.ml.diarization.factory import create_diarization_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
    )
    # The suite runs with HF_HUB_OFFLINE=1 + sockets blocked, so the gated model must
    # already be in the active HF cache (run ``make preload-ml-models`` with a token).
    # Skip — don't fail — when it isn't, distinguishing "not provisioned" from a real
    # diarization regression.
    try:
        provider = create_diarization_provider(cfg)
        result = provider.diarize(str(_FIXTURE), min_speakers=1, max_speakers=5)
    except Exception as exc:  # noqa: BLE001 - provisioning vs regression
        haystack = f"{exc} {type(exc).__name__}".lower()
        provisioning = (
            "offlinemode",
            "offline mode",
            "gatedrepo",
            "localentrynotfound",
            "local cache",
            "cannot reach",
            "socketblocked",
            "connection",
        )
        if any(k in haystack for k in provisioning):
            pytest.skip(f"pyannote model not in offline cache (preload it): {type(exc).__name__}")
        raise

    assert result.segments, "pyannote returned no speaker turns"
    # Two distinct TTS voices (Maya/Samantha + Liam/Daniel) -> at least two speakers.
    assert (
        result.num_speakers >= 2
    ), f"expected >= 2 speakers for the host+guest fixture, got {result.num_speakers}"
    # Turns should cover a meaningful span of the audio, not a single blip.
    covered = sum(seg.end - seg.start for seg in result.segments)
    assert covered > 1.0, f"diarization covered only {covered:.2f}s of audio"
